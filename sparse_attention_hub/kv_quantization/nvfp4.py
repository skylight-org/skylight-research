"""NVFP4 fake quantization in pure torch.

Fake quantization means quantize and immediately dequantize, returning a tensor
of the original dtype and shape. Nothing is stored packed, so there is no memory
saving — what you get is the exact accuracy loss the format imposes, on any
device, with no nvidia-modelopt and no Blackwell silicon. Use it to measure what
NVFP4 costs a model; use the real backend to actually save memory.

The block scales follow nvidia-modelopt's rule, which is what this package's
real ``nvfp4`` backend uses, so on a GPU the output is bit-identical to it
(verified by the heavy GPU suite on molab):

- tensor scale ``amax / (6 * 448)``;
- block scale ``block_amax / (6 * tensor_scale)``, clamped to ``[2^-9, 448]``
  (2^-9 is the smallest E4M3 *subnormal*) and rounded to E4M3; an all-zero
  block gets scale 1;
- codes from a true division ``x / (block_scale * tensor_scale)``, rounded
  to E2M1.

The scale arithmetic runs in float32. float64 is not a free accuracy upgrade
here: a scale sitting near an E4M3 rounding tie flips to the other side, which
changes the block scale and every code under it.
"""

from typing import Tuple

import torch

from .constants import DEFAULT_BLOCK_SIZE, E2M1_MAX, E4M3_MAX, E4M3_MIN_SUBNORMAL

_FP32_MANTISSA_BITS: int = 23
_FP32_EXPONENT_MASK: int = 0xFF
_FP32_EXPONENT_BIAS: int = 127


def _floor_log2(magnitude: torch.Tensor) -> torch.Tensor:
    """``floor(log2(x))`` for positive float32 values, read exactly from the bits.

    A float32 stores its exponent in bits 23-30 with a bias of 127, so this is
    exact on every device and vendor. ``torch.log2`` is an approximate maths
    function whose result can differ between implementations (e.g. 2.9999999
    for an exact 8.0), and ``torch.frexp`` isn't implemented on Apple MPS;
    a shift and a mask are supported everywhere. Zeros and float32 subnormals
    come out as -127, which callers clamp to the target format's minimum anyway.
    """
    bits: torch.Tensor = magnitude.contiguous().view(torch.int32)
    return ((bits >> _FP32_MANTISSA_BITS) & _FP32_EXPONENT_MASK) - _FP32_EXPONENT_BIAS


def _pow2(exponent: torch.Tensor) -> torch.Tensor:
    """Exact ``2.0 ** exponent`` as float32 for integer exponents in [-126, 127].

    Builds the float directly from its bits instead of calling ``torch.exp2``,
    for the same reason as :func:`_floor_log2`.
    """
    return (
        ((exponent + _FP32_EXPONENT_BIAS) << _FP32_MANTISSA_BITS)
        .to(torch.int32)
        .view(torch.float32)
    )


def _round_to_minifloat(
    x: torch.Tensor, e_bits: int, m_bits: int, bias: int, max_val: float
) -> torch.Tensor:
    """Round magnitudes to the nearest value a small float format can hold.

    Covers normals and the subnormal ramp near zero, rounds half to even the
    way the hardware does, and saturates out-of-range inputs instead of sending
    them to infinity. Uses only exact operations (bit shifts, float32 scaling by
    powers of two, round-half-to-even), so the result is bit-identical on every
    device.

    Args:
        x: Input tensor; sign is preserved, magnitude is rounded.
        e_bits: Exponent bits.
        m_bits: Mantissa bits.
        bias: Exponent bias.
        max_val: Largest representable magnitude.

    Returns:
        Tensor of the same shape and dtype, holding only representable values.
    """
    magnitude: torch.Tensor = x.abs().clamp(max=max_val).to(torch.float32)

    max_exp: int = (1 << e_bits) - 1 - bias
    min_exp: int = 1 - bias  # below this the step stops shrinking

    exponent: torch.Tensor = _floor_log2(magnitude).clamp(min_exp, max_exp)

    step: torch.Tensor = _pow2(exponent - m_bits)
    quantized: torch.Tensor = torch.round(magnitude / step) * step
    quantized = torch.where(
        magnitude == 0, torch.zeros_like(quantized), quantized
    ).clamp(max=max_val)

    # Grid values have at most 4 significant bits, so the cast back is exact.
    return torch.sign(x) * quantized.to(x.dtype)


def q_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round to the 4-bit element grid {0, .5, 1, 1.5, 2, 3, 4, 6}."""
    return _round_to_minifloat(x, e_bits=2, m_bits=1, bias=1, max_val=E2M1_MAX)


def q_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round to E4M3, the 8-bit format NVFP4 uses for its per-block scale."""
    return _round_to_minifloat(x, e_bits=4, m_bits=3, bias=7, max_val=E4M3_MAX)


def _to_blocks(x: torch.Tensor, block_size: int) -> Tuple[torch.Tensor, int]:
    """Split the last axis into blocks, zero-padding if it does not divide.

    Args:
        x: Input tensor.
        block_size: Elements per block.

    Returns:
        The blocked tensor of shape ``(..., n_blocks, block_size)`` and the
        original length of the last axis.
    """
    length: int = x.shape[-1]
    padding: int = (-length) % block_size
    if padding:
        x = torch.nn.functional.pad(x, (0, padding))
    return x.reshape(x.shape[:-1] + (x.shape[-1] // block_size, block_size)), length


def _from_blocks(blocked: torch.Tensor, length: int) -> torch.Tensor:
    """Undo :func:`_to_blocks`, dropping any padding."""
    flat: torch.Tensor = blocked.reshape(blocked.shape[:-2] + (-1,))
    return flat[..., :length]


def quantize_nvfp4(
    x: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize along the last axis and return NVFP4's three parts, unpacked.

    ``x ≈ codes * block_scale * tensor_scale``. This is what the packed format
    stores (4-bit codes, one E4M3 scale per block, one FP32 scale per tensor),
    kept as float32 tensors so each part can be inspected and compared with
    other implementations.

    Args:
        x: Tensor to quantize; the last axis is blocked (zero-padded to a
            multiple of ``block_size``).
        block_size: Elements sharing one E4M3 scale.

    Returns:
        ``(codes, block_scale, tensor_scale)``: ``codes`` has shape
        ``(..., n_blocks, block_size)`` with values on the E2M1 grid,
        ``block_scale`` ``(..., n_blocks, 1)`` with E4M3 values, and
        ``tensor_scale`` a float32 scalar. For an all-zero ``x`` every part is 0.
    """
    x32: torch.Tensor = x.to(torch.float32)
    blocked, _ = _to_blocks(x32, block_size)
    block_amax: torch.Tensor = blocked.abs().amax(dim=-1, keepdim=True)

    tensor_amax: torch.Tensor = x32.abs().amax()
    if tensor_amax == 0:
        zeros: torch.Tensor = torch.zeros_like(blocked)
        return zeros, torch.zeros_like(block_amax), tensor_amax

    # Dividing by both maxima is what lets the E4M3 scales use their full range.
    # Drop the E2M1 max and the largest block scale only ever reaches 448/6.
    tensor_scale: torch.Tensor = tensor_amax / (E4M3_MAX * E2M1_MAX)

    # Clamp before the E4M3 rounding: without the lower bound a block whose amax
    # is tiny next to the tensor amax would get scale 0 and divide by zero.
    block_scale: torch.Tensor = q_e4m3(
        (block_amax / (E2M1_MAX * tensor_scale)).clamp(E4M3_MIN_SUBNORMAL, E4M3_MAX)
    )
    block_scale = torch.where(
        block_amax == 0, torch.ones_like(block_scale), block_scale
    )
    codes: torch.Tensor = q_e2m1(blocked / (block_scale * tensor_scale))
    codes = torch.where(block_amax == 0, torch.zeros_like(codes), codes)
    return codes, block_scale, tensor_scale


def fake_quantize_nvfp4(
    x: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """Round trip a tensor through NVFP4 along its last axis.

    One E4M3 scale per block of ``block_size`` elements, plus a single FP32
    per-tensor scale. To block over a different axis, transpose before calling.

    Args:
        x: Tensor to quantize. Any dtype and shape; the last axis is blocked.
        block_size: Elements sharing one E4M3 scale.

    Returns:
        A tensor of the input's shape and dtype, holding only values NVFP4 can
        represent.
    """
    if x.to(torch.float32).abs().amax() == 0:
        return x.clone()
    codes, block_scale, tensor_scale = quantize_nvfp4(x, block_size)
    dequantized: torch.Tensor = codes * (block_scale * tensor_scale)
    return _from_blocks(dequantized, x.shape[-1]).to(x.dtype)


def bits_per_element(numel: int, block_size: int = DEFAULT_BLOCK_SIZE) -> float:
    """Average bits per element the real (packed) format would use.

    Fake quantization stores nothing packed, so this is what the accuracy loss
    would buy you on hardware: 4 bits per element, an 8-bit scale per block, and
    one 32-bit per-tensor scale.

    Args:
        numel: Number of elements in the tensor.
        block_size: Elements sharing one E4M3 scale.

    Returns:
        Bits per element.
    """
    return 4.0 + 8.0 / block_size + 32.0 / numel
