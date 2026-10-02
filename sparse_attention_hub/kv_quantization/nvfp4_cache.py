"""Hardware-aware NVFP4 quantized KV cache.

Stores each layer's KV cache in the NVFP4 format that Blackwell-class hardware
consumes natively: E2M1 4-bit elements, one E4M3 scale per block of
``block_size`` elements, plus a single FP32 per-tensor scale. Quantization is
delegated to ``nvidia-modelopt``'s :class:`NVFP4QTensor`, NVIDIA's reference
implementation. The simulator in :mod:`nvfp4` follows the same rules and
reproduces it bit for bit on a GPU.

The most recent ``residual_length`` tokens are kept unquantized (inherited
``QuantizedLayer`` behaviour), which is what keeps decoding accurate: those are
the tokens whose keys/values are still being written a few elements at a time,
where block-wise scales have too little data to be meaningful.

For the same accuracy behaviour without modelopt or a GPU, and without the
memory saving, see :mod:`fake_nvfp4_cache`.

Example:
    Used through the adapter flag rather than directly::

        adapter = ModelAdapterHF(
            model_name="Qwen/Qwen3-4B",
            sparse_attention_config=config,
            quantize_kv_cache="nvfp4",
            kv_quantization_kwargs={"axis_key": 0},
        )
"""

from dataclasses import dataclass
from typing import Any

import torch

from .constants import (
    AXIS_PER_CHANNEL,
    AXIS_PER_TOKEN,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_RESIDUAL_LENGTH,
    VALID_AXES,
)
from .layers import QuantizePrefillMixin, build_cache_layers

try:
    from transformers.cache_utils import Cache, QuantizedLayer
except ImportError as exc:  # pragma: no cover - depends on the installed transformers
    raise ImportError(
        "NVFP4 KV cache quantization needs the layer-based cache API from "
        "transformers>=5.0 (transformers.cache_utils.QuantizedLayer). Upgrade "
        "transformers, or construct the adapter with quantize_kv_cache=False."
    ) from exc

try:
    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "NVFP4 KV cache quantization needs nvidia-modelopt for the hardware "
        "NVFP4 format. Install it with `pip install nvidia-modelopt`, or pass "
        'quantize_kv_cache="fake_nvfp4" for the torch-only path.'
    ) from exc


@dataclass
class NVFP4Quantized:
    """A packed NVFP4 tensor together with everything needed to invert it.

    ``NVFP4QTensor`` holds only the packed codes; the two scale tensors and the
    block size live outside it, so dequantization needs them handed back.

    Attributes:
        qtensor: Packed E2M1 codes.
        scale: Per-block E4M3 scales.
        scale2: Per-tensor FP32 scale.
        block_size: Elements per block along the quantized axis.
        dtype: Dtype to restore on dequantization.
        transposed: Whether the tensor was transposed before quantizing, i.e.
            whether blocking ran over the token axis instead of ``head_dim``.
        length: Length of the quantized (last) axis before zero-padding to a
            multiple of ``block_size``; dequantization trims back to it.
    """

    qtensor: NVFP4QTensor
    scale: torch.Tensor
    scale2: torch.Tensor
    block_size: int
    dtype: torch.dtype
    transposed: bool
    length: int


class NVFP4QuantizedLayer(QuantizePrefillMixin, QuantizedLayer):
    """One layer's KV cache, physically packed as NVFP4 past ``residual_length``.

    Keys and values get independent axis choices because their outlier
    structure differs: keys carry persistent large-magnitude channels (the
    attention-sink dimensions), so a scale shared across channels is set by an
    outlier and degrades every well-behaved channel in the block. Values are
    roughly homogeneous across channels, with the occasional loud token.
    """

    def __init__(
        self,
        block_size: int = DEFAULT_BLOCK_SIZE,
        residual_length: int = DEFAULT_RESIDUAL_LENGTH,
        axis_key: int = AXIS_PER_TOKEN,
        axis_value: int = AXIS_PER_TOKEN,
        quantize_prefill: bool = False,
    ) -> None:
        """Initialize the layer.

        Args:
            block_size: Elements sharing one E4M3 scale.
            residual_length: Trailing tokens kept unquantized.
            axis_key: ``-1`` for per-token blocking, ``0`` for per-channel.
            axis_value: Same, for values.
            quantize_prefill: See ``QuantizePrefillMixin``.

        Raises:
            ValueError: If either axis is not ``-1`` or ``0``.
        """
        for name, axis in (("axis_key", axis_key), ("axis_value", axis_value)):
            if axis not in VALID_AXES:
                raise ValueError(
                    f"{name!r} must be {AXIS_PER_TOKEN} (per-token) or "
                    f"{AXIS_PER_CHANNEL} (per-channel, KIVI-style), got {axis}"
                )
        super().__init__(
            nbits=4,
            axis_key=axis_key,
            axis_value=axis_value,
            q_group_size=block_size,
            residual_length=residual_length,
        )
        self.block_size: int = block_size
        self.quantize_prefill: bool = quantize_prefill

    def _quantize(self, tensor: torch.Tensor, axis: int) -> NVFP4Quantized:
        """Pack a KV tensor into NVFP4.

        Args:
            tensor: Shape ``(batch, heads, tokens, head_dim)``.
            axis: ``-1`` to block over ``head_dim``, ``0`` over tokens.

        Returns:
            An :class:`NVFP4Quantized` holding the packed codes, both scales and
            what :meth:`_dequantize` needs to restore the original layout
            (transpose flag, unpadded length).
        """
        transposed: bool = axis == AXIS_PER_CHANNEL
        if transposed:
            tensor = tensor.transpose(-1, -2).contiguous()
        # Per-channel blocking runs along tokens, whose count is rarely a
        # multiple of the block size. modelopt pads when quantizing but its
        # dequantize reshapes to the unpadded shape and fails, so pad here
        # (zeros don't change any block's amax) and trim in _dequantize.
        length: int = tensor.shape[-1]
        padding: int = (-length) % self.block_size
        if padding:
            tensor = torch.nn.functional.pad(tensor, (0, padding))
        qtensor, scale, scale2 = NVFP4QTensor.quantize(tensor, self.block_size)
        return NVFP4Quantized(
            qtensor=qtensor,
            scale=scale,
            scale2=scale2,
            block_size=self.block_size,
            dtype=tensor.dtype,
            transposed=transposed,
            length=length,
        )

    def _dequantize(self, qtensor: NVFP4Quantized) -> torch.Tensor:
        """Unpack NVFP4 back to ``qtensor.dtype``.

        Args:
            qtensor: Output of :meth:`_quantize`.

        Returns:
            Tensor of shape ``(batch, heads, tokens, head_dim)``.
        """
        out: torch.Tensor = qtensor.qtensor.dequantize(
            dtype=qtensor.dtype,
            scale=qtensor.scale,
            double_scale=qtensor.scale2,
            block_sizes={AXIS_PER_TOKEN: qtensor.block_size},
        )[..., : qtensor.length]
        return out.transpose(-1, -2) if qtensor.transposed else out


class NVFP4QuantizedCache(Cache):
    """A :class:`Cache` whose per-layer KV storage is packed NVFP4."""

    def __init__(
        self,
        config: Any,
        block_size: int = DEFAULT_BLOCK_SIZE,
        residual_length: int = DEFAULT_RESIDUAL_LENGTH,
        axis_key: int = AXIS_PER_TOKEN,
        axis_value: int = AXIS_PER_TOKEN,
        quantize_prefill: bool = False,
    ) -> None:
        """Build one quantized layer per full-attention decoder layer.

        Args:
            config: The model's ``PreTrainedConfig``.
            block_size: Elements sharing one E4M3 scale.
            residual_length: Trailing tokens kept unquantized.
            axis_key: Blocking axis for keys.
            axis_value: Blocking axis for values.
            quantize_prefill: Return quantized KV from the prefill call too
                (for single-pass perplexity); see ``QuantizePrefillMixin``.

        Raises:
            ValueError: If any decoder layer is not full attention
                (sliding-window, chunked or linear-attention layers have their
                own cache layouts that this does not implement).
        """
        super().__init__(
            layers=build_cache_layers(
                config,
                lambda _layer_idx: NVFP4QuantizedLayer(
                    block_size=block_size,
                    residual_length=residual_length,
                    axis_key=axis_key,
                    axis_value=axis_value,
                    quantize_prefill=quantize_prefill,
                ),
                cache_name="NVFP4QuantizedCache",
            )
        )
