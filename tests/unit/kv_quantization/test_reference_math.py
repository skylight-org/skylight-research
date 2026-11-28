"""Independent numerical oracles for quantization formats and level tables."""

import math

import pytest
import torch

from sparse_attention_hub.kv_quantization import (
    FakeNVFP4Quantizer,
    FakeNVFP4QuantizerConfig,
    FP8Quantizer,
    FP8QuantizerConfig,
    ParamDType,
    QuantGranularity,
)

pytestmark = pytest.mark.unit
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("granularity", list(QuantGranularity))
def test_nvfp4_matches_independent_scalar_scaling_rounding_and_packing(
    dtype, granularity
):
    """No production grouping, rounding, pack or decode helper is used by the oracle."""
    x = torch.randn(2, 2, 33, 32, generator=torch.Generator().manual_seed(429)).to(
        dtype
    )
    x[0, 0, 0] = 0
    x[1, 0] *= 20
    axis = 2 if granularity == QuantGranularity.PER_CHANNEL else 3
    ordered = x.float().movedim(axis, -1).contiguous()
    length = ordered.shape[-1]
    padded = math.ceil(length / 16) * 16
    rows = ordered.reshape(-1, length)
    expanded = torch.cat([rows, rows[:, -1:].expand(-1, padded - length)], dim=-1)
    groups = expanded.reshape(-1, 16)
    global_scale = (groups.abs().max() / (6 * 448)).float()
    if global_scale == 0:
        global_scale = torch.tensor(1.0)
    scale_codes, integer_codes, reconstructed = [], [], []
    for group in groups:
        local = ((group.abs().max() / (6 * global_scale)).clamp(max=448)).to(
            torch.float8_e4m3fn
        )
        scale_codes.append(local.view(torch.uint8).item())
        scale = local.float() * global_scale
        scaled = group / (scale if scale > 0 else 1)
        decoded = []
        for value in scaled.tolist():
            index = min(
                range(8),
                key=lambda i: (abs(abs(value) - E2M1_MAGNITUDES[i]), i % 2),
            )
            code = index + (8 if value < 0 and index != 0 else 0)
            integer_codes.append(code)
            decoded.append((-1 if code >= 8 else 1) * E2M1_MAGNITUDES[index])
        reconstructed.append(torch.tensor(decoded) * scale)
    expected_bytes = [
        integer_codes[i] | (integer_codes[i + 1] << 4)
        for i in range(0, len(integer_codes), 2)
    ]
    expected = torch.stack(reconstructed).reshape(rows.shape[0], padded)[:, :length]
    expected = expected.reshape(ordered.shape).movedim(-1, axis).to(dtype)
    codec = FakeNVFP4Quantizer(FakeNVFP4QuantizerConfig(granularity=granularity))
    record = codec.quantize(x)
    assert record.codes.tolist() == expected_bytes
    assert record.scales.flatten().tolist() == scale_codes
    assert torch.equal(record.global_scale, global_scale.reshape(1))
    assert torch.equal(codec.dequantize(record), expected)
    assert record.storage_bytes() == len(expected_bytes) + len(scale_codes) + 4


@pytest.mark.parametrize("format,maximum", [("E4M3", 448.0), ("E5M2", 57344.0)])
def test_fp8_all_positive_levels_and_midpoints_obey_round_to_nearest_even(
    format, maximum
):
    """A max-value anchor fixes the scale to one; enumerate the format analytically."""
    exponent_bits, mantissa_bits, bias = (4, 3, 7) if format == "E4M3" else (5, 2, 15)
    positive = []
    for exponent in range(2**exponent_bits):
        for mantissa in range(2**mantissa_bits):
            if format == "E5M2" and exponent == 31:
                continue
            if format == "E4M3" and exponent == 15 and mantissa == 7:
                continue
            value = (
                mantissa / 2**mantissa_bits * 2.0 ** (1 - bias)
                if exponent == 0
                else (1 + mantissa / 2**mantissa_bits) * 2.0 ** (exponent - bias)
            )
            positive.append(value)
    assert positive[-1] == maximum
    data, expected = list(positive), list(positive)
    for index, (left, right) in enumerate(zip(positive, positive[1:])):
        data.append((left + right) / 2)
        expected.append(left if index % 2 == 0 else right)
    x = torch.tensor(data).reshape(1, 1, 1, -1)
    codec = FP8Quantizer(
        FP8QuantizerConfig(
            format=format, group_size=x.numel(), scale_dtype=ParamDType.FP32
        )
    )
    record = codec.quantize(x)
    assert record.scales.item() == 1
    assert torch.equal(codec.dequantize(record).flatten(), torch.tensor(expected))


@pytest.mark.parametrize("bits", [1, 2, 4, 8])
def test_gaussian_levels_are_symmetric_and_match_known_low_bit_centroids(bits):
    """Validate every table; independently check the known one/two-bit centroids."""
    import math

    from sparse_attention_hub.kv_quantization.quantizers.implementations.non_uniform import (
        lloyd_max_gaussian,
    )

    levels = lloyd_max_gaussian(bits).double()
    assert torch.isfinite(levels).all() and (levels[1:] > levels[:-1]).all()
    torch.testing.assert_close(levels, -levels.flip(0), atol=1e-6, rtol=0)
    if bits == 1:
        assert levels[1].item() == pytest.approx(math.sqrt(2 / math.pi), rel=1e-6)
    if bits == 2:
        torch.testing.assert_close(
            levels,
            torch.tensor([-1.510418, -0.45278, 0.45278, 1.510418]).double(),
            atol=3e-6,
            rtol=0,
        )
