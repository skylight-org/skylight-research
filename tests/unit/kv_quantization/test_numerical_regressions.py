"""Independent packing, numerical and calibration regression checks."""

import pytest
import torch

from sparse_attention_hub.kv_quantization import (
    FP8QuantizerConfig,
    IntQuantizer,
    IntQuantizerConfig,
    KVQuantizer,
    NonUniformQuantizer,
    NonUniformQuantizerConfig,
    ParamDType,
    PrecisionPlan,
    QuantGranularity,
)
from sparse_attention_hub.kv_quantization.context import AttentionContext
from sparse_attention_hub.kv_quantization.quantizers.common import (
    pack_codes,
    unpack_codes,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("bits", [1, 2, 4, 8])
def test_bit_packing_against_independent_python_byte_oracle(bits):
    values = list(range(2**bits)) + [1, 0, 1]
    packed, count = pack_codes(torch.tensor(values), bits)
    expected = []
    per_byte = 8 // bits
    for start in range(0, len(values), per_byte):
        byte = 0
        for shift, value in enumerate(values[start : start + per_byte]):
            byte |= value << (shift * bits)
        expected.append(byte)
    assert packed.tolist() == expected
    assert count == len(values)
    assert unpack_codes(packed, bits, count).tolist() == values


def test_known_two_bit_affine_values_without_using_encoder_as_oracle():
    x = torch.tensor([-1.0, -0.2, 0.4, 1.0]).reshape(1, 1, 1, 4)
    codec = IntQuantizer(
        IntQuantizerConfig(
            bits=2,
            group_size=4,
            scale_dtype=ParamDType.FP32,
            offset_dtype=ParamDType.FP32,
        )
    )
    encoded = codec.quantize(x)
    assert encoded.codes.tolist() == [228]
    torch.testing.assert_close(
        codec.dequantize(encoded).flatten(), torch.tensor([-1.0, -1 / 3, 1 / 3, 1.0])
    )


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("granularity", list(QuantGranularity))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("symmetric", [False, True])
def test_integer_roundtrip_multiple_batches_partial_groups_and_noncontiguous_input(
    bits, granularity, dtype, symmetric
):
    source = torch.randn(2, 3, 8, 7, generator=torch.Generator().manual_seed(9))
    x = source.to(dtype).transpose(-1, -2)  # [2, 3, 7, 8], noncontiguous
    before = x.clone()
    codec = IntQuantizer(
        IntQuantizerConfig(
            bits=bits, granularity=granularity, group_size=4, symmetric=symmetric
        )
    )
    encoded = codec.quantize(x)
    restored = codec.dequantize(encoded)
    assert restored.shape == x.shape and restored.dtype == dtype
    assert restored.device == x.device and torch.isfinite(restored).all()
    assert torch.equal(x, before)
    rounding = torch.finfo(dtype).eps * x.abs().max()
    assert (restored.float() - x.float()).abs().max() <= (
        encoded.scales.float().max() / 2 + rounding + 1e-6
    )


def test_kept_token_does_not_widen_another_per_channel_group():
    x = torch.tensor([100.0, 100.0, 100.0, 100.0, 9999.0, 1.0, 2.0, 3.0]).reshape(
        1, 1, 8, 1
    )
    keep = torch.zeros(1, 1, 8, dtype=torch.bool)
    keep[..., 4] = True
    quantizer = IntQuantizer(
        IntQuantizerConfig(
            bits=2,
            group_size=4,
            granularity=QuantGranularity.PER_CHANNEL,
            scale_dtype=ParamDType.FP32,
            offset_dtype=ParamDType.FP32,
        )
    )
    encoded = quantizer.quantize(x, PrecisionPlan(bits=2, keep_tokens=keep))
    # The second group's surviving [1,2,3] must set its range, independently of
    # both the kept 9999 and the preceding group's 100s.
    assert encoded.zero_points.flatten().tolist() == [100.0, 1.0]
    assert encoded.scales.flatten()[1].item() == pytest.approx(2 / 3)
    assert torch.allclose(
        quantizer.dequantize(encoded).flatten()[5:], torch.tensor([1.0, 7 / 3, 3.0])
    )


def test_calibration_cannot_change_live_encoded_history():
    codec = NonUniformQuantizer(
        NonUniformQuantizerConfig(bits=2, mode="calibrated", calibration_iters=3)
    )
    samples = torch.linspace(-2, 2, 32).reshape(1, 1, 8, 4)
    context = AttentionContext(0, "key", 0)
    codec.calibrate({(0, "key"): samples})
    encoded = codec.quantize(samples, context=context)
    before = codec.dequantize(encoded)
    try:
        codec.calibrate({(0, "key"): samples + 100})
    except (ValueError, RuntimeError):
        pass
    assert torch.equal(codec.dequantize(encoded), before)


@pytest.mark.parametrize("config", [IntQuantizerConfig(bits=2), FP8QuantizerConfig()])
def test_finite_fp16_extrema_decode_finitely_by_saturation(config):
    x = torch.tensor([-65504.0, 65504.0] * 16, dtype=torch.float16).reshape(1, 1, 1, 32)
    codec = KVQuantizer.create_from_config(config)
    restored = codec.fake_quantize(x)
    assert torch.isfinite(restored).all(), restored
    assert torch.equal(restored, x)
