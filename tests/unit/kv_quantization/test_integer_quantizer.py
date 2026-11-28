"""Unit tests for IntQuantizer's scale / offset storage dtypes."""

import itertools

import pytest
import torch

from sparse_attention_hub.kv_quantization import (
    IntQuantizer,
    IntQuantizerConfig,
    KVQuantizer,
    KVQuantizerConfig,
    ParamDType,
    QuantGranularity,
)

pytestmark = pytest.mark.unit

GROUP = 32


def x(tokens: int = 64, channels: int = 32, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(1, 2, tokens, channels)


def rel_error(original: torch.Tensor, restored: torch.Tensor) -> float:
    return ((original - restored).norm() / original.norm()).item()


class TestDefaults:
    def test_attached_directly_to_kvquantizer(self):
        assert IntQuantizer.__bases__ == (KVQuantizer,)

    def test_scales_and_offsets_default_to_fp16(self):
        config = KVQuantizerConfig()
        assert config.scale_dtype == ParamDType.FP16
        assert config.offset_dtype == ParamDType.FP16
        encoded = IntQuantizer(IntQuantizerConfig(bits=2)).quantize(x())
        assert encoded.scales.dtype == torch.float16
        assert encoded.zero_points.dtype == torch.float16


class TestStorage:
    @pytest.mark.parametrize(
        "scale_dtype, offset_dtype", list(itertools.product(ParamDType, ParamDType))
    )
    def test_every_combination_is_stored_as_configured(self, scale_dtype, offset_dtype):
        config = IntQuantizerConfig(
            bits=2, group_size=GROUP, scale_dtype=scale_dtype, offset_dtype=offset_dtype
        )
        encoded = IntQuantizer(config).quantize(x())
        assert encoded.scales.dtype == scale_dtype.torch_dtype
        assert encoded.zero_points.dtype == offset_dtype.torch_dtype

        groups = encoded.scales.numel()
        code_bytes = x().numel() * 2 // 8
        param_bytes = (
            groups
            * (
                torch.finfo(scale_dtype.torch_dtype).bits
                + torch.finfo(offset_dtype.torch_dtype).bits
            )
            // 8
        )
        assert encoded.storage_bytes() == code_bytes + param_bytes

    def test_fp16_halves_parameter_overhead(self):
        def bytes_for(dtype):
            config = IntQuantizerConfig(
                bits=2, group_size=GROUP, scale_dtype=dtype, offset_dtype=dtype
            )
            return IntQuantizer(config).quantize(x()).storage_bytes()

        code_bytes = x().numel() * 2 // 8
        assert (
            bytes_for(ParamDType.FP16) - code_bytes
            == (bytes_for(ParamDType.FP32) - code_bytes) // 2
        )

    def test_symmetric_stores_no_offset(self):
        config = IntQuantizerConfig(bits=4, symmetric=True)
        encoded = IntQuantizer(config).quantize(x())
        assert encoded.zero_points is None
        restored = IntQuantizer(config).dequantize(encoded)
        assert rel_error(x(), restored) < 0.15


class TestDecoding:
    def test_decoding_uses_exactly_the_stored_values(self):
        config = IntQuantizerConfig(
            bits=4, group_size=GROUP, scale_dtype=ParamDType.BF16
        )
        quantizer = IntQuantizer(config)
        encoded = quantizer.quantize(x())
        # Re-encoding the decoded values reproduces the same parameters and codes.
        again = quantizer.quantize(quantizer.dequantize(encoded))
        assert torch.equal(again.codes, encoded.codes)

    def test_16_bit_scales_cost_little_accuracy_on_ordinary_data(self):
        def error(dtype):
            config = IntQuantizerConfig(
                bits=4, group_size=GROUP, scale_dtype=dtype, offset_dtype=dtype
            )
            quantizer = IntQuantizer(config)
            return rel_error(x(), quantizer.fake_quantize(x()))

        assert error(ParamDType.FP16) == pytest.approx(error(ParamDType.FP32), rel=0.01)

    def test_bf16_offsets_cost_precision_far_from_zero(self):
        # Values 10.06 .. 10.11: near 10, bf16 steps are 0.0625 and fp16 steps
        # 0.0078, so the bf16 offset widens each group's range about twice as much.
        noise = torch.rand(1, 2, 64, 32, generator=torch.Generator().manual_seed(0))
        data = 10.06 + 0.05 * noise

        def error(offset_dtype):
            config = IntQuantizerConfig(
                bits=4, group_size=GROUP, offset_dtype=offset_dtype
            )
            restored = IntQuantizer(config).fake_quantize(data)
            return (data - restored).abs().max().item()

        assert error(ParamDType.BF16) > 1.5 * error(ParamDType.FP16)

    @pytest.mark.parametrize("dtype", list(ParamDType))
    def test_stored_range_never_clips(self, dtype):
        """Offsets round down and scales round up, so every value is covered."""
        noise = torch.rand(1, 2, 64, 32, generator=torch.Generator().manual_seed(1))
        data = 10.04 + 0.05 * noise  # 10.04 rounds *up* to 10.0625 in bf16
        config = IntQuantizerConfig(
            bits=4, group_size=GROUP, scale_dtype=dtype, offset_dtype=dtype
        )
        quantizer = IntQuantizer(config)
        encoded = quantizer.quantize(data)
        offsets = encoded.zero_points.float()
        top = offsets + encoded.scales.float() * 15
        groups = data.reshape(1, 2, 64, 1, 32)
        assert (offsets <= groups.amin(-1, keepdim=True)).all()
        assert (top >= groups.amax(-1, keepdim=True)).all()
        step = encoded.scales.float().max().item()
        assert (data - quantizer.fake_quantize(data)).abs().max() <= step / 2 + 1e-6

    def test_constant_group_with_representable_value_is_exact(self):
        data = torch.full((1, 2, 4, 32), 0.5)
        restored = IntQuantizer(IntQuantizerConfig(bits=2)).fake_quantize(data)
        assert torch.equal(restored, data)

    def test_tiny_ranges_stay_finite(self):
        data = 1e-9 * x()
        restored = IntQuantizer(IntQuantizerConfig(bits=4)).fake_quantize(data)
        assert torch.isfinite(restored).all()
        assert (data - restored).abs().max() < 1e-4


class TestRangeErrors:
    def test_offset_too_large_for_fp16_raises(self):
        data = 1e6 + x()
        with pytest.raises(ValueError, match="offset exceeds the fp16 range"):
            IntQuantizer(IntQuantizerConfig(bits=4)).quantize(data)

    def test_wider_offset_dtype_handles_it(self):
        data = 1e6 + x()
        config = IntQuantizerConfig(bits=4, offset_dtype=ParamDType.FP32)
        restored = IntQuantizer(config).fake_quantize(data)
        assert torch.isfinite(restored).all()


class TestPerBlock:
    def test_group_size_must_divide_head_dim(self):
        config = IntQuantizerConfig(
            bits=4, granularity=QuantGranularity.PER_BLOCK, group_size=24
        )
        with pytest.raises(ValueError, match="divide head_dim"):
            IntQuantizer(config).quantize(x(channels=32))

    @pytest.mark.parametrize("group_size", [8, 16, 32])
    def test_blocks_stay_inside_one_token(self, group_size):
        """With blocks inside tokens, per-block equals per-token grouping."""
        data = x(tokens=37, channels=32)
        per_block = IntQuantizerConfig(
            bits=4, granularity=QuantGranularity.PER_BLOCK, group_size=group_size
        )
        per_token = IntQuantizerConfig(
            bits=4, granularity=QuantGranularity.PER_TOKEN, group_size=group_size
        )
        assert torch.equal(
            IntQuantizer(per_block).fake_quantize(data),
            IntQuantizer(per_token).fake_quantize(data),
        )
