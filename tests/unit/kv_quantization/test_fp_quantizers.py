"""Unit tests for the floating-point quantizers: FP8, FakeNVFP4 and MXFP."""

import pytest
import torch

from sparse_attention_hub.kv_quantization import (
    FakeNVFP4Quantizer,
    FakeNVFP4QuantizerConfig,
    FP8Quantizer,
    FP8QuantizerConfig,
    IntQuantizer,
    IntQuantizerConfig,
    KVQuantization,
    KVQuantizationConfig,
    KVQuantizer,
    KVQuantizerRegistry,
    MXFPQuantizer,
    MXFPQuantizerConfig,
    QuantGranularity,
)
from sparse_attention_hub.kv_quantization.quantizers.common import (
    fp_grid,
    round_to_grid,
)

pytestmark = pytest.mark.unit

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def gaussian(tokens: int = 64, channels: int = 128, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, 2, tokens, channels, generator=generator)


def rel_error(x: torch.Tensor, quantizer: KVQuantizer) -> float:
    restored = quantizer.fake_quantize(x)
    return ((x - restored).norm() / x.norm()).item()


def one_token(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, -1)


MX4 = MXFPQuantizerConfig()
MX8 = MXFPQuantizerConfig(format="E4M3", bits=8)
ALL = [
    (FP8QuantizerConfig(), FP8Quantizer),
    (FP8QuantizerConfig(format="E5M2"), FP8Quantizer),
    (FakeNVFP4QuantizerConfig(), FakeNVFP4Quantizer),
    (MX4, MXFPQuantizer),
    (MX8, MXFPQuantizer),
]


class TestShared:
    @pytest.mark.parametrize("config, cls", ALL)
    def test_registry_and_direct_parent(self, config, cls):
        assert type(KVQuantizerRegistry.create(config)) is cls
        assert cls.__bases__ == (KVQuantizer,)

    @pytest.mark.parametrize("config, cls", ALL)
    def test_shape_and_dtype_come_back(self, config, cls):
        x = gaussian().to(torch.bfloat16)
        restored = cls(config).fake_quantize(x)
        assert restored.shape == x.shape and restored.dtype == torch.bfloat16

    @pytest.mark.parametrize("config, cls", ALL)
    def test_zeros_stay_zero(self, config, cls):
        x = torch.zeros(1, 2, 32, 128)
        assert torch.equal(cls(config).fake_quantize(x), x)

    @pytest.mark.parametrize("config, cls", ALL)
    def test_blocks_along_tokens(self, config, cls):
        """PER_CHANNEL puts the blocks along tokens (KIVI-style keys)."""
        x = gaussian(tokens=48)
        per_channel = cls(
            config.__class__(
                **{**config.__dict__, "granularity": QuantGranularity.PER_CHANNEL}
            )
        )
        restored = per_channel.fake_quantize(x)
        assert restored.shape == x.shape
        assert ((x - restored).norm() / x.norm()).item() < 0.2

    @pytest.mark.parametrize("config, cls", ALL)
    def test_inside_kv_quantization_fake_equals_packed(self, config, cls):
        keys, values = gaussian(200), gaussian(200, seed=1)
        outputs = []
        for store_codes in (False, True):
            kvq = KVQuantization.create_from_config(
                KVQuantizationConfig(
                    key_quantizer=config,
                    value_quantizer=config,
                    store_codes=store_codes,
                )
            )
            outputs.append(kvq.process(0, keys, values, {}, num_new_tokens=0))
        assert torch.equal(outputs[0][0], outputs[1][0])
        assert torch.equal(outputs[0][1], outputs[1][1])


class TestGrid:
    def test_e2m1_values(self):
        assert fp_grid(2, 1).tolist() == E2M1

    def test_fp6_grid_maxima(self):
        assert fp_grid(2, 3).max().item() == 7.5  # E2M3
        assert fp_grid(3, 2).max().item() == 28.0  # E3M2

    def test_ties_round_to_even(self):
        grid = torch.tensor(E2M1)
        ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
        assert grid[round_to_grid(ties, grid)].tolist() == [
            0.0,
            1.0,
            1.0,
            2.0,
            2.0,
            4.0,
            4.0,
        ]

    def test_nearest_and_clamped(self):
        grid = torch.tensor(E2M1)
        values = torch.tensor([0.1, 0.6, 2.4, 5.1, 9.0])
        assert grid[round_to_grid(values, grid)].tolist() == [0.0, 0.5, 2.0, 6.0, 6.0]


class TestFP8:
    def test_representable_values_are_exact(self):
        # The block's max is 448 (E4M3's max), so the scale is exactly 1.
        values = [448.0, -256.0, 1.5, -0.875, 0.0, 3.25, 96.0, -12.0] * 4
        x = one_token(values)
        config = FP8QuantizerConfig(group_size=32)
        assert torch.equal(FP8Quantizer(config).fake_quantize(x), x)

    def test_matches_torch_reference(self):
        x = gaussian()
        config = FP8QuantizerConfig(group_size=32)
        quantizer = FP8Quantizer(config)
        encoded = quantizer.quantize(x)
        scales = encoded.scales.float().reshape(1, 2, 64, 4, 1)
        groups = x.reshape(1, 2, 64, 4, 32)
        expected = (groups / scales).to(torch.float8_e4m3fn).float() * scales
        assert torch.equal(quantizer.dequantize(encoded), expected.reshape(x.shape))

    def test_large_values_do_not_overflow(self):
        x = gaussian() * 1e4
        restored = FP8Quantizer(FP8QuantizerConfig()).fake_quantize(x)
        assert torch.isfinite(restored).all()
        assert ((x - restored).norm() / x.norm()).item() < 0.05

    def test_e4m3_is_more_precise_than_e5m2(self):
        x = gaussian()
        e4m3 = rel_error(x, FP8Quantizer(FP8QuantizerConfig()))
        e5m2 = rel_error(x, FP8Quantizer(FP8QuantizerConfig(format="E5M2")))
        assert e4m3 < e5m2 < 0.08

    def test_storage(self):
        x = gaussian()
        encoded = FP8Quantizer(FP8QuantizerConfig(group_size=32)).quantize(x)
        assert encoded.storage_bytes() == x.numel() + (x.numel() // 32) * 2

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"format": "E3M4"}, "format"),
            ({"bits": 4}, "8 bits"),
            ({"symmetric": False}, "symmetric"),
        ],
    )
    def test_invalid_settings(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            FP8Quantizer(FP8QuantizerConfig(**kwargs))


class TestFakeNVFP4:
    def test_e2m1_values_are_exact_at_full_scale(self):
        # amax 6 -> global scale 6 / 2688, block scale exactly 448 (E4M3), scale 1.
        x = one_token(E2M1 + [-v for v in E2M1])
        assert torch.equal(
            FakeNVFP4Quantizer(FakeNVFP4QuantizerConfig()).fake_quantize(x), x
        )

    def test_each_block_gets_its_own_scale(self):
        first = E2M1 + [-v for v in E2M1]  # block amax 6
        second = [
            v / 2 for v in first
        ]  # block amax 3 -> block scale 224, exact in E4M3
        x = one_token(first + second)
        quantizer = FakeNVFP4Quantizer(FakeNVFP4QuantizerConfig())
        encoded = quantizer.quantize(x)
        block_scales = encoded.scales.view(torch.float8_e4m3fn).float().flatten()
        assert block_scales.tolist() == [448.0, 224.0]
        assert torch.equal(quantizer.dequantize(encoded), x)

    def test_storage_is_4_5_bits_plus_global_scale(self):
        x = gaussian()
        encoded = FakeNVFP4Quantizer(FakeNVFP4QuantizerConfig()).quantize(x)
        assert encoded.storage_bytes() == x.numel() // 2 + x.numel() // 16 + 4
        assert encoded.global_scale.dtype == torch.float32

    def test_error_is_in_the_4_bit_range(self):
        error = rel_error(gaussian(), FakeNVFP4Quantizer(FakeNVFP4QuantizerConfig()))
        assert 0.05 < error < 0.13

    def test_head_dim_must_be_a_multiple_of_16(self):
        with pytest.raises(ValueError, match="multiple of 16"):
            FakeNVFP4Quantizer(FakeNVFP4QuantizerConfig()).quantize(
                gaussian(channels=24)
            )

    def test_fixed_format_settings(self):
        with pytest.raises(ValueError, match="blocks of 16"):
            FakeNVFP4Quantizer(FakeNVFP4QuantizerConfig(group_size=32))
        with pytest.raises(ValueError, match="symmetric"):
            FakeNVFP4Quantizer(FakeNVFP4QuantizerConfig(symmetric=False))


class TestMXFP:
    def test_power_of_two_scale_from_block_max(self):
        # amax 6 -> floor(log2 6) - 2 = 0 -> scale 1: E2M1 values are exact.
        x = one_token((E2M1 + [-v for v in E2M1]) * 2)
        quantizer = MXFPQuantizer(MX4)
        encoded = quantizer.quantize(x)
        assert encoded.scales.flatten().tolist() == [127]
        assert torch.equal(quantizer.dequantize(encoded), x)

    def test_scaled_values_saturate(self):
        # amax 7 still gives scale 1 (floor(log2 7) = 2); 7 saturates to 6.
        x = one_token([7.0] + [0.5] * 31)
        restored = MXFPQuantizer(MX4).fake_quantize(x)
        assert restored.flatten()[0].item() == 6.0

    def test_mxfp8_representable_values_are_exact(self):
        values = [448.0, -256.0, 1.5, -0.875, 0.0, 3.25, 96.0, -12.0] * 4
        x = one_token(values)
        assert torch.equal(MXFPQuantizer(MX8).fake_quantize(x), x)

    def test_storage(self):
        x = gaussian()
        assert (
            MXFPQuantizer(MX4).quantize(x).storage_bytes()
            == x.numel() // 2 + x.numel() // 32
        )
        assert (
            MXFPQuantizer(MX8).quantize(x).storage_bytes()
            == x.numel() + x.numel() // 32
        )

    def test_power_of_two_scale_costs_accuracy_versus_nvfp4(self):
        x = gaussian()
        nvfp4 = rel_error(x, FakeNVFP4Quantizer(FakeNVFP4QuantizerConfig()))
        assert rel_error(x, MXFPQuantizer(MX4)) > nvfp4

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"format": "E2M3"}, "not supported"),
            ({"format": "E4M3"}, "bits=8"),
            ({"group_size": 16}, "blocks of 32"),
            ({"symmetric": False}, "symmetric"),
        ],
    )
    def test_invalid_settings(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            MXFPQuantizer(MXFPQuantizerConfig(**kwargs))


class TestAgainstInt:
    def test_fp8_beats_int4_and_nvfp4_is_close_to_int4(self):
        x = gaussian()
        int4 = rel_error(x, IntQuantizer(IntQuantizerConfig(bits=4)))
        assert rel_error(x, FP8Quantizer(FP8QuantizerConfig())) < int4
        assert rel_error(x, FakeNVFP4Quantizer(FakeNVFP4QuantizerConfig())) < 2 * int4
