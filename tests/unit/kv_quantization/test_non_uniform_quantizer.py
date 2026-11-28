"""Unit tests for NonUniformQuantizer (Gaussian Lloyd-Max and calibrated tables)."""

import math

import pytest
import torch

from sparse_attention_hub.kv_quantization import (
    AttentionContext,
    KVQuantization,
    KVQuantizationConfig,
    KVQuantizer,
    KVQuantizerRegistry,
    NonUniformQuantizer,
    NonUniformQuantizerConfig,
    ParamDType,
    QuantGranularity,
    RandomRotationConfig,
)
from sparse_attention_hub.kv_quantization.quantizers.implementations.non_uniform import (
    lloyd_max_gaussian,
)

pytestmark = pytest.mark.unit

CONTEXT = AttentionContext(0, "key", 0)


def gaussian(tokens: int = 256, channels: int = 128, seed: int = 0) -> torch.Tensor:
    return torch.randn(
        1, 2, tokens, channels, generator=torch.Generator().manual_seed(seed)
    )


def outlier_channels(tokens: int = 256, seed: int = 0) -> torch.Tensor:
    x = gaussian(tokens, seed=seed)
    x[..., [3, 50]] *= 20.0
    return x


def rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / a.norm()).item()


class TestLloydMax:
    @pytest.mark.parametrize(
        "bits, positive_levels",
        [
            (1, [0.7979]),
            (2, [0.4528, 1.5104]),
            (3, [0.2451, 0.7560, 1.3439, 2.1520]),
        ],
    )
    def test_matches_the_known_gaussian_tables(self, bits, positive_levels):
        levels = lloyd_max_gaussian(bits)
        half = levels[len(levels) // 2 :].tolist()
        assert half == pytest.approx(positive_levels, abs=2e-4)
        assert torch.allclose(levels, -levels.flip(0), atol=1e-6)  # symmetric


class TestGaussianMode:
    def test_registry_and_direct_parent(self):
        assert (
            type(KVQuantizerRegistry.create(NonUniformQuantizerConfig()))
            is NonUniformQuantizer
        )
        assert NonUniformQuantizer.__bases__ == (KVQuantizer,)

    @pytest.mark.parametrize("bits, expected", [(1, 0.6), (2, 0.34), (4, 0.097)])
    def test_error_matches_lloyd_max_distortion(self, bits, expected):
        """For Gaussian vectors, relative error is close to the Lloyd-Max distortion."""
        quantizer = NonUniformQuantizer(NonUniformQuantizerConfig(bits=bits))
        error = rel_error(gaussian(), quantizer.fake_quantize(gaussian()))
        assert error == pytest.approx(expected, rel=0.08)

    def test_stores_codes_and_one_norm_per_token(self):
        x = gaussian()
        encoded = NonUniformQuantizer(NonUniformQuantizerConfig(bits=2)).quantize(x)
        tokens = 2 * 256
        assert encoded.norms.dtype == torch.float16
        assert encoded.storage_bytes() == x.numel() * 2 // 8 + tokens * 2

    def test_norm_dtype_follows_scale_dtype(self):
        config = NonUniformQuantizerConfig(bits=2, scale_dtype=ParamDType.FP32)
        assert (
            NonUniformQuantizer(config).quantize(gaussian()).norms.dtype
            == torch.float32
        )

    def test_scale_invariant(self):
        quantizer = NonUniformQuantizer(NonUniformQuantizerConfig(bits=4))
        x = gaussian()
        small = quantizer.fake_quantize(x)
        large = quantizer.fake_quantize(x * 64)
        assert torch.allclose(large, small * 64, rtol=1e-3, atol=1e-3)

    def test_zero_vectors_stay_zero(self):
        x = torch.zeros(1, 1, 4, 128)
        assert torch.equal(
            NonUniformQuantizer(NonUniformQuantizerConfig()).fake_quantize(x), x
        )

    def test_random_rotation_rescues_outlier_channels(self):
        """TurboQuant's recipe: rotate first, so coordinates look Gaussian."""
        x = outlier_channels()

        def run(transform):
            kwargs = {} if transform is None else {"key_transform": transform}
            kvq = KVQuantization.create_from_config(
                KVQuantizationConfig(
                    key_quantizer=NonUniformQuantizerConfig(bits=2), **kwargs
                )
            )
            return kvq.process(0, x, x, {}, num_new_tokens=0)[0]

        assert rel_error(x, run(RandomRotationConfig())) < 0.5 * rel_error(x, run(None))

    def test_needs_per_token(self):
        with pytest.raises(ValueError, match="PER_TOKEN"):
            NonUniformQuantizer(
                NonUniformQuantizerConfig(granularity=QuantGranularity.PER_CHANNEL)
            )


class TestCalibratedMode:
    def calibrated(self, bits: int = 2) -> NonUniformQuantizer:
        quantizer = NonUniformQuantizer(
            NonUniformQuantizerConfig(bits=bits, mode="calibrated")
        )
        quantizer.calibrate({(0, "key"): outlier_channels(2048, seed=7)})
        return quantizer

    def test_tables_are_sorted_per_channel(self):
        table = self.calibrated().tables[(0, "key")]
        assert table.shape == (128, 4)
        assert torch.equal(table, table.sort(dim=-1).values)
        # The large channels learn wide tables.
        assert table[3].abs().max() > 5 * table[0].abs().max()

    def test_handles_outlier_channels_without_a_rotation(self):
        x = outlier_channels()
        error = rel_error(x, self.calibrated().fake_quantize(x, context=CONTEXT))
        gaussian_error = rel_error(
            x, NonUniformQuantizer(NonUniformQuantizerConfig(bits=2)).fake_quantize(x)
        )
        assert error < 0.6 * gaussian_error

    def test_only_codes_are_stored(self):
        x = outlier_channels()
        encoded = self.calibrated().quantize(x, context=CONTEXT)
        assert encoded.storage_bytes() == x.numel() * 2 // 8

    def test_needs_calibration_and_context(self):
        quantizer = NonUniformQuantizer(NonUniformQuantizerConfig(mode="calibrated"))
        with pytest.raises(RuntimeError, match="run calibrate"):
            quantizer.quantize(gaussian(), context=CONTEXT)
        with pytest.raises(ValueError, match="context"):
            self.calibrated().quantize(gaussian())

    def test_tables_are_per_layer_and_kind(self):
        quantizer = self.calibrated()
        with pytest.raises(RuntimeError):
            quantizer.quantize(gaussian(), context=AttentionContext(0, "value", 0))
        with pytest.raises(RuntimeError):
            quantizer.quantize(gaussian(), context=AttentionContext(1, "key", 0))

    def test_save_and_load(self, tmp_path):
        quantizer = self.calibrated()
        path = str(tmp_path / "tables.pt")
        quantizer.save(path)
        loaded = NonUniformQuantizer(
            NonUniformQuantizerConfig(bits=2, mode="calibrated", codebook_path=path)
        )
        x = outlier_channels()
        assert torch.equal(
            loaded.fake_quantize(x, context=CONTEXT),
            quantizer.fake_quantize(x, context=CONTEXT),
        )
        with pytest.raises(ValueError, match="2-bit"):
            NonUniformQuantizer(
                NonUniformQuantizerConfig(bits=4, mode="calibrated", codebook_path=path)
            )

    def test_inside_kv_quantization_fake_equals_packed(self):
        x = outlier_channels()
        config = NonUniformQuantizerConfig(bits=2, mode="calibrated")
        outputs = []
        for store_codes in (False, True):
            kvq = KVQuantization.create_from_config(
                KVQuantizationConfig(key_quantizer=config, store_codes=store_codes)
            )
            kvq.key_quantizer.calibrate({(0, "key"): outlier_channels(2048, seed=7)})
            outputs.append(kvq.process(0, x, x, {}, num_new_tokens=0)[0])
        assert torch.equal(outputs[0], outputs[1])


class TestSettings:
    @pytest.mark.parametrize(
        "kwargs, message", [({"mode": "pca"}, "mode"), ({"bits": 3}, "bits")]
    )
    def test_invalid_settings(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            NonUniformQuantizer(NonUniformQuantizerConfig(**kwargs))

    def test_calibrate_is_only_for_calibrated_mode(self):
        with pytest.raises(ValueError, match="calibrated"):
            NonUniformQuantizer(NonUniformQuantizerConfig()).calibrate(
                {(0, "key"): gaussian()}
            )

    def test_one_bit_works(self):
        restored = NonUniformQuantizer(NonUniformQuantizerConfig(bits=1)).fake_quantize(
            gaussian()
        )
        assert math.isfinite(restored.sum().item())
