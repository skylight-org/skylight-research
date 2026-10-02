"""Tests for the NVFP4 quantized KV cache itself.

Needs nvidia-modelopt, and a GPU for anything that actually packs a tensor:
modelopt's NVFP4 kernels are CUDA-only. Everything here skips cleanly when
either is missing, so the suite still runs on a laptop.
"""

from typing import Any, List, Tuple

import pytest
import torch

pytest.importorskip("modelopt", reason="nvidia-modelopt is not installed")

from sparse_attention_hub.kv_quantization.constants import (  # noqa: E402
    AXIS_PER_CHANNEL,
    AXIS_PER_TOKEN,
)
from sparse_attention_hub.kv_quantization.nvfp4_cache import (  # noqa: E402
    NVFP4QuantizedCache,
    NVFP4QuantizedLayer,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="modelopt's NVFP4 packing needs CUDA"
)

BATCH: int = 1
HEADS: int = 4
TOKENS: int = 256
HEAD_DIM: int = 64
OUTLIER_CHANNELS: List[int] = [3, 17, 40]


def _relative_error(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Relative Frobenius error between a tensor and its round trip."""
    original = original.float()
    difference: torch.Tensor = reconstructed.float() - original
    return float(
        (
            torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(original)
        ).item()
    )


def _per_channel_error(
    original: torch.Tensor, reconstructed: torch.Tensor
) -> torch.Tensor:
    """Relative error per ``head_dim`` channel, reduced over batch and tokens."""
    original = original.float()
    difference: torch.Tensor = reconstructed.float() - original
    return torch.linalg.vector_norm(
        difference, dim=(0, 1, 2)
    ) / torch.linalg.vector_norm(original, dim=(0, 1, 2))


def _make_keys(device: str) -> torch.Tensor:
    """Keys with persistent large-magnitude channels, as real keys have.

    A few residual-stream dimensions carry large, nearly token-independent
    values so the model can do reliable no-op attention at sink tokens, and that
    structure survives into the keys. This is the pattern that makes the choice
    of blocking axis matter.
    """
    generator: torch.Generator = torch.Generator(device="cpu").manual_seed(0)
    keys: torch.Tensor = torch.randn(
        BATCH, HEADS, TOKENS, HEAD_DIM, generator=generator
    )
    for channel in OUTLIER_CHANNELS:
        keys[:, :, :, channel] = 25.0 + torch.randn(
            BATCH, HEADS, TOKENS, generator=generator
        )
    return keys.to(device)


def _make_values(device: str) -> torch.Tensor:
    """Values: homogeneous across channels, with the occasional loud token."""
    generator: torch.Generator = torch.Generator(device="cpu").manual_seed(1)
    values: torch.Tensor = torch.randn(
        BATCH, HEADS, TOKENS, HEAD_DIM, generator=generator
    )
    values[:, :, ::100, :] *= 6.0
    return values.to(device)


@pytest.mark.unit
class TestLayerValidation:
    """Argument validation, which needs neither modelopt kernels nor a GPU."""

    @pytest.mark.parametrize("axis", [1, -2, 2])
    def test_rejects_unsupported_key_axis(self, axis: int) -> None:
        """Only per-token and per-channel blocking are implemented."""
        with pytest.raises(ValueError, match="axis_key"):
            NVFP4QuantizedLayer(axis_key=axis)

    @pytest.mark.parametrize("axis", [1, -2, 2])
    def test_rejects_unsupported_value_axis(self, axis: int) -> None:
        """Same check for values."""
        with pytest.raises(ValueError, match="axis_value"):
            NVFP4QuantizedLayer(axis_value=axis)

    def test_defaults_are_per_token(self) -> None:
        """Default layout matches the hardware's native per-token blocking."""
        layer: NVFP4QuantizedLayer = NVFP4QuantizedLayer()
        assert layer.axis_key == AXIS_PER_TOKEN
        assert layer.axis_value == AXIS_PER_TOKEN
        assert layer.block_size == 16


@pytest.mark.unit
@requires_cuda
class TestRoundTrip:
    """Quantize/dequantize fidelity and shape preservation."""

    @pytest.mark.parametrize("axis", [AXIS_PER_TOKEN, AXIS_PER_CHANNEL])
    def test_shape_and_dtype_survive(self, axis: int) -> None:
        """A round trip returns the tensor it was given, not its transpose."""
        layer: NVFP4QuantizedLayer = NVFP4QuantizedLayer()
        original: torch.Tensor = _make_values("cuda").to(torch.bfloat16)
        reconstructed: torch.Tensor = layer._dequantize(
            layer._quantize(original, axis=axis)
        )
        assert reconstructed.shape == original.shape
        assert reconstructed.dtype == original.dtype

    def test_error_is_small_for_well_behaved_tensor(self) -> None:
        """4 bits per element should cost a few percent on gaussian values.

        The bound is loose on purpose: the point is to catch a broken scale or a
        wrong axis, both of which land far above this, not to pin down the
        format's exact error.
        """
        layer: NVFP4QuantizedLayer = NVFP4QuantizedLayer()
        original: torch.Tensor = _make_values("cuda")
        reconstructed: torch.Tensor = layer._dequantize(
            layer._quantize(original, axis=AXIS_PER_TOKEN)
        )
        assert _relative_error(original, reconstructed) < 0.15

    def test_transposed_flag_tracks_the_axis(self) -> None:
        """Per-channel blocking is implemented as a transpose, per-token is not."""
        layer: NVFP4QuantizedLayer = NVFP4QuantizedLayer()
        original: torch.Tensor = _make_values("cuda")
        assert layer._quantize(original, axis=AXIS_PER_CHANNEL).transposed
        assert not layer._quantize(original, axis=AXIS_PER_TOKEN).transposed


@pytest.mark.unit
@requires_cuda
class TestAxisChoice:
    """The reason axis_key and axis_value are separate knobs.

    Keys have outlier channels. Blocking over head_dim puts an outlier channel
    in the same block as well-behaved ones, and the shared scale is set by the
    outlier, so the quiet channels lose most of their precision. Blocking over
    tokens isolates each channel with its own scales.
    """

    def test_per_channel_protects_quiet_key_channels(self) -> None:
        """Quiet channels are more accurate when the scale is not shared."""
        layer: NVFP4QuantizedLayer = NVFP4QuantizedLayer()
        keys: torch.Tensor = _make_keys("cuda")
        normal_channels: List[int] = [
            channel for channel in range(HEAD_DIM) if channel not in OUTLIER_CHANNELS
        ]

        per_token: torch.Tensor = _per_channel_error(
            keys, layer._dequantize(layer._quantize(keys, axis=AXIS_PER_TOKEN))
        )
        per_channel: torch.Tensor = _per_channel_error(
            keys, layer._dequantize(layer._quantize(keys, axis=AXIS_PER_CHANNEL))
        )

        assert per_channel[normal_channels].mean() < per_token[normal_channels].mean()

    def test_the_trade_off_is_worth_taking(self) -> None:
        """Per-channel helps the quiet channels far more than it costs the loud ones.

        Per-channel blocking is not free for an outlier channel: these channels
        are a large near-constant value plus small jitter, and a grid whose
        spacing is set by the constant loses the jitter. But the effect is an
        order of magnitude smaller than what the quiet channels gain, which is
        what makes axis_key=0 the better choice for keys.
        """
        layer: NVFP4QuantizedLayer = NVFP4QuantizedLayer()
        keys: torch.Tensor = _make_keys("cuda")
        normal_channels: List[int] = [
            channel for channel in range(HEAD_DIM) if channel not in OUTLIER_CHANNELS
        ]

        per_token: torch.Tensor = _per_channel_error(
            keys, layer._dequantize(layer._quantize(keys, axis=AXIS_PER_TOKEN))
        )
        per_channel: torch.Tensor = _per_channel_error(
            keys, layer._dequantize(layer._quantize(keys, axis=AXIS_PER_CHANNEL))
        )

        quiet_gain: torch.Tensor = (
            per_token[normal_channels].mean() - per_channel[normal_channels].mean()
        )
        outlier_cost: torch.Tensor = (
            per_channel[OUTLIER_CHANNELS].mean() - per_token[OUTLIER_CHANNELS].mean()
        )
        assert quiet_gain > 0
        assert quiet_gain > 5 * outlier_cost


def _small_llama_config(num_hidden_layers: int = 2) -> Any:
    """A tiny full-attention config, built locally so no download is needed."""
    from transformers import LlamaConfig

    return LlamaConfig(
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        hidden_size=HEADS * HEAD_DIM,
        intermediate_size=128,
        vocab_size=128,
    )


@pytest.mark.unit
class TestCacheConstruction:
    """Cache-level wiring, which needs no packing and so no GPU."""

    def test_one_layer_per_decoder_layer(self) -> None:
        """The cache mirrors the model's layer count."""
        cache: NVFP4QuantizedCache = NVFP4QuantizedCache(_small_llama_config(3))
        assert len(cache.layers) == 3
        assert all(isinstance(layer, NVFP4QuantizedLayer) for layer in cache.layers)

    def test_options_reach_every_layer(self) -> None:
        """Cache options are not silently dropped on the way to the layers."""
        cache: NVFP4QuantizedCache = NVFP4QuantizedCache(
            _small_llama_config(),
            block_size=16,
            residual_length=64,
            axis_key=AXIS_PER_CHANNEL,
        )
        for layer in cache.layers:
            assert layer.residual_length == 64
            assert layer.axis_key == AXIS_PER_CHANNEL
            assert layer.axis_value == AXIS_PER_TOKEN

    @pytest.mark.parametrize("other", ["sliding_attention", "linear_attention"])
    def test_rejects_models_that_are_not_all_full_attention(self, other: str) -> None:
        """Only all-full-attention models are supported; other layer types
        (sliding-window, linear attention) have different cache layouts."""
        config: Any = _small_llama_config()
        config.layer_types = ["full_attention", other]
        with pytest.raises(
            ValueError, match="only supports models whose layers are all full"
        ):
            NVFP4QuantizedCache(config)


@pytest.mark.integration
@requires_cuda
class TestAgreementWithFakeBackend:
    """The fake backend must model the real one, not merely resemble it.

    The simulator follows modelopt's rules, so on the GPU the two are
    bit-identical (asserted exactly by the heavy suite in ``tests/integration/kv_quantization/``). This
    quicker check only requires that the two round trips land far closer to
    each other than either lands to the original, which is what makes
    fake-quantized accuracy numbers say something about the real backend.
    """

    @pytest.mark.parametrize("axis", [AXIS_PER_TOKEN, AXIS_PER_CHANNEL])
    def test_both_backends_produce_nearly_the_same_tensor(self, axis: int) -> None:
        """Divergence between the backends is small next to quantization error."""
        from sparse_attention_hub.kv_quantization.fake_nvfp4_cache import (
            FakeNVFP4QuantizedLayer,
        )

        keys: torch.Tensor = _make_keys("cuda")

        real_layer: NVFP4QuantizedLayer = NVFP4QuantizedLayer()
        real: torch.Tensor = real_layer._dequantize(
            real_layer._quantize(keys, axis=axis)
        )
        fake: torch.Tensor = FakeNVFP4QuantizedLayer()._quantize(keys, axis=axis)

        quantization_error: float = _relative_error(keys, real)
        backend_divergence: float = _relative_error(real, fake)
        assert backend_divergence < quantization_error / 2


@pytest.mark.integration
@requires_cuda
class TestCacheUpdate:
    """The cache behaviour a model actually depends on when it reads KV back."""

    def _update(
        self, cache: NVFP4QuantizedCache, tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        keys: torch.Tensor = _make_keys("cuda")[:, :, :tokens, :].to(torch.bfloat16)
        values: torch.Tensor = _make_values("cuda")[:, :, :tokens, :].to(torch.bfloat16)
        out_keys, out_values = cache.update(keys, values, layer_idx=0)
        return keys, values, out_keys, out_values

    def test_returns_full_sequence(self) -> None:
        """What comes back covers every token that went in."""
        cache: NVFP4QuantizedCache = NVFP4QuantizedCache(
            _small_llama_config(), residual_length=64
        )
        keys, _, out_keys, out_values = self._update(cache, tokens=TOKENS)
        assert out_keys.shape == keys.shape
        assert out_values.shape == keys.shape

    def test_reconstruction_is_close_to_the_input(self) -> None:
        """Reading the cache back gives approximately what was written."""
        cache: NVFP4QuantizedCache = NVFP4QuantizedCache(
            _small_llama_config(), residual_length=64
        )
        keys, values, out_keys, out_values = self._update(cache, tokens=TOKENS)
        assert _relative_error(keys, out_keys) < 0.15
        assert _relative_error(values, out_values) < 0.15

    def test_short_sequence_is_not_quantized_at_all(self) -> None:
        """Below residual_length everything stays in the residual buffer, exact."""
        cache: NVFP4QuantizedCache = NVFP4QuantizedCache(
            _small_llama_config(), residual_length=128
        )
        keys, values, out_keys, out_values = self._update(cache, tokens=32)
        assert torch.equal(out_keys, keys)
        assert torch.equal(out_values, values)
