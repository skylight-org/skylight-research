"""Tests for the fake-quantized KV cache and the backend factory.

Runs on CPU with no modelopt, which is the whole point of the fake backend: the
cache semantics (residual buffer, axis choice, per-layer wiring) get tested here
even on a machine that cannot run the real one.
"""

from typing import Any, List, Tuple

import pytest
import torch

from sparse_attention_hub.kv_quantization import (
    FAKE_NVFP4,
    NVFP4,
    create_quantized_kv_cache,
    validate_backend,
)
from sparse_attention_hub.kv_quantization.constants import (
    AXIS_PER_CHANNEL,
    AXIS_PER_TOKEN,
)
from sparse_attention_hub.kv_quantization.fake_nvfp4_cache import (
    FakeNVFP4QuantizedCache,
    FakeNVFP4QuantizedLayer,
)

BATCH: int = 1
HEADS: int = 4
TOKENS: int = 256
HEAD_DIM: int = 64
OUTLIER_CHANNELS: List[int] = [3, 17, 40]


def _relative_error(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Relative Frobenius error between a tensor and its round trip."""
    difference: torch.Tensor = reconstructed.float() - original.float()
    return float(
        (
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(original.float())
        ).item()
    )


def _per_channel_error(
    original: torch.Tensor, reconstructed: torch.Tensor
) -> torch.Tensor:
    """Relative error per ``head_dim`` channel, reduced over batch and tokens."""
    difference: torch.Tensor = reconstructed.float() - original.float()
    return torch.linalg.vector_norm(
        difference, dim=(0, 1, 2)
    ) / torch.linalg.vector_norm(original.float(), dim=(0, 1, 2))


def _make_keys() -> torch.Tensor:
    """Keys with persistent large-magnitude channels, as real keys have.

    A few residual-stream dimensions hold large, nearly token-independent values
    so the model can do reliable no-op attention at sink tokens, and that
    structure survives into the keys. This is the pattern that makes the choice
    of blocking axis matter.
    """
    generator: torch.Generator = torch.Generator().manual_seed(0)
    keys: torch.Tensor = torch.randn(
        BATCH, HEADS, TOKENS, HEAD_DIM, generator=generator
    )
    for channel in OUTLIER_CHANNELS:
        keys[:, :, :, channel] = 25.0 + torch.randn(
            BATCH, HEADS, TOKENS, generator=generator
        )
    return keys


def _make_values() -> torch.Tensor:
    """Values: homogeneous across channels, with the occasional loud token."""
    generator: torch.Generator = torch.Generator().manual_seed(1)
    values: torch.Tensor = torch.randn(
        BATCH, HEADS, TOKENS, HEAD_DIM, generator=generator
    )
    values[:, :, ::100, :] *= 6.0
    return values


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
class TestLayerValidation:
    """Argument validation."""

    @pytest.mark.parametrize("axis", [1, -2, 2])
    def test_rejects_unsupported_key_axis(self, axis: int) -> None:
        """Only per-token and per-channel blocking are implemented."""
        with pytest.raises(ValueError, match="axis_key"):
            FakeNVFP4QuantizedLayer(axis_key=axis)

    @pytest.mark.parametrize("axis", [1, -2, 2])
    def test_rejects_unsupported_value_axis(self, axis: int) -> None:
        """Same check for values."""
        with pytest.raises(ValueError, match="axis_value"):
            FakeNVFP4QuantizedLayer(axis_value=axis)


@pytest.mark.unit
class TestLayerRoundTrip:
    """Fake quantization keeps the tensor's layout, unlike the packed backend."""

    @pytest.mark.parametrize("axis", [AXIS_PER_TOKEN, AXIS_PER_CHANNEL])
    def test_shape_and_dtype_survive(self, axis: int) -> None:
        """A round trip returns the tensor it was given, not its transpose."""
        layer: FakeNVFP4QuantizedLayer = FakeNVFP4QuantizedLayer()
        original: torch.Tensor = _make_values().to(torch.bfloat16)
        reconstructed: torch.Tensor = layer._dequantize(
            layer._quantize(original, axis=axis)
        )
        assert reconstructed.shape == original.shape
        assert reconstructed.dtype == original.dtype

    def test_error_is_small(self) -> None:
        """4 bits per element costs a few percent on well-behaved values."""
        layer: FakeNVFP4QuantizedLayer = FakeNVFP4QuantizedLayer()
        original: torch.Tensor = _make_values()
        reconstructed: torch.Tensor = layer._dequantize(
            layer._quantize(original, axis=AXIS_PER_TOKEN)
        )
        assert _relative_error(original, reconstructed) < 0.15


@pytest.mark.unit
class TestAxisChoice:
    """The reason axis_key and axis_value are separate knobs.

    Keys have outlier channels. Blocking over head_dim puts an outlier channel
    in the same block as well-behaved ones, and the shared scale is set by the
    outlier, so the quiet channels lose most of their precision. Blocking over
    tokens isolates each channel with its own scales.
    """

    def _errors(self, axis: int) -> torch.Tensor:
        layer: FakeNVFP4QuantizedLayer = FakeNVFP4QuantizedLayer()
        keys: torch.Tensor = _make_keys()
        return _per_channel_error(keys, layer._quantize(keys, axis=axis))

    def test_per_channel_protects_quiet_key_channels(self) -> None:
        """Quiet channels are more accurate when the scale is not shared."""
        normal_channels: List[int] = [
            channel for channel in range(HEAD_DIM) if channel not in OUTLIER_CHANNELS
        ]
        per_token: torch.Tensor = self._errors(AXIS_PER_TOKEN)
        per_channel: torch.Tensor = self._errors(AXIS_PER_CHANNEL)
        assert per_channel[normal_channels].mean() < per_token[normal_channels].mean()

    def test_the_trade_off_is_worth_taking(self) -> None:
        """Per-channel helps the quiet channels far more than it costs the loud ones.

        Per-channel blocking is not free for an outlier channel: these channels
        are a large near-constant value plus small jitter, and a grid whose
        spacing is set by the constant loses the jitter. But the effect is an
        order of magnitude smaller than what the quiet channels gain, which is
        what makes axis_key=0 the better choice for keys.
        """
        normal_channels: List[int] = [
            channel for channel in range(HEAD_DIM) if channel not in OUTLIER_CHANNELS
        ]
        per_token: torch.Tensor = self._errors(AXIS_PER_TOKEN)
        per_channel: torch.Tensor = self._errors(AXIS_PER_CHANNEL)

        quiet_gain: torch.Tensor = (
            per_token[normal_channels].mean() - per_channel[normal_channels].mean()
        )
        outlier_cost: torch.Tensor = (
            per_channel[OUTLIER_CHANNELS].mean() - per_token[OUTLIER_CHANNELS].mean()
        )
        assert quiet_gain > 0
        assert quiet_gain > 5 * outlier_cost


@pytest.mark.unit
class TestCacheConstruction:
    """Cache-level wiring."""

    def test_one_layer_per_decoder_layer(self) -> None:
        """The cache mirrors the model's layer count."""
        cache: FakeNVFP4QuantizedCache = FakeNVFP4QuantizedCache(_small_llama_config(3))
        assert len(cache.layers) == 3
        assert all(isinstance(layer, FakeNVFP4QuantizedLayer) for layer in cache.layers)

    def test_options_reach_every_layer(self) -> None:
        """Cache options are not silently dropped on the way to the layers."""
        cache: FakeNVFP4QuantizedCache = FakeNVFP4QuantizedCache(
            _small_llama_config(),
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
            FakeNVFP4QuantizedCache(config)


@pytest.mark.unit
class TestCacheUpdate:
    """The cache behaviour a model depends on when it reads KV back."""

    def _update(
        self, cache: FakeNVFP4QuantizedCache, tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        keys: torch.Tensor = _make_keys()[:, :, :tokens, :].to(torch.bfloat16)
        values: torch.Tensor = _make_values()[:, :, :tokens, :].to(torch.bfloat16)
        out_keys, out_values = cache.update(keys, values, layer_idx=0)
        return keys, values, out_keys, out_values

    def test_returns_full_sequence(self) -> None:
        """What comes back covers every token that went in."""
        cache: FakeNVFP4QuantizedCache = FakeNVFP4QuantizedCache(
            _small_llama_config(), residual_length=64
        )
        keys, _, out_keys, out_values = self._update(cache, tokens=TOKENS)
        assert out_keys.shape == keys.shape
        assert out_values.shape == keys.shape

    def test_reconstruction_is_close_to_the_input(self) -> None:
        """Reading the cache back gives approximately what was written."""
        cache: FakeNVFP4QuantizedCache = FakeNVFP4QuantizedCache(
            _small_llama_config(), residual_length=64
        )
        keys, values, out_keys, out_values = self._update(cache, tokens=TOKENS)
        assert _relative_error(keys, out_keys) < 0.15
        assert _relative_error(values, out_values) < 0.15

    def test_short_sequence_is_not_quantized_at_all(self) -> None:
        """Below residual_length everything stays in the residual buffer, exact."""
        cache: FakeNVFP4QuantizedCache = FakeNVFP4QuantizedCache(
            _small_llama_config(), residual_length=128
        )
        keys, values, out_keys, out_values = self._update(cache, tokens=32)
        assert torch.equal(out_keys, keys)
        assert torch.equal(out_values, values)


@pytest.mark.unit
class TestFactory:
    """Backend dispatch, which is what the adapter flag actually calls."""

    def test_fake_backend_needs_no_modelopt(self) -> None:
        """The fake path must not import the real one."""
        cache: Any = create_quantized_kv_cache(
            _small_llama_config(), backend=FAKE_NVFP4
        )
        assert isinstance(cache, FakeNVFP4QuantizedCache)

    def test_options_are_forwarded_through_the_factory(self) -> None:
        """Cache options survive the dispatch."""
        cache: Any = create_quantized_kv_cache(
            _small_llama_config(), backend=FAKE_NVFP4, residual_length=32
        )
        assert cache.layers[0].residual_length == 32

    @pytest.mark.parametrize("backend", ["fp4", "fake", "real", "NVFP4", ""])
    def test_unknown_backend_is_rejected(self, backend: str) -> None:
        """A typo in the backend name must not silently pick something."""
        with pytest.raises(ValueError, match="unknown KV cache quantization backend"):
            validate_backend(backend)

    def test_true_is_rejected(self) -> None:
        """quantize_kv_cache=True has no sensible reading.

        The two backends differ in what they need and what they give back, so
        "quantize, I don't mind how" would silently pick for the caller.
        """
        with pytest.raises(ValueError, match="unknown KV cache quantization backend"):
            validate_backend(True)

    @pytest.mark.parametrize("backend", [NVFP4, FAKE_NVFP4])
    def test_known_backends_validate(self, backend: str) -> None:
        """Both documented names are accepted, without importing either."""
        assert validate_backend(backend) == backend
