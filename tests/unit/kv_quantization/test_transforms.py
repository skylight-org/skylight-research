"""Unit tests for the KV transforms: Identity, Hadamard, RandomHadamard, RandomRotation."""

import pytest
import torch

from sparse_attention_hub.kv_quantization import (
    AttentionContext,
    HadamardTransform,
    HadamardTransformConfig,
    IdentityTransform,
    IdentityTransformConfig,
    IntQuantizer,
    IntQuantizerConfig,
    KVQuantization,
    KVQuantizationConfig,
    KVTransform,
    KVTransformRegistry,
    RandomHadamardTransform,
    RandomHadamardTransformConfig,
    RandomRotation,
    RandomRotationConfig,
)
from sparse_attention_hub.kv_quantization.transforms.implementations.hadamard import (
    hadamard_matrix,
)

pytestmark = pytest.mark.unit

DIM = 64
INT4 = IntQuantizerConfig(bits=4, group_size=DIM)  # per token, one group per vector


def context(layer: int = 0, kind: str = "key") -> AttentionContext:
    return AttentionContext(layer, kind, 0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("preserve", [False, True])
def test_hadamard_forward_dtype_is_explicit_and_inverse_uses_saved_dtype(
    dtype, preserve
):
    from tests.reference.quarot import normalized_matrix

    transform = HadamardTransform(
        HadamardTransformConfig(preserve_input_dtype=preserve)
    )
    x = torch.randn(1, 2, 5, 256, generator=torch.Generator().manual_seed(812)).to(
        dtype
    )[..., ::2]
    original = x.clone()
    rotated, state = transform.forward(x, context())
    expected = x.float() @ normalized_matrix(128)
    if preserve:
        expected = expected.to(dtype)
    assert rotated.dtype == (dtype if preserve else torch.float32)
    assert torch.equal(rotated, expected)
    restored = transform.inverse(rotated, context(), state)
    assert restored.dtype == dtype
    expected_inverse = (expected.float() @ normalized_matrix(128).T).to(dtype)
    assert torch.equal(restored, expected_inverse)
    assert torch.equal(x, original)


def test_hadamard_dtype_flag_must_be_boolean():
    with pytest.raises(ValueError, match="boolean"):
        HadamardTransform(HadamardTransformConfig(preserve_input_dtype=1))


def gaussian(tokens: int = 128, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(1, 2, tokens, DIM)


def with_outlier_channels(tokens: int = 128, seed: int = 0) -> torch.Tensor:
    """Keys with a few huge fixed channels, as real keys have."""
    x = gaussian(tokens, seed)
    x[..., [3, 17]] *= 25.0
    return x


def quantization_error(transform: KVTransform, x: torch.Tensor) -> float:
    """Relative error of int4 per-token quantization in the transform's space."""
    rotated, state = transform.forward(x, context())
    restored = transform.inverse(
        IntQuantizer(INT4).fake_quantize(rotated), context(), state
    )
    return ((x - restored).norm() / x.norm()).item()


ROTATIONS = [
    (HadamardTransform, HadamardTransformConfig()),
    (RandomHadamardTransform, RandomHadamardTransformConfig(seed=3)),
    (RandomRotation, RandomRotationConfig(seed=1)),
]


class TestRegistry:
    @pytest.mark.parametrize(
        "config, cls",
        [
            (IdentityTransformConfig(), IdentityTransform),
            (HadamardTransformConfig(), HadamardTransform),
            (RandomRotationConfig(), RandomRotation),
            (RandomHadamardTransformConfig(), RandomHadamardTransform),
        ],
    )
    def test_each_config_selects_its_transform(self, config, cls):
        transform = KVTransformRegistry.create(config)
        assert type(transform) is cls
        assert cls.__bases__ == (KVTransform,)


class TestOrthogonalRotations:
    @pytest.mark.parametrize("cls, config", ROTATIONS)
    def test_inverse_restores_the_input(self, cls, config):
        transform = cls(config)
        x = gaussian()
        rotated, state = transform.forward(x, context())
        assert not torch.allclose(rotated, x)
        restored = transform.inverse(rotated, context(), state)
        assert torch.allclose(restored, x, atol=1e-5)

    @pytest.mark.parametrize("cls, config", ROTATIONS)
    def test_lengths_are_preserved(self, cls, config):
        transform = cls(config)
        x = gaussian()
        rotated, _ = transform.forward(x, context())
        assert torch.allclose(rotated.norm(dim=-1), x.norm(dim=-1), rtol=1e-5)

    @pytest.mark.parametrize("cls, config", ROTATIONS)
    def test_original_dtype_comes_back(self, cls, config):
        transform = cls(config)
        x = gaussian().to(torch.bfloat16)
        rotated, state = transform.forward(x, context())
        assert rotated.dtype == torch.float32
        assert transform.inverse(rotated, context(), state).dtype == torch.bfloat16

    @pytest.mark.parametrize("cls, config", ROTATIONS)
    def test_rotation_lowers_error_on_outlier_channels(self, cls, config):
        x = with_outlier_channels()
        identity_error = quantization_error(
            IdentityTransform(IdentityTransformConfig()), x
        )
        assert quantization_error(cls(config), x) < 0.7 * identity_error


class TestHadamard:
    def test_matrix_is_orthogonal(self):
        matrix = hadamard_matrix(DIM)
        assert torch.allclose(matrix @ matrix.T, torch.eye(DIM), atol=1e-6)

    def test_head_dim_must_be_a_power_of_two(self):
        with pytest.raises(ValueError, match="power of 2"):
            HadamardTransform(HadamardTransformConfig()).forward(
                torch.randn(1, 1, 4, 48), context()
            )


class TestRandomHadamard:
    def test_is_signed_hadamard_and_orthogonal(self):
        matrix = RandomHadamardTransform(RandomHadamardTransformConfig()).matrix(
            0, "key", DIM
        )
        assert torch.allclose(matrix @ matrix.T, torch.eye(DIM), atol=1e-6)
        assert torch.equal(matrix.abs(), hadamard_matrix(DIM).abs())

    def test_reproducible_and_distinct_per_layer_and_kind(self):
        one = RandomHadamardTransform(RandomHadamardTransformConfig(seed=1))
        two = RandomHadamardTransform(RandomHadamardTransformConfig(seed=1))
        assert torch.equal(one.matrix(0, "key", DIM), two.matrix(0, "key", DIM))
        assert not torch.equal(one.matrix(0, "key", DIM), one.matrix(0, "value", DIM))
        assert not torch.equal(one.matrix(0, "key", DIM), one.matrix(1, "key", DIM))
        other = RandomHadamardTransform(RandomHadamardTransformConfig(seed=2))
        assert not torch.equal(one.matrix(0, "key", DIM), other.matrix(0, "key", DIM))


class TestRandomRotation:
    def test_reproducible_and_distinct_per_layer_and_kind(self):
        one = RandomRotation(RandomRotationConfig(seed=0))
        two = RandomRotation(RandomRotationConfig(seed=0))
        assert torch.equal(one.matrix(0, "key", DIM), two.matrix(0, "key", DIM))
        assert not torch.equal(one.matrix(0, "key", DIM), one.matrix(0, "value", DIM))
        assert not torch.equal(one.matrix(0, "key", DIM), one.matrix(1, "key", DIM))
        other_seed = RandomRotation(RandomRotationConfig(seed=1))
        assert not torch.equal(
            one.matrix(0, "key", DIM), other_seed.matrix(0, "key", DIM)
        )

    def test_matrix_is_orthogonal(self):
        matrix = RandomRotation(RandomRotationConfig()).matrix(2, "value", DIM)
        assert torch.allclose(matrix @ matrix.T, torch.eye(DIM), atol=1e-5)


class TestInsideKVQuantization:
    @pytest.mark.parametrize(
        "transform",
        [
            HadamardTransformConfig(),
            RandomHadamardTransformConfig(seed=1),
            RandomRotationConfig(seed=4),
        ],
    )
    def test_fake_and_packed_agree_and_beat_identity(self, transform):
        keys = with_outlier_channels(200).to(torch.bfloat16)
        values = gaussian(200, seed=1).to(torch.bfloat16)

        def run(key_transform, store_codes=False):
            kvq = KVQuantization.create_from_config(
                KVQuantizationConfig(
                    key_transform=key_transform,
                    value_transform=key_transform,
                    key_quantizer=INT4,
                    value_quantizer=INT4,
                    store_codes=store_codes,
                )
            )
            return kvq.process(0, keys, values, {}, num_new_tokens=0)

        fake_k, fake_v = run(transform)
        packed_k, packed_v = run(transform, store_codes=True)
        assert torch.equal(fake_k, packed_k) and torch.equal(fake_v, packed_v)
        assert fake_k.dtype == torch.bfloat16 and fake_k.shape == keys.shape

        plain_k, _ = run(IdentityTransformConfig())

        def err(a):
            return ((a.float() - keys.float()).norm() / keys.float().norm()).item()

        assert err(fake_k) < 0.7 * err(plain_k)
