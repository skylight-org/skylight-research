"""TurboQuant split into RandomRotation (transform) + codebook codec.

The split must give the same codebook indices as the bundled TurboQuantQuantizer
(same rotation seed), the same QJL signs when its projection is S R, an unbiased
inner-product estimate, and a working end-to-end recipe.
"""

import math

import numpy as np
import pytest
import torch

from sparse_attention_hub.kv_quantization import (
    AttentionContext,
    IdentityTransformConfig,
    KVQuantization,
    RandomRotation,
    RandomRotationConfig,
    TurboQuantCodebookQuantizer,
    TurboQuantCodebookQuantizerConfig,
    TurboQuantKVConfig,
    TurboQuantQuantizer,
    TurboQuantQuantizerConfig,
    TurboQuantSplitKV,
    TurboQuantSplitKVConfig,
)
from sparse_attention_hub.kv_quantization.quantizers.common import unpack_bitstream

pytestmark = pytest.mark.unit
CONTEXT = AttentionContext(2, "key", 0)


def rotation(dimension, context=CONTEXT, seed=0):
    return RandomRotation(RandomRotationConfig(seed=seed)).matrix(
        context.layer_idx, context.tensor_kind, dimension
    )


def indices(record, bits):
    return unpack_bitstream(record.codes, bits, record.metadata["code_count"])


def both(variant, bits, x):
    bundled = TurboQuantQuantizer(TurboQuantQuantizerConfig(bits=bits, variant=variant))
    split = TurboQuantCodebookQuantizer(
        TurboQuantCodebookQuantizerConfig(bits=bits, variant=variant)
    )
    R = rotation(x.shape[-1])
    return bundled, split, R


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
@pytest.mark.parametrize("dimension", [16, 128])
def test_mse_split_matches_bundled_indices_and_reconstruction(bits, dimension):
    x = torch.randn(2, 2, 64, dimension, generator=torch.Generator().manual_seed(1))
    bundled, split, R = both("mse", bits, x)
    rb = bundled.quantize(x, context=CONTEXT)
    rs = split.quantize(x.float() @ R, context=CONTEXT)
    same = indices(rb, bits) == indices(rs, bits)
    # Only FP32 rounding of normalize-then-rotate vs rotate-then-normalize can
    # move a coordinate across a codebook boundary.
    assert same.float().mean() > 0.999
    assert torch.equal(rb.norms, rb.norms) and torch.allclose(
        rs.norms, rb.norms, rtol=1e-6
    )
    restored = split.dequantize(rs) @ R.T
    rows = same.reshape(x.shape).all(dim=-1)
    torch.testing.assert_close(
        restored[rows], bundled.dequantize(rb)[rows], rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("bits", [2, 3])
def test_prod_split_with_projection_s_r_equals_bundled(bits):
    """With projection S R the split computes exactly the bundled Algorithm 2."""
    x = torch.randn(1, 2, 64, 64, generator=torch.Generator().manual_seed(2))
    bundled, split, R = both("prod", bits, x)
    S = bundled.state_for(CONTEXT, 64)["projection"]
    split._projections[(CONTEXT.layer_idx, CONTEXT.tensor_kind, 64)] = S @ R
    rb = bundled.quantize(x, context=CONTEXT)
    split_record = split.quantize(x.float() @ R, context=CONTEXT)
    signs_b = unpack_bitstream(rb.qjl_codes, 1, x.numel())
    signs_s = unpack_bitstream(split_record.qjl_codes, 1, x.numel())
    assert (signs_b == signs_s).float().mean() > 0.999
    torch.testing.assert_close(
        split_record.residual_norms, rb.residual_norms, rtol=1e-4, atol=1e-6
    )
    restored = split.dequantize(split_record) @ R.T
    error = (restored - bundled.dequantize(rb)).norm() / bundled.dequantize(rb).norm()
    assert error < 1e-3


def test_prod_split_inner_products_are_unbiased_over_projection_seeds():
    generator = torch.Generator().manual_seed(315)
    x = torch.randn(1, 1, 1, 8, generator=generator)
    query = torch.randn(8, generator=generator, dtype=torch.float64)
    R = rotation(8)
    estimates = []
    for seed in range(512):
        codec = TurboQuantCodebookQuantizer(
            TurboQuantCodebookQuantizerConfig(variant="prod", bits=2, qjl_seed=seed)
        )
        record = codec.quantize(x.float() @ R, context=CONTEXT)
        estimate = (codec.dequantize(record) @ R.T).double().reshape(-1)
        estimates.append(float(query @ estimate))
    target = float(query @ x.double().flatten())
    standard_error = np.std(estimates, ddof=1) / math.sqrt(len(estimates))
    assert abs(np.mean(estimates) - target) < 6 * standard_error


@pytest.mark.parametrize("variant", ["mse", "prod"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_split_recipe_runs_and_is_as_accurate_as_bundled(variant, dtype):
    generator = torch.Generator().manual_seed(5)
    keys = torch.randn(1, 2, 48, 128, generator=generator).to(dtype)
    values = torch.randn(1, 2, 48, 128, generator=generator).to(dtype)
    quantizer = {"bits": 3, "variant": variant}
    bundled = KVQuantization.create_from_config(
        TurboQuantKVConfig(
            key_quantizer=TurboQuantQuantizerConfig(**quantizer),
            value_quantizer=TurboQuantQuantizerConfig(**quantizer),
        )
    )
    split = KVQuantization.create_from_config(
        TurboQuantSplitKVConfig(
            key_quantizer=TurboQuantCodebookQuantizerConfig(**quantizer),
            value_quantizer=TurboQuantCodebookQuantizerConfig(**quantizer),
        )
    )
    assert type(split) is TurboQuantSplitKV
    metas: list = [{}, {}]
    errors: list = [[], []]
    for end in range(40, 49):
        new, phase = (40, "prefill") if end == 40 else (1, "decode")
        for i, (coordinator, meta) in enumerate(zip((bundled, split), metas)):
            k, v = coordinator.process(
                0,
                keys[..., :end, :],
                values[..., :end, :],
                meta,
                num_new_tokens=new,
                phase=phase,
            )
            assert k.dtype == dtype and k.shape == keys[..., :end, :].shape
            errors[i].append(float((k.float() - keys[..., :end, :].float()).norm()))
    if variant == "mse":
        # Same rotation and codebook: the reconstructions agree.
        torch.testing.assert_close(
            torch.tensor(errors[1]), torch.tensor(errors[0]), rtol=1e-2, atol=0
        )
    else:
        # Different (equally valid) QJL draws: same error scale.
        ratio = sum(errors[1]) / sum(errors[0])
        assert 0.8 < ratio < 1.25


def test_split_recipe_requires_a_random_rotation():
    with pytest.raises(ValueError, match="RandomRotationConfig"):
        TurboQuantSplitKV(
            TurboQuantSplitKVConfig(key_transform=IdentityTransformConfig())
        )
    with pytest.raises(ValueError, match="TurboQuantCodebookQuantizerConfig"):
        TurboQuantSplitKV(
            TurboQuantSplitKVConfig(value_quantizer=TurboQuantQuantizerConfig())
        )


def test_split_codec_rejects_other_records_and_bad_settings():
    x = torch.randn(1, 1, 4, 16, generator=torch.Generator().manual_seed(3))
    mse = TurboQuantCodebookQuantizer(TurboQuantCodebookQuantizerConfig())
    prod = TurboQuantCodebookQuantizer(
        TurboQuantCodebookQuantizerConfig(variant="prod")
    )
    with pytest.raises(ValueError, match="variant/format"):
        prod.dequantize(mse.quantize(x))
    with pytest.raises(ValueError, match="context"):
        prod.quantize(x)
    for bad in ({"variant": "other"}, {"bits": 5}, {"group_size": 64}):
        with pytest.raises(ValueError):
            TurboQuantCodebookQuantizer(TurboQuantCodebookQuantizerConfig(**bad))
