"""Independent Algorithms 1/2, estimator, metadata and numerical edge checks."""

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from sparse_attention_hub.kv_quantization import (
    AttentionContext,
    ParamDType,
    PrecisionPlan,
    QuantGranularity,
    RandomRotationConfig,
    TurboQuantQuantizer,
    TurboQuantQuantizerConfig,
)
from tests.reference.turboquant import decode, encode, unpack

pytestmark = pytest.mark.unit
CONTEXT = AttentionContext(2, "key", 0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("variant", ["mse", "prod"])
@pytest.mark.parametrize("bits", [1, 2, 3, 4])
@pytest.mark.parametrize("dimension", [3, 16, 128])
def test_indices_norms_signs_and_decode_follow_independent_numpy(
    dtype, variant, bits, dimension
):
    codec = TurboQuantQuantizer(TurboQuantQuantizerConfig(bits=bits, variant=variant))
    x = torch.randn(
        1, 2, 3, dimension * 2, generator=torch.Generator().manual_seed(814)
    ).to(dtype)[..., ::2]
    original = x.clone()
    state = codec.state_for(CONTEXT, dimension)
    expected = encode(x, state, variant=variant, bits=bits)
    actual = codec.quantize(x, context=CONTEXT)
    coarse_bits = bits - (variant == "prod")
    if coarse_bits:
        logical = unpack(actual.codes, coarse_bits, x.numel()).reshape(x.shape)
        np.testing.assert_array_equal(logical, expected.indices)
    else:
        assert actual.codes.numel() == 0
    np.testing.assert_allclose(
        actual.norms.numpy(), expected.norms, rtol=3e-7, atol=1e-7
    )
    assert actual.norms.dtype == torch.float32
    if variant == "prod":
        signs = unpack(actual.qjl_codes, 1, x.numel()).reshape(x.shape)
        np.testing.assert_array_equal(signs, expected.sign_bits)
        np.testing.assert_allclose(
            actual.residual_norms.numpy(), expected.residual_norms, rtol=3e-5, atol=2e-6
        )
        assert actual.residual_norms.dtype == torch.float32
    else:
        assert actual.qjl_codes is None and actual.residual_norms is None
    reference_decoded = torch.tensor(decode(actual, state), dtype=torch.float32)
    actual_decoded = codec.dequantize(actual)
    # Floating GEMMs can reorder sums; integer codes above remain exact.
    tolerance = (
        3e-6 * math.sqrt(dimension) * max(1.0, reference_decoded.abs().max().item())
    )
    torch.testing.assert_close(
        actual_decoded.float(),
        reference_decoded,
        rtol=2 * torch.finfo(dtype).eps,
        atol=tolerance,
    )
    assert torch.equal(x, original) and actual_decoded.dtype == dtype
    count = x.numel()
    vectors = math.prod(x.shape[:-1])
    expected_bytes = (
        (count * coarse_bits + 7) // 8 if coarse_bits else 0
    ) + vectors * 4
    if variant == "prod":
        expected_bytes += (count + 7) // 8 + vectors * 4
    assert actual.storage_bytes() == expected_bytes


def test_fixed_fixture_checks_reference_algebra_and_orientation():
    data = json.loads(
        (
            Path(__file__).resolve().parents[2] / "reference/turboquant/vectors.json"
        ).read_text()
    )
    x = torch.tensor(data["x"]).reshape(1, 1, 1, 3)
    state = {
        name: torch.tensor(data[name])
        for name in ("rotation", "projection", "codebook")
    }
    mse = encode(x, state, variant="mse", bits=2)
    prod = encode(x, state, variant="prod", bits=3)
    np.testing.assert_array_equal(mse.indices.reshape(-1), data["indices"])
    np.testing.assert_allclose(
        mse.reconstructed.reshape(-1), data["mse_reconstructed"], atol=1e-12
    )
    np.testing.assert_allclose(
        prod.reconstructed.reshape(-1), data["prod_reconstructed"], atol=3e-8
    )


@pytest.mark.parametrize("variant", ["mse", "prod"])
def test_zero_vectors_and_one_bit_coarse_stage_have_defined_payloads(variant):
    codec = TurboQuantQuantizer(TurboQuantQuantizerConfig(variant=variant, bits=1))
    x = torch.zeros(2, 2, 3, 7)
    q = codec.quantize(x, context=CONTEXT)
    assert not q.norms.any()
    assert torch.equal(codec.dequantize(q), x)
    if variant == "prod":
        assert q.codes.numel() == 0 and q.metadata["coarse_bits"] == 0
        assert not q.residual_norms.any()


def test_deterministic_resources_are_orthogonal_independent_and_do_not_change_global_rng():
    config = TurboQuantQuantizerConfig(
        variant="prod", rotation=RandomRotationConfig(seed=7), qjl_seed=18
    )
    codec = TurboQuantQuantizer(config)
    torch.manual_seed(42)
    before = torch.get_rng_state().clone()
    state = codec.state_for(CONTEXT, 32)
    assert torch.equal(torch.get_rng_state(), before)
    rotation = state["rotation"].double()
    torch.testing.assert_close(
        rotation.T @ rotation, torch.eye(32, dtype=torch.float64), rtol=1e-6, atol=1e-6
    )
    assert not torch.allclose(
        state["projection"].T @ state["projection"], torch.eye(32)
    )
    fresh = TurboQuantQuantizer(config).state_for(CONTEXT, 32)
    assert all(torch.equal(state[k], fresh[k]) for k in state)
    different = codec.state_for(AttentionContext(2, "value", 0), 32)
    assert not torch.equal(state["rotation"], different["rotation"])
    assert not torch.equal(state["projection"], different["projection"])
    size = codec.shared_state_bytes()
    codec.state_for(CONTEXT, 32)
    assert codec.shared_state_bytes() == size
    state["rotation"].zero_()
    state["codebook"].zero_()
    assert torch.equal(codec.state_for(CONTEXT, 32)["rotation"], fresh["rotation"])
    # CPU and device aliases share allocations; two contexts share one table.
    assert size == 4 * (4 * 32 * 32 + 2**codec.coarse_bits)


def test_qjl_dense_correction_matches_projected_query_and_is_unbiased_in_seed_average():
    generator = torch.Generator().manual_seed(315)
    x = torch.randn(1, 1, 1, 8, generator=generator)
    query = torch.randn(8, generator=generator, dtype=torch.float64)
    estimates = []
    # Independent projection seeds, with the same original vector and rotation.
    for seed in range(512):
        codec = TurboQuantQuantizer(
            TurboQuantQuantizerConfig(variant="prod", bits=2, qjl_seed=seed)
        )
        q = codec.quantize(x, context=CONTEXT)
        state = codec.state_for(CONTEXT, 8)
        estimate = decode(q, state).reshape(-1)
        estimates.append(float(query.numpy() @ estimate))
        if seed == 0:
            signs = 2 * unpack(q.qjl_codes, 1, 8) - 1
            projection = state["projection"].double().numpy()
            correction = (
                (signs @ projection)
                * q.residual_norms.item()
                * math.sqrt(math.pi / 2)
                / 8
                * q.norms.item()
            )
            direct = query.numpy() @ correction
            projected = (
                ((query.numpy() @ projection.T) @ signs)
                * q.residual_norms.item()
                * math.sqrt(math.pi / 2)
                / 8
                * q.norms.item()
            )
            assert abs(direct - projected) < 1e-12
    target = float(query @ x.double().flatten())
    standard_error = np.std(estimates, ddof=1) / math.sqrt(len(estimates))
    assert abs(np.mean(estimates) - target) < 6 * standard_error


def test_fresh_matching_codec_decodes_and_changed_identity_is_rejected():
    config = TurboQuantQuantizerConfig(variant="prod")
    codec = TurboQuantQuantizer(config)
    x = torch.randn(1, 1, 2, 16)
    q = codec.quantize(x, context=CONTEXT)
    assert torch.equal(codec.dequantize(q), TurboQuantQuantizer(config).dequantize(q))
    for changed in (
        replace(config, qjl_seed=2),
        replace(config, rotation=RandomRotationConfig(seed=1)),
        replace(config, codebook_distribution="gaussian_limit"),
        replace(config, variant="mse"),
        replace(config, bits=4),
    ):
        with pytest.raises(ValueError):
            TurboQuantQuantizer(changed).dequantize(q)


@pytest.mark.parametrize(
    "changes",
    [
        {"bits": 0},
        {"bits": True},
        {"bits": 5},
        {"variant": "bad"},
        {"codebook_distribution": "bad"},
        {"group_size": 64},
        {"symmetric": True},
        {"scale_dtype": ParamDType.FP32},
        {"offset_dtype": ParamDType.BF16},
        {"granularity": QuantGranularity.PER_CHANNEL},
        {"qjl_seed": True},
        {"qjl_seed": -1},
        {"rotation": RandomRotationConfig(seed=-1)},
    ],
)
def test_invalid_configurations_fail_early(changes):
    with pytest.raises(ValueError):
        TurboQuantQuantizer(TurboQuantQuantizerConfig(**changes))


def test_invalid_input_context_and_plans_are_rejected():
    codec = TurboQuantQuantizer(TurboQuantQuantizerConfig())
    x = torch.randn(1, 1, 2, 16)
    for context in (
        None,
        AttentionContext(-1, "key", 0),
        AttentionContext(True, "key", 0),
        AttentionContext(0, "bad", 0),
    ):
        with pytest.raises(ValueError):
            codec.quantize(x, context=context)
    for bad in (
        torch.ones(3),
        torch.ones(1, 1, 1, 1),
        x.double(),
        torch.full_like(x, float("nan")),
        torch.full_like(x, float("inf")),
        torch.ones_like(x, dtype=torch.int32),
    ):
        with pytest.raises(ValueError):
            codec.quantize(bad, context=CONTEXT)
    for plan in (PrecisionPlan(4), PrecisionPlan(True)):
        with pytest.raises(ValueError):
            codec.quantize(x, plan, context=CONTEXT)
    with pytest.raises(NotImplementedError):
        plan = PrecisionPlan(3, keep_tokens=torch.zeros(1, 1, 2, dtype=torch.bool))
        codec.quantize(x, plan, context=CONTEXT)
    # channel_bits (the outlier split) is supported, but only as two integer
    # widths: the configured bits and a larger one.
    for channel_bits in (
        torch.ones(1, 1, 16),
        torch.full((1, 1, 16), 3, dtype=torch.int64),
    ):
        with pytest.raises(ValueError):
            codec.quantize(x, PrecisionPlan(3, channel_bits=channel_bits), context=CONTEXT)


@pytest.mark.parametrize(
    "mutation",
    [
        "format",
        "bits",
        "shape",
        "dtype",
        "norm_none",
        "norm_shape",
        "norm_nan",
        "negative_norm",
        "norm_dtype",
        "residual_none",
        "residual_nan",
        "negative_residual",
        "sign_missing",
        "sign_size",
        "sign_padding",
        "index_size",
        "index_padding",
        "metadata",
        "extra",
        "zero_with_residual",
    ],
)
def test_invalid_records_do_not_silently_produce_attention(mutation):
    codec = TurboQuantQuantizer(TurboQuantQuantizerConfig(variant="prod", bits=2))
    q = codec.quantize(torch.randn(1, 1, 1, 3), context=CONTEXT)
    if mutation == "format":
        q = replace(q, format_id="wrong")
    elif mutation == "bits":
        q = replace(q, bits=3)
    elif mutation == "shape":
        q = replace(q, shape=(1, 1, 1, 4))
    elif mutation == "dtype":
        q = replace(q, dtype=torch.float64)
    elif mutation == "norm_none":
        q = replace(q, norms=None)
    elif mutation == "norm_shape":
        q = replace(q, norms=q.norms.expand(2, 1, 1, 1))
    elif mutation == "norm_nan":
        q = replace(q, norms=torch.full_like(q.norms, float("nan")))
    elif mutation == "negative_norm":
        q = replace(q, norms=-q.norms)
    elif mutation == "norm_dtype":
        q = replace(q, norms=q.norms.half())
    elif mutation == "residual_none":
        q = replace(q, residual_norms=None)
    elif mutation == "residual_nan":
        q = replace(q, residual_norms=torch.full_like(q.residual_norms, float("inf")))
    elif mutation == "negative_residual":
        q = replace(q, residual_norms=-q.residual_norms)
    elif mutation == "sign_missing":
        q = replace(q, qjl_codes=None)
    elif mutation == "sign_size":
        q = replace(q, qjl_codes=q.qjl_codes[:0])
    elif mutation == "sign_padding":
        q = replace(q, qjl_codes=q.qjl_codes | 128)
    elif mutation == "index_size":
        q = replace(q, codes=q.codes[:0])
    elif mutation == "index_padding":
        q = replace(q, codes=q.codes | 128)
    elif mutation == "metadata":
        q = replace(q, metadata={**q.metadata, "rotation_id": "wrong"})
    elif mutation == "extra":
        q = replace(q, codebook=torch.ones(2))
    else:
        q = replace(q, norms=torch.zeros_like(q.norms))
    with pytest.raises(ValueError):
        codec.dequantize(q)


def test_norm_extremes_and_final_finite_overflow_rule():
    codec = TurboQuantQuantizer(TurboQuantQuantizerConfig(variant="prod"))
    x = torch.zeros(1, 1, 2, 4)
    x[..., 0, :] = torch.finfo(torch.float32).smallest_normal / 1024
    x[..., 1, :] = 1e20
    q = codec.quantize(x, context=CONTEXT)
    assert (q.norms > 0).all() and torch.isfinite(codec.dequantize(q)).all()
    with pytest.raises(ValueError, match="norms"):
        codec.quantize(
            torch.full((1, 1, 1, 16), torch.finfo(torch.float32).max), context=CONTEXT
        )
    half_record = codec.quantize(torch.ones(1, 1, 1, 4).half(), context=CONTEXT)
    huge = replace(half_record, norms=torch.full_like(half_record.norms, 1e8))
    restored = codec.dequantize(huge)
    assert (
        torch.isfinite(restored).all()
        and restored.abs().max() == torch.finfo(torch.float16).max
    )
