"""Native attention visibility stays aligned with independently retained history."""

import pytest
import torch

from sparse_attention_hub.kv_quantization.attention_mask import align_attention_mask

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("kind", ["none", "bool", "float"])
def test_sliding_visibility_and_causality_on_a_longer_history(kind):
    mask = None
    if kind == "bool":
        mask = torch.ones(1, 1, 2, 5, dtype=torch.bool)
        mask[..., 1, 3] = False
    elif kind == "float":
        mask = torch.zeros(1, 1, 2, 5)
        mask[..., 1, 3] = float("-inf")
    result = align_attention_mask(
        mask,
        source_key_length=5,
        target_key_length=10,
        query_length=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
        sliding_window=4,
    )
    visible = result if result.dtype == torch.bool else torch.isfinite(result)
    expected = torch.zeros(1, 1, 2, 10, dtype=torch.bool)
    expected[..., 0, 5:9] = True
    expected[..., 1, 6:10] = True
    if kind != "none":
        expected[..., 1, 8] = False
    assert torch.equal(visible, expected)


def test_additive_bias_and_batched_padding_are_preserved():
    mask = torch.zeros(2, 1, 1, 4)
    mask[0, :, :, 1] = float("-inf")
    mask[1, :, :, 2] = -0.75
    result = align_attention_mask(
        mask,
        source_key_length=4,
        target_key_length=8,
        query_length=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert result.shape == (2, 1, 2, 8)
    assert torch.isneginf(result[..., :4]).all()
    assert torch.isneginf(result[0, :, :, 5]).all()
    assert (result[1, :, :, 6] == -0.75).all()
    assert torch.isneginf(result[..., 0, 7]).all()  # future current-call token


def test_full_width_source_mask_is_mapped_to_the_actual_hf_suffix():
    result = align_attention_mask(
        torch.zeros(1, 1, 1, 9),
        source_key_length=4,
        target_key_length=9,
        query_length=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert torch.isneginf(result[..., :5]).all()
    assert (result[..., 5:] == 0).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query_length": 0},
        {"source_key_length": 11},
        {"sliding_window": 0},
    ],
)
def test_invalid_mask_dimensions_fail_explicitly(kwargs):
    options = dict(
        source_key_length=4,
        target_key_length=10,
        query_length=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    options.update(kwargs)
    with pytest.raises(ValueError):
        align_attention_mask(None, **options)


@pytest.mark.parametrize(
    "mask",
    [
        torch.zeros(1, 2, 4),
        torch.zeros(1, 1, 3, 4),
        torch.zeros(1, 1, 2, 6),
        torch.zeros(1, 1, 2, 4, dtype=torch.int32),
    ],
)
def test_unsupported_native_mask_shapes_and_dtypes_fail_explicitly(mask):
    with pytest.raises(ValueError):
        align_attention_mask(
            mask,
            source_key_length=4,
            target_key_length=10,
            query_length=2,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
