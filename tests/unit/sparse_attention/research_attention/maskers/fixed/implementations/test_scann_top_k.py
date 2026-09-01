"""Tests for ScaNNTopKMasker."""

import pytest
import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    MaskerConfig,
    MaskerRegistry,
    ResearchMasker,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.base import (
    TopKMasker,
    TopKMaskerConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    OracleTopK,
    OracleTopKConfig,
    ScaNNTopKMasker,
    ScaNNTopKMaskerConfig,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask

HEAD_DIM = 8


def _config(**overrides) -> ScaNNTopKMaskerConfig:
    """A small but non-degenerate ScaNN configuration."""
    params = dict(
        heavy_size=3,
        num_leaves=4,
        num_leaves_to_search=4,
        dimensions_per_block=2,
        num_centers=4,
        kmeans_iter=3,
        ah_kmeans_iter=3,
        noise_shaping_rounds=3,
    )
    params.update(overrides)
    return ScaNNTopKMaskerConfig(**params)


def _inputs(
    batch_size=1,
    num_query_heads=2,
    num_kv_heads=2,
    seq_len_queries=4,
    seq_len_keys=32,
    previous="empty",
    seed=0,
):
    """Keys, queries, values and a previous Mask, mirroring the OracleTopK harness."""
    torch.manual_seed(seed)
    keys = torch.randn(batch_size, num_kv_heads, seq_len_keys, HEAD_DIM)
    queries = torch.randn(batch_size, num_query_heads, seq_len_queries, HEAD_DIM)
    values = torch.randn(batch_size, num_kv_heads, seq_len_keys, HEAD_DIM)
    shape = (batch_size, num_query_heads, seq_len_queries, seq_len_keys)
    if previous == "empty":
        mask = Mask.create_empty_mask(shape, dtype=torch.float32, device=keys.device)
    elif previous == "full":
        mask = Mask.create_full_mask(shape, dtype=torch.float32, device=keys.device)
    else:
        dense = torch.zeros(shape)
        dense[..., :2] = 1.0
        mask = Mask.create_mask_from_dense_mask(shape, dense, dtype=torch.float32)
    return keys, queries, values, mask


def _apply(masker, keys, queries, values, previous, attention_mask=None, meta=None):
    return masker.add_mask(
        keys=keys,
        queries=queries,
        values=values,
        attention_mask=attention_mask,
        scaling=1.0,
        dropout=0.0,
        sparse_meta_data={} if meta is None else meta,
        previous_mask=previous,
        layer_idx=0,
    )


@pytest.mark.unit
class TestScaNNTopKMaskerConfig:
    def test_config_creation(self) -> None:
        config = _config(heavy_size=0.05, anisotropic_threshold=0.3)
        assert config.heavy_size == 0.05
        assert config.anisotropic_threshold == 0.3
        assert config.dimensions_per_block == 2

    def test_config_inheritance(self) -> None:
        assert issubclass(ScaNNTopKMaskerConfig, TopKMaskerConfig)

    @pytest.mark.parametrize(
        "overrides,message",
        [
            ({"heavy_size": 0}, "heavy_size must be > 0"),
            ({"num_leaves": 0}, "num_leaves must be in"),
            ({"num_leaves": 40000, "num_leaves_to_search": 4}, "num_leaves must be in"),
            ({"num_leaves_to_search": 0}, "num_leaves_to_search must be in"),
            (
                {"num_leaves_to_search": 9, "num_leaves": 4},
                "num_leaves_to_search must be in",
            ),
            ({"dimensions_per_block": 0}, "dimensions_per_block must be > 0"),
            ({"num_centers": 1}, "num_centers must be in"),
            ({"num_centers": 512}, "num_centers must be in"),
            ({"anisotropic_threshold": 1.0}, "anisotropic_threshold must be None"),
            ({"anisotropic_threshold": 0.0}, "anisotropic_threshold must be None"),
            ({"kmeans_iter": 0}, "kmeans_iter and ah_kmeans_iter must be > 0"),
            ({"ah_kmeans_iter": 0}, "kmeans_iter and ah_kmeans_iter must be > 0"),
            ({"noise_shaping_rounds": -1}, "noise_shaping_rounds must be >= 0"),
            ({"reorder_size": -1}, "reorder_size must be >= 0"),
            ({"init_offset": -1}, "init_offset must be >= 0"),
        ],
    )
    def test_validation(self, overrides, message) -> None:
        with pytest.raises(ValueError, match=message):
            _config(**overrides)

    def test_threshold_may_be_disabled(self) -> None:
        assert _config(anisotropic_threshold=None).anisotropic_threshold is None


@pytest.mark.unit
class TestScaNNTopKMaskerCreation:
    def test_creation(self) -> None:
        config = _config()
        masker = ScaNNTopKMasker(config)
        assert type(masker) is ScaNNTopKMasker
        assert masker.config == config

    def test_creation_from_config(self) -> None:
        config = _config()
        assert type(ScaNNTopKMasker.create_from_config(config)) is ScaNNTopKMasker

    def test_creation_from_invalid_config(self) -> None:
        with pytest.raises(ValueError, match="Invalid config type"):
            ScaNNTopKMasker.create_from_config(MaskerConfig())

    def test_inheritance(self) -> None:
        assert issubclass(ScaNNTopKMasker, TopKMasker)

    def test_registry_dispatch(self) -> None:
        assert MaskerRegistry.get_masker_class(ScaNNTopKMaskerConfig) is ScaNNTopKMasker
        masker = ResearchMasker.create_masker_from_config(_config())
        assert type(masker) is ScaNNTopKMasker


@pytest.mark.unit
class TestScaNNTopKMaskerAddMask:
    def test_full_previous_mask_short_circuits(self) -> None:
        keys, queries, values, previous = _inputs(previous="full")
        result = _apply(ScaNNTopKMasker(_config()), keys, queries, values, previous)
        assert result.is_full_mask()
        assert result.shape == previous.shape

    def test_small_sequence_uses_full_attention(self) -> None:
        keys, queries, values, previous = _inputs(seq_len_keys=8)
        result = _apply(ScaNNTopKMasker(_config()), keys, queries, values, previous)
        assert result.is_full_mask()

    def test_small_sequence_accounts_for_the_codebook(self) -> None:
        """Both k-means stages need as many points as clusters, not just the tree."""
        keys, queries, values, previous = _inputs(seq_len_keys=12)
        masker = ScaNNTopKMasker(
            _config(heavy_size=1, num_leaves=2, num_leaves_to_search=2, num_centers=8)
        )
        assert _apply(masker, keys, queries, values, previous).is_full_mask()

    def test_requires_sparse_meta_data(self) -> None:
        keys, queries, values, previous = _inputs()
        with pytest.raises(ValueError, match="sparse_meta_data must be a dict"):
            ScaNNTopKMasker(_config()).add_mask(
                keys=keys,
                queries=queries,
                values=values,
                attention_mask=None,
                scaling=1.0,
                dropout=0.0,
                sparse_meta_data=None,
                previous_mask=previous,
                layer_idx=0,
            )

    def test_requires_layer_idx(self) -> None:
        keys, queries, values, previous = _inputs()
        with pytest.raises(ValueError, match="layer_idx must be provided"):
            ScaNNTopKMasker(_config()).add_mask(
                keys=keys,
                queries=queries,
                values=values,
                attention_mask=None,
                scaling=1.0,
                dropout=0.0,
                sparse_meta_data={},
                previous_mask=previous,
            )

    def test_rejects_indivisible_head_dim(self) -> None:
        keys, queries, values, previous = _inputs()
        masker = ScaNNTopKMasker(_config(dimensions_per_block=3))
        with pytest.raises(
            ValueError, match="must be divisible by dimensions_per_block"
        ):
            _apply(masker, keys, queries, values, previous)

    @pytest.mark.parametrize("heavy_size,expected", [(3, 3), (5, 5), (0.25, 8)])
    def test_exact_density(self, heavy_size, expected) -> None:
        keys, queries, values, previous = _inputs()
        result = _apply(
            ScaNNTopKMasker(_config(heavy_size=heavy_size)),
            keys,
            queries,
            values,
            previous,
        )
        assert result.shape == previous.shape
        assert torch.all((result.get_dense_mask() != 0).sum(-1) == expected)

    def test_merges_with_previous_without_reselecting(self) -> None:
        keys, queries, values, previous = _inputs(previous="partial")
        result = _apply(ScaNNTopKMasker(_config()), keys, queries, values, previous)
        dense = result.get_dense_mask()
        assert torch.all(dense[..., :2] == 1.0)
        assert torch.all((dense[..., 2:] != 0).sum(-1) == 3)

    def test_honours_init_offset(self) -> None:
        keys, queries, values, previous = _inputs()
        result = _apply(
            ScaNNTopKMasker(_config(init_offset=4)), keys, queries, values, previous
        )
        dense = result.get_dense_mask()
        assert torch.all(dense[..., :4] == 0.0)
        assert torch.all((dense != 0).sum(-1) == 3)

    def test_gqa(self) -> None:
        keys, queries, values, previous = _inputs(num_query_heads=8, num_kv_heads=2)
        result = _apply(ScaNNTopKMasker(_config()), keys, queries, values, previous)
        assert result.shape == previous.shape
        assert torch.all((result.get_dense_mask() != 0).sum(-1) == 3)

    def test_respects_a_causal_attention_mask(self) -> None:
        keys, queries, values, previous = _inputs(seq_len_queries=4, seq_len_keys=32)
        allowed = torch.arange(32).view(1, -1) < torch.arange(29, 33).view(-1, 1)
        attention_mask = torch.zeros(1, 1, 4, 32).masked_fill(~allowed, float("-inf"))
        result = _apply(
            ScaNNTopKMasker(_config()), keys, queries, values, previous, attention_mask
        )
        dense = result.get_dense_mask()
        assert torch.all((dense != 0).sum(-1) == 3)
        assert torch.all(dense.bool() <= allowed)

    def test_is_deterministic(self) -> None:
        keys, queries, values, previous = _inputs()
        masker = ScaNNTopKMasker(_config())
        first = _apply(masker, keys, queries, values, previous, meta={})
        _, _, _, previous_again = _inputs()
        second = _apply(masker, keys, queries, values, previous_again, meta={})
        assert torch.equal(first.get_dense_mask(), second.get_dense_mask())

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_leaf_gate_stays_above_the_selected_sentinel(self, dtype) -> None:
        """finfo.min must not underflow onto -inf, or gated keys tie with active ones."""
        keys, queries, values, _ = _inputs(seq_len_keys=64)
        keys, queries = (keys * 4).to(dtype), (queries * 4).to(dtype)
        shape = (1, 2, 4, 64)
        dense = torch.zeros(shape, dtype=dtype)
        dense[..., :2] = 1.0
        previous = Mask.create_mask_from_dense_mask(shape, dense, dtype=dtype)
        result = _apply(
            ScaNNTopKMasker(_config(num_leaves=8, num_leaves_to_search=1)),
            keys,
            queries,
            values,
            previous,
        )
        selected = result.get_dense_mask()
        assert torch.all(selected[..., :2] == 1.0)
        assert torch.all((selected[..., 2:] != 0).sum(-1) == 3)

    def test_rescoring_never_reselects_an_active_position(self) -> None:
        """The exact-rescore pass must preserve the already-selected sentinel."""
        keys, queries, values, previous = _inputs(seq_len_keys=64, previous="partial")
        result = _apply(
            ScaNNTopKMasker(_config(heavy_size=4, reorder_size=0.5)),
            keys,
            queries,
            values,
            previous,
        )
        dense = result.get_dense_mask()
        assert torch.all(dense[..., :2] == 1.0)
        assert torch.all((dense[..., 2:] != 0).sum(-1) == 4)

    def test_full_rescoring_reproduces_exact_top_k(self) -> None:
        """reorder_size covering every key must select exactly OracleTopK's positions."""
        keys, queries, values, previous = _inputs(seq_len_keys=64)
        scann = _apply(
            ScaNNTopKMasker(_config(heavy_size=5, reorder_size=1.0)),
            keys,
            queries,
            values,
            previous,
        )
        _, _, _, oracle_previous = _inputs(seq_len_keys=64)
        oracle = OracleTopK(OracleTopKConfig(heavy_size=5)).add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=None,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data={},
            previous_mask=oracle_previous,
        )
        assert torch.equal(scann.get_dense_mask(), oracle.get_dense_mask())

    def test_leaf_gating_restricts_candidates(self) -> None:
        """Probing one leaf must confine the selection to that leaf's keys."""
        keys, queries, values, previous = _inputs(
            seq_len_keys=64, num_query_heads=1, num_kv_heads=1
        )
        meta: dict = {}
        result = _apply(
            ScaNNTopKMasker(
                _config(heavy_size=2, num_leaves=8, num_leaves_to_search=1)
            ),
            keys,
            queries,
            values,
            previous,
            meta=meta,
        )
        leaves = meta["scann"][0]["leaves"][0]
        picked = result.get_dense_mask()[0, 0].nonzero(as_tuple=True)[1].view(4, 2)
        probed = leaves[picked]
        assert torch.all(probed[:, 0] == probed[:, 1])


@pytest.mark.unit
class TestScaNNTopKMaskerIndexState:
    def test_index_is_built_once_and_extended(self) -> None:
        keys, queries, values, previous = _inputs(seq_len_keys=64)
        masker = ScaNNTopKMasker(_config())
        meta: dict = {}
        _apply(masker, keys, queries, values, previous, meta=meta)
        state = meta["scann"][0]
        centroids = state["centroids"].clone()
        assert state["codes"].shape == (2, 64, HEAD_DIM // 2)
        assert state["leaves"].shape == (2, 64)
        assert state["codes"].dtype == torch.uint8
        assert state["leaves"].dtype == torch.int16

        grown_keys = torch.cat([keys, torch.randn(1, 2, 3, HEAD_DIM)], dim=2)
        _, decode_queries, _, _ = _inputs(seq_len_queries=1, seq_len_keys=67)
        decode_previous = Mask.create_empty_mask(
            (1, 2, 1, 67), dtype=torch.float32, device=keys.device
        )
        result = _apply(
            masker, grown_keys, decode_queries, values, decode_previous, meta=meta
        )
        assert meta["scann"][0]["codes"].shape[1] == 67
        assert meta["scann"][0]["leaves"].shape[1] == 67
        assert torch.equal(meta["scann"][0]["centroids"], centroids)
        assert torch.all((result.get_dense_mask() != 0).sum(-1) == 3)

    def test_state_is_separate_per_layer(self) -> None:
        keys, queries, values, previous = _inputs(seq_len_keys=64)
        masker = ScaNNTopKMasker(_config())
        meta: dict = {}
        _apply(masker, keys, queries, values, previous, meta=meta)
        _, _, _, second_previous = _inputs(seq_len_keys=64)
        masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=None,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data=meta,
            previous_mask=second_previous,
            layer_idx=1,
        )
        assert set(meta["scann"]) == {0, 1}

    def test_shrinking_key_count_raises(self) -> None:
        keys, queries, values, previous = _inputs(seq_len_keys=64)
        masker = ScaNNTopKMasker(_config())
        meta: dict = {}
        _apply(masker, keys, queries, values, previous, meta=meta)
        smaller = keys[:, :, :40, :]
        smaller_previous = Mask.create_empty_mask(
            (1, 2, 4, 40), dtype=torch.float32, device=keys.device
        )
        with pytest.raises(ValueError, match="Key count shrank"):
            _apply(masker, smaller, queries, values, smaller_previous, meta=meta)
