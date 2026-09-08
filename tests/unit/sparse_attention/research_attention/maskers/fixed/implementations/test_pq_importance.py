"""
:summary: Tests for the PQImportance (categorical-with-replacement PQCache) masker.
"""

import pytest
import torch


def _pq_kwargs(init_offset: int = 0):
    return dict(
        pq_group_factor=2,
        pq_bits=4,
        kmeans_iter=3,
        init_offset=init_offset,
        metric="euclidean",
    )


@pytest.mark.unit
class TestPQImportanceConfig:
    def test_config_creation(self):
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQImportanceConfig,
        )

        config = PQImportanceConfig(heavy_size=10, **_pq_kwargs(4))
        assert config.heavy_size == 10
        assert config.temperature == 1.0

        config = PQImportanceConfig(heavy_size=0.1, temperature=2.5, **_pq_kwargs())
        assert config.temperature == 2.5

    def test_config_validation(self):
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQImportanceConfig,
        )

        with pytest.raises(ValueError):
            PQImportanceConfig(heavy_size=0, **_pq_kwargs())
        with pytest.raises(ValueError):
            PQImportanceConfig(heavy_size=10, temperature=-1.0, **_pq_kwargs())
        with pytest.raises(ValueError):
            PQImportanceConfig(
                heavy_size=10,
                pq_group_factor=2,
                pq_bits=4,
                kmeans_iter=3,
                init_offset=0,
                metric="cosine",
            )

    def test_masker_creation_and_registry(self):
        from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
            ResearchMasker,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            TopKMasker,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQImportance,
            PQImportanceConfig,
        )

        config = PQImportanceConfig(heavy_size=10, **_pq_kwargs(4))
        masker = PQImportance.create_from_config(config)
        assert type(masker) is PQImportance
        assert isinstance(masker, PQCache)
        assert isinstance(masker, TopKMasker)
        assert masker.temperature == 1.0
        # registry dispatches on the config type
        assert type(ResearchMasker.create_masker_from_config(config)) is PQImportance

    def test_create_from_config_rejects_other_configs(self):
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCacheConfig,
            PQImportance,
        )

        with pytest.raises(ValueError):
            PQImportance.create_from_config(
                PQCacheConfig(heavy_size=10, **_pq_kwargs(4))
            )


@pytest.mark.unit
class TestCategoricalInclusion:
    """iid categorical draws + exact pi_i = 1 - (1 - p_i)^m."""

    def test_inclusion_matches_closed_form(self):
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.gumbel_utils import (
            sample_categorical_inclusion,
        )

        torch.manual_seed(0)
        logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
        m = 5
        p = torch.softmax(logits, dim=-1)
        expected = 1.0 - (1.0 - p) ** m
        _, _, _, dense = sample_categorical_inclusion(logits, m, temperature=1.0)
        selected = dense > 0
        assert bool(selected.any())
        assert torch.allclose(dense[selected], expected[selected], atol=1e-5)
        assert bool((dense[~selected] == 0).all())

    def test_empirical_inclusion_matches_formula(self):
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.gumbel_utils import (
            sample_categorical_inclusion,
        )

        torch.manual_seed(0)
        logits = torch.tensor([2.0, 1.0, 0.0, -1.0])
        m = 3
        p = torch.softmax(logits, dim=-1)
        expected = 1.0 - (1.0 - p) ** m
        trials = 20_000
        batched = logits.expand(trials, -1)
        _, _, _, dense = sample_categorical_inclusion(batched, m, temperature=1.0)
        empirical = (dense > 0).float().mean(dim=0)
        assert torch.allclose(empirical, expected, atol=0.02)

    def test_masked_logits_are_never_sampled(self):
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.gumbel_utils import (
            sample_categorical_inclusion,
        )

        torch.manual_seed(0)
        scores = torch.tensor([[1.0, float("-inf"), 0.5, float("-inf")]])
        _, _, _, dense = sample_categorical_inclusion(scores, 8, temperature=1.0)
        assert bool((dense[..., 1] == 0).all())
        assert bool((dense[..., 3] == 0).all())
        assert bool((dense[..., [0, 2]] > 0).any())

    def test_zero_temperature_is_deterministic_topk(self):
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.gumbel_utils import (
            sample_categorical_inclusion,
        )

        scores = torch.tensor([[0.1, 4.0, 2.0, 3.0, -1.0]])
        _, _, _, dense = sample_categorical_inclusion(scores, 2, temperature=0.0)
        assert torch.equal(dense, torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0]]))


@pytest.mark.unit
class TestPQImportanceMask:
    """Behaviour of the produced mask."""

    @staticmethod
    def _setup(seq_len_keys=256, num_heads=2, seq_len_queries=4, head_dim=16, seed=0):
        torch.manual_seed(seed)
        keys = torch.randn(1, num_heads, seq_len_keys, head_dim)
        # queries near random keys => peaked (realistic) attention
        idx = torch.randint(0, seq_len_keys, (seq_len_queries,))
        queries = 2.0 * keys[:, :, idx, :] + 0.5 * torch.randn(
            1, num_heads, seq_len_queries, head_dim
        )
        values = torch.randn(1, num_heads, seq_len_keys, head_dim)
        return keys, queries, values, head_dim**-0.5

    @staticmethod
    def _add_mask(masker, keys, queries, values, scaling, meta):
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        shape = (queries.shape[0], queries.shape[1], queries.shape[2], keys.shape[2])
        previous_mask = Mask.create_empty_mask(
            shape, dtype=torch.float32, device=keys.device
        )
        return masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=None,
            scaling=scaling,
            dropout=0.0,
            sparse_meta_data=meta,
            previous_mask=previous_mask,
            layer_idx=0,
        )

    def test_mask_is_weighted_and_within_budget(self):
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQImportance,
            PQImportanceConfig,
        )

        keys, queries, values, scaling = self._setup()
        masker = PQImportance(PQImportanceConfig(heavy_size=32, **_pq_kwargs(8)))
        dense = self._add_mask(
            masker, keys, queries, values, scaling, {}
        ).get_dense_mask()

        active = dense > 0
        # inclusion probabilities on the unique set: (0, 1], |S| <= m
        selected = dense[active]
        assert bool((selected > 0).all())
        assert bool((selected <= 1.0).all())
        assert bool((active.sum(dim=-1) <= 32).all())
        assert bool((active.sum(dim=-1) > 0).all())
        # nothing is selected inside the sink (init_offset) region
        assert not bool(active[:, :, :, :8].any())

    def test_zero_temperature_matches_pq_cache(self):
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
            PQImportance,
            PQImportanceConfig,
        )

        keys, queries, values, scaling = self._setup()
        sampled = PQImportance(
            PQImportanceConfig(heavy_size=32, temperature=0.0, **_pq_kwargs(8))
        )
        top_k = PQCache(PQCacheConfig(heavy_size=32, **_pq_kwargs(8)))

        torch.manual_seed(1)
        mask_sampled = self._add_mask(
            sampled, keys, queries, values, scaling, {}
        ).get_dense_mask()
        torch.manual_seed(1)
        mask_top_k = self._add_mask(
            top_k, keys, queries, values, scaling, {}
        ).get_dense_mask()

        assert torch.equal(mask_sampled, mask_top_k)

    def test_temperature_makes_selection_stochastic(self):
        """Different draws pick different keys, but stay near the top scores."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
            PQImportance,
            PQImportanceConfig,
        )

        keys, queries, values, scaling = self._setup()
        masker = PQImportance(
            PQImportanceConfig(heavy_size=32, temperature=1.0, **_pq_kwargs(8))
        )
        meta = {}
        torch.manual_seed(2)
        first = self._add_mask(
            masker, keys, queries, values, scaling, meta
        ).get_dense_mask()
        second = self._add_mask(
            masker, keys, queries, values, scaling, meta
        ).get_dense_mask()
        assert not torch.equal(first, second)

        top_k = PQCache(PQCacheConfig(heavy_size=32, **_pq_kwargs(8)))
        deterministic = self._add_mask(
            top_k, keys, queries, values, scaling, {}
        ).get_dense_mask()
        # unique |S| <= m; still better than a uniform draw of the same cardinality
        n_scored = keys.shape[2] - 8
        n_sampled = first.bool().sum(dim=-1).float()
        overlap = (first.bool() & deterministic.bool()).sum(dim=-1).float()
        expected_random = n_sampled * 32.0 / n_scored
        assert float(overlap.mean()) > float(expected_random.mean())

    def test_full_attention_for_short_sequences(self):
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQImportance,
            PQImportanceConfig,
        )

        keys, queries, values, scaling = self._setup(seq_len_keys=32)
        masker = PQImportance(PQImportanceConfig(heavy_size=16, **_pq_kwargs()))
        mask = self._add_mask(masker, keys, queries, values, scaling, {})
        assert mask.is_full_mask()

    def test_incremental_decoding_reuses_centroids(self):
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQImportance,
            PQImportanceConfig,
        )

        keys, queries, values, scaling = self._setup()
        masker = PQImportance(PQImportanceConfig(heavy_size=32, **_pq_kwargs(8)))
        meta = {}
        self._add_mask(masker, keys, queries, values, scaling, meta)
        centroids = meta["pq_centroids"][0]

        new_keys = torch.cat([keys, torch.randn(1, keys.shape[1], 4, keys.shape[3])], 2)
        new_values = torch.cat(
            [values, torch.randn(1, values.shape[1], 4, values.shape[3])], 2
        )
        new_queries = torch.randn(1, queries.shape[1], 1, queries.shape[3])
        mask = self._add_mask(masker, new_keys, new_queries, new_values, scaling, meta)

        assert torch.equal(meta["pq_centroids"][0], centroids)
        assert meta["pq_codebook"][0].shape[1] == new_keys.shape[2] - 8
        assert mask.shape == (1, queries.shape[1], 1, new_keys.shape[2])

    def test_previously_selected_keys_are_not_reselected(self):
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQImportance,
            PQImportanceConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        keys, queries, values, scaling = self._setup()
        shape = (queries.shape[0], queries.shape[1], queries.shape[2], keys.shape[2])
        previous_dense = torch.zeros(shape, dtype=torch.float32)
        previous_dense[..., 8:24] = 1.0
        previous_mask = Mask.create_mask_from_dense_mask(
            shape, previous_dense, dtype=torch.float32
        )

        masker = PQImportance(PQImportanceConfig(heavy_size=32, **_pq_kwargs(8)))
        mask = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=None,
            scaling=scaling,
            dropout=0.0,
            sparse_meta_data={},
            previous_mask=previous_mask,
            layer_idx=0,
        )

        dense = mask.get_dense_mask()
        active = dense > 0
        # the 16 pre-selected keys are kept; with-replacement unique set is <= 32
        assert bool(active[..., 8:24].all())
        assert bool((dense[..., 8:24] == 1.0).all())
        newly = (previous_dense == 0) & active
        assert bool(newly.any())
        assert not bool(newly[..., 8:24].any())
        assert bool((newly.sum(dim=-1) <= 32).all())
        assert bool((active.sum(dim=-1) <= 16 + 32).all())
        sampled = dense[newly]
        assert bool((sampled > 0).all() and (sampled <= 1.0).all())
