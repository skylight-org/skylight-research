"""
:summary: Tests for the PQImportance (PQ importance sampling) masker.
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

        config = PQImportanceConfig(heavy_size=10, sample_size=20, **_pq_kwargs(4))
        assert config.heavy_size == 10
        assert config.sample_size == 20
        assert config.temperature == 1.0

    def test_config_allows_zero_heavy_size(self):
        """Pure importance sampling is valid, unlike for other top-k maskers."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQImportanceConfig,
        )

        config = PQImportanceConfig(heavy_size=0, sample_size=0.1, **_pq_kwargs())
        assert config.heavy_size == 0

    def test_config_validation(self):
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQImportanceConfig,
        )

        with pytest.raises(ValueError):
            PQImportanceConfig(heavy_size=0, sample_size=0, **_pq_kwargs())
        with pytest.raises(ValueError):
            PQImportanceConfig(heavy_size=-1, sample_size=10, **_pq_kwargs())
        with pytest.raises(ValueError):
            PQImportanceConfig(heavy_size=10, sample_size=-1, **_pq_kwargs())
        with pytest.raises(ValueError):
            PQImportanceConfig(
                heavy_size=10, sample_size=10, temperature=0.0, **_pq_kwargs()
            )
        with pytest.raises(ValueError):
            PQImportanceConfig(
                heavy_size=10,
                sample_size=10,
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

        config = PQImportanceConfig(heavy_size=10, sample_size=10, **_pq_kwargs(4))
        masker = PQImportance.create_from_config(config)
        assert type(masker) is PQImportance
        assert isinstance(masker, PQCache)
        assert isinstance(masker, TopKMasker)
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
    def _add_mask(masker, keys, queries, values, scaling, meta, attention_mask=None):
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        shape = (
            queries.shape[0],
            queries.shape[1],
            queries.shape[2],
            keys.shape[2],
        )
        previous_mask = Mask.create_empty_mask(
            shape, dtype=torch.float32, device=keys.device
        )
        return masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=0.0,
            sparse_meta_data=meta,
            previous_mask=previous_mask,
            layer_idx=0,
        )

    def test_mask_values_are_probabilities_and_within_budget(self):
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQImportance,
            PQImportanceConfig,
        )

        keys, queries, values, scaling = self._setup()
        masker = PQImportance(
            PQImportanceConfig(heavy_size=16, sample_size=16, **_pq_kwargs(8))
        )
        mask = self._add_mask(masker, keys, queries, values, scaling, {})

        dense = mask.get_dense_mask()
        active = dense > 0
        assert dense.max() <= 1.0
        assert dense.min() >= 0.0
        # heavy positions are kept with probability one
        assert (dense == 1.0).any()
        # with-replacement sampling can only produce fewer distinct positions
        num_active = active.sum(dim=-1)
        assert bool((num_active <= 16 + 16).all())
        assert bool((num_active > 16).all())
        # nothing is selected inside the sink (init_offset) region
        assert not bool(active[:, :, :, :8].any())

    def test_sample_size_zero_matches_pq_cache(self):
        """heavy-only configuration degenerates to plain PQCache top-k."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
            PQImportance,
            PQImportanceConfig,
        )

        keys, queries, values, scaling = self._setup()
        importance = PQImportance(
            PQImportanceConfig(heavy_size=32, sample_size=0, **_pq_kwargs(8))
        )
        top_k = PQCache(PQCacheConfig(heavy_size=32, **_pq_kwargs(8)))

        torch.manual_seed(1)
        mask_importance = self._add_mask(
            importance, keys, queries, values, scaling, {}
        ).get_dense_mask()
        torch.manual_seed(1)
        mask_top_k = self._add_mask(
            top_k, keys, queries, values, scaling, {}
        ).get_dense_mask()

        assert torch.equal(mask_importance, mask_top_k)

    def test_masked_out_keys_are_never_selected(self):
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQImportance,
            PQImportanceConfig,
        )

        keys, queries, values, scaling = self._setup()
        attention_mask = torch.zeros(1, 1, queries.shape[2], keys.shape[2])
        attention_mask[..., -32:] = float("-inf")

        masker = PQImportance(
            PQImportanceConfig(heavy_size=16, sample_size=16, **_pq_kwargs(8))
        )
        mask = self._add_mask(
            masker, keys, queries, values, scaling, {}, attention_mask=attention_mask
        )
        assert not bool((mask.get_dense_mask()[..., -32:] > 0).any())

    def test_full_attention_for_short_sequences(self):
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQImportance,
            PQImportanceConfig,
        )

        keys, queries, values, scaling = self._setup(seq_len_keys=32)
        masker = PQImportance(
            PQImportanceConfig(heavy_size=8, sample_size=8, **_pq_kwargs())
        )
        mask = self._add_mask(masker, keys, queries, values, scaling, {})
        assert mask.is_full_mask()

    def test_incremental_decoding_reuses_centroids(self):
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQImportance,
            PQImportanceConfig,
        )

        keys, queries, values, scaling = self._setup()
        masker = PQImportance(
            PQImportanceConfig(heavy_size=16, sample_size=16, **_pq_kwargs(8))
        )
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

    def test_estimator_is_unbiased(self):
        """1/inclusion-probability weighting must recover the softmax denominator.

        Top-k truncation systematically under-estimates it; importance sampling
        should not.
        """
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
            PQImportance,
            PQImportanceConfig,
        )

        keys, queries, values, scaling = self._setup(seq_len_keys=512, seed=3)
        logits = torch.matmul(queries, keys.transpose(-2, -1)) * scaling
        exp_weights = torch.exp(logits - logits.max(dim=-1, keepdim=True).values)
        true_denominator = exp_weights.sum(dim=-1)

        def mean_denominator(masker, trials):
            meta = {}
            estimates = []
            for _ in range(trials):
                mask = self._add_mask(masker, keys, queries, values, scaling, meta)
                estimates.append(mask.apply_inv_mask(exp_weights).sum(dim=-1))
            return torch.stack(estimates).mean(dim=0)

        torch.manual_seed(7)
        importance = PQImportance(
            PQImportanceConfig(heavy_size=16, sample_size=48, **_pq_kwargs())
        )
        importance_ratio = (mean_denominator(importance, 200) / true_denominator).mean()
        top_k = PQCache(PQCacheConfig(heavy_size=64, **_pq_kwargs()))
        top_k_ratio = (mean_denominator(top_k, 1) / true_denominator).mean()

        assert abs(float(importance_ratio) - 1.0) < 0.05
        assert float(top_k_ratio) < float(importance_ratio)
