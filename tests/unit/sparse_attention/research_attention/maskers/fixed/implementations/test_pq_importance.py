"""
:summary: Tests for the PQImportance (Gumbel-sampled PQCache) masker.
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
class TestGumbelNoise:
    """The Gumbel-top-k trick itself, independent of PQ."""

    def test_noise_matches_gumbel_distribution(self):
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.pq_importance import (
            _sample_gumbel_noise,
        )

        torch.manual_seed(0)
        noise = _sample_gumbel_noise(torch.zeros(200_000))
        assert torch.isfinite(noise).all()
        # Gumbel(0, 1): mean = Euler-Mascheroni, std = pi / sqrt(6)
        assert abs(float(noise.mean()) - 0.5772) < 0.02
        assert abs(float(noise.std()) - 1.2825) < 0.02

    def test_noise_keeps_score_dtype(self):
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.pq_importance import (
            _sample_gumbel_noise,
        )

        scores = torch.zeros(4, 8, dtype=torch.float16)
        noise = _sample_gumbel_noise(scores)
        assert noise.dtype == torch.float16
        assert noise.shape == scores.shape
        assert torch.isfinite(noise).all()

    def test_argmax_of_perturbed_logits_follows_softmax(self):
        """Gumbel-top-1 samples from softmax(logits); the point of the masker."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.pq_importance import (
            _sample_gumbel_noise,
        )

        torch.manual_seed(0)
        logits = torch.tensor([2.0, 1.0, 0.0, -1.0])
        trials = 40_000
        batched = logits.expand(trials, -1)
        picks = (batched + _sample_gumbel_noise(batched)).argmax(dim=-1)
        empirical = torch.bincount(picks, minlength=logits.numel()) / trials
        assert torch.allclose(empirical, torch.softmax(logits, dim=-1), atol=0.01)


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

    def test_mask_is_binary_and_within_budget(self):
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
        # selection is a plain top-k mask; only the ranking is randomised
        assert set(dense.unique().tolist()) == {0.0, 1.0}
        assert bool((active.sum(dim=-1) == 32).all())
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
        # sampling still concentrates on the keys top-k would have chosen
        overlap = (first.bool() & deterministic.bool()).sum(dim=-1).float().mean()
        assert float(overlap) > 16

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

        active = mask.get_dense_mask() > 0
        # the 16 pre-selected keys are kept and 32 fresh ones are added
        assert bool(active[..., 8:24].all())
        assert bool((active.sum(dim=-1) == 16 + 32).all())
