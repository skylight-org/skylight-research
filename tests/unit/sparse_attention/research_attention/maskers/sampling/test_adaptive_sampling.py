"""Tests for AdaptiveSamplingMasker implementation."""

import pytest
import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations.adaptive_sampling import (
    AdaptiveSamplingMasker,
    AdaptiveSamplingMaskerConfig,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask


@pytest.mark.unit
class TestAdaptiveSamplingMaskerConfig:
    """Test AdaptiveSamplingMaskerConfig validation."""

    def test_valid_float_config(self):
        """Test valid configuration with float base_rate_sampling."""
        config = AdaptiveSamplingMaskerConfig(
            base_rate_sampling=0.5,
            epsilon=0.1,
            delta=0.05,
            init_offset=0,
            local_offset=0,
        )
        assert config.base_rate_sampling == 0.5
        assert config.epsilon == 0.1
        assert config.delta == 0.05

    def test_valid_int_config(self):
        """Test valid configuration with int base_rate_sampling."""
        config = AdaptiveSamplingMaskerConfig(
            base_rate_sampling=10,
            epsilon=0.1,
            delta=0.05,
            init_offset=0,
            local_offset=0,
        )
        assert config.base_rate_sampling == 10

    def test_valid_zero_float_base_rate_sampling(self):
        """Test valid float base_rate_sampling with 0.0."""
        config = AdaptiveSamplingMaskerConfig(
            base_rate_sampling=0.0,
            epsilon=0.1,
            delta=0.05,
            init_offset=0,
            local_offset=0,
        )
        assert config.base_rate_sampling == 0.0

    def test_valid_zero_int_base_rate_sampling(self):
        """Test valid int base_rate_sampling with 0."""
        config = AdaptiveSamplingMaskerConfig(
            base_rate_sampling=0,
            epsilon=0.1,
            delta=0.05,
            init_offset=0,
            local_offset=0,
        )
        assert config.base_rate_sampling == 0

    def test_invalid_float_base_rate_sampling(self):
        """Test invalid float base_rate_sampling values."""
        with pytest.raises(
            ValueError, match="base_rate_sampling must be in \\[0, 1\\) if float"
        ):
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=1.0,
                epsilon=0.1,
                delta=0.05,
                init_offset=0,
                local_offset=0,
            )

    def test_invalid_int_base_rate_sampling(self):
        """Test invalid int base_rate_sampling values."""
        with pytest.raises(
            ValueError, match="base_rate_sampling must be non-negative if int"
        ):
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=-1,
                epsilon=0.1,
                delta=0.05,
                init_offset=0,
                local_offset=0,
            )

    def test_invalid_epsilon(self):
        """Test invalid epsilon values."""
        with pytest.raises(ValueError, match="epsilon must be in \\(0, 1\\)"):
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.5,
                epsilon=0.0,
                delta=0.05,
                init_offset=0,
                local_offset=0,
            )

        with pytest.raises(ValueError, match="epsilon must be in \\(0, 1\\)"):
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.5,
                epsilon=1.0,
                delta=0.05,
                init_offset=0,
                local_offset=0,
            )

    def test_invalid_delta(self):
        """Test invalid delta values."""
        with pytest.raises(ValueError, match="delta must be in \\(0, 1\\)"):
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.5,
                epsilon=0.1,
                delta=0.0,
                init_offset=0,
                local_offset=0,
            )

        with pytest.raises(ValueError, match="delta must be in \\(0, 1\\)"):
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.5,
                epsilon=0.1,
                delta=1.0,
                init_offset=0,
                local_offset=0,
            )

    def test_invalid_offsets(self):
        """Test invalid offset values."""
        with pytest.raises(ValueError, match="init_offset must be non-negative"):
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.5,
                epsilon=0.1,
                delta=0.05,
                init_offset=-1,
                local_offset=0,
            )

        with pytest.raises(ValueError, match="local_offset must be non-negative"):
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.5,
                epsilon=0.1,
                delta=0.05,
                init_offset=0,
                local_offset=-1,
            )


@pytest.mark.unit
class TestAdaptiveSamplingMasker:
    """Test AdaptiveSamplingMasker implementation."""

    @pytest.fixture
    def config(self):
        """Create a valid configuration for testing."""
        return AdaptiveSamplingMaskerConfig(
            base_rate_sampling=0.1,
            epsilon=0.1,
            delta=0.05,
            init_offset=0,
            local_offset=0,
        )

    @pytest.fixture
    def masker(self, config):
        """Create an AdaptiveSamplingMasker instance."""
        return AdaptiveSamplingMasker(config)

    @pytest.fixture
    def sample_tensors(self):
        """Create sample tensors for testing."""
        batch_size, num_heads, seq_len_queries, seq_len_keys, head_dim = 2, 4, 8, 16, 32

        keys = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, head_dim)
        values = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)
        attention_mask = torch.zeros(
            batch_size, num_heads, seq_len_queries, seq_len_keys
        )

        return keys, queries, values, attention_mask

    def test_init(self, config):
        """Test masker initialization."""
        masker = AdaptiveSamplingMasker(config)
        assert masker.base_rate_sampling == 0.1
        assert masker.epsilon == 0.1
        assert masker.delta == 0.05
        assert masker.init_offset == 0
        assert masker.local_offset == 0
        assert isinstance(masker.delta_ppf, float)
        assert masker.delta_ppf > 0

    def test_compute_exp_attention_scores(self, masker, sample_tensors):
        """Test exponential attention scores computation."""
        keys, queries, _, _ = sample_tensors

        exp_scores = masker._compute_exp_attention_scores(
            queries, keys, scaling=1.0, attention_mask=None
        )

        assert exp_scores.shape == (2, 4, 8, 16)
        assert torch.all(exp_scores >= 0)  # Exponential should be non-negative
        assert torch.all(torch.isfinite(exp_scores))  # Should be finite

    def test_get_sampling_range(self, masker):
        """Test sampling range calculation."""
        seq_len_keys = 16

        start_idx, end_idx, sampling_range = masker._get_sampling_range(seq_len_keys)

        assert start_idx == 0
        assert end_idx == 16
        assert sampling_range == 16

    def test_get_sampling_range_with_offsets(self):
        """Test sampling range with non-zero offsets."""
        config = AdaptiveSamplingMaskerConfig(
            base_rate_sampling=0.1,
            epsilon=0.1,
            delta=0.05,
            init_offset=2,
            local_offset=3,
        )
        masker = AdaptiveSamplingMasker(config)

        start_idx, end_idx, sampling_range = masker._get_sampling_range(16)

        assert start_idx == 2
        assert end_idx == 13
        assert sampling_range == 11

    def test_get_sampling_range_invalid(self):
        """Test invalid sampling range returns full mask."""
        config = AdaptiveSamplingMaskerConfig(
            base_rate_sampling=0.1,
            epsilon=0.1,
            delta=0.05,
            init_offset=10,
            local_offset=10,
        )
        masker = AdaptiveSamplingMasker(config)

        # Test that _get_sampling_range returns a negative sampling range
        start_idx, end_idx, sampling_range = masker._get_sampling_range(16)
        assert sampling_range == -4  # 6 - 10 = -4

        # Test that should_return_full_mask returns True for negative sampling range
        assert masker.should_return_full_mask(sampling_range) is True

    def test_get_base_sample_count_float(self, masker):
        """Test base sample count calculation with float."""
        sampling_range = 1000
        count = masker._get_base_sample_count(sampling_range)
        expected = int(0.1 * 1000)  # 0.1 * 1000 = 100 -> int(100) = 100
        assert count == expected

    def test_get_base_sample_count_int(self):
        """Test base sample count calculation with int."""
        config = AdaptiveSamplingMaskerConfig(
            base_rate_sampling=5,
            epsilon=0.1,
            delta=0.05,
            init_offset=0,
            local_offset=0,
        )
        masker = AdaptiveSamplingMasker(config)

        sampling_range = 16
        count = masker._get_base_sample_count(sampling_range)
        assert count == 5

    def test_get_std_estimate_using_base_sample(self, masker, sample_tensors):
        """Test standard deviation estimation using base sampling."""
        batch_size, num_heads, seq_len_queries, seq_len_keys = 2, 4, 8, 1024
        expwts = torch.randn(batch_size, num_heads, seq_len_queries, seq_len_keys)

        start_idx, end_idx = 0, seq_len_keys
        num_base_samples = 5
        dtype = torch.float32

        base_mask, std_estimate = masker._get_std_estimate_using_base_sample(
            expwts,
            batch_size,
            num_heads,
            seq_len_queries,
            seq_len_keys,
            start_idx,
            end_idx,
            num_base_samples,
            dtype,
        )

        assert isinstance(base_mask, Mask)
        assert base_mask.shape == (batch_size, num_heads, seq_len_queries, seq_len_keys)
        assert std_estimate.shape == (2, 4, 8, 1)
        assert torch.all(std_estimate >= 1e-8)  # Should be clamped to minimum

        dense_mask = base_mask.get_dense_mask()
        dense_mask_2d = dense_mask.view(-1, seq_len_keys)
        std_estimate_2d = std_estimate.view(-1, 1)
        expwts_2d = expwts.view(-1, seq_len_keys)

        for i in range(dense_mask_2d.shape[0]):
            true_std = torch.std(expwts_2d[i][dense_mask_2d[i] > 0])
            achieved_std = std_estimate_2d[i][0]
            # for this to be true repetitions should not happen. so set seq_lent ot large
            # and budget to small
            print(f"row: {i}, true_std: {true_std}, achieved_std: {achieved_std}")
            torch.testing.assert_close(true_std, achieved_std, rtol=0.1, atol=0.05)

    @pytest.mark.parametrize(
        "epsilon, delta", [(0.2, 0.2), (0.25, 0.25), (0.5, 0.5), (0.2, 0.1)]
    )
    def test_compute_adaptive_budget(self, masker, epsilon, delta):
        """Test adaptive budget computation."""
        std_estimate = torch.ones(1, 1)  # 1
        sampling_range = 100000
        data = torch.randn(1, sampling_range)
        static_denominator = 10000
        true_denominator = data.sum(dim=-1, keepdim=True) + static_denominator
        print(
            f"true_denominator: {true_denominator} = {data.sum(dim=-1, keepdim=True)} + {static_denominator}"
        )
        masker = AdaptiveSamplingMasker(
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.1,
                epsilon=epsilon,
                delta=delta,
                init_offset=0,
                local_offset=0,
            )
        )
        # i.e. assuming that data comes from a N(0,1) distribution
        budget = masker._compute_adaptive_budget(
            std_estimate, true_denominator, sampling_range
        )
        budget = int(budget.item())
        num_extreme_values = 0
        total_runs = 1000
        for i in range(total_runs):
            indices = torch.randperm(sampling_range)[:budget]
            data_sampled = data[:, indices]
            estimated_sum = (
                data_sampled.sum(dim=-1) * (sampling_range / budget)
            ).item() + static_denominator
            true_sum = true_denominator.item()
            extreme_value_present = (
                true_sum - estimated_sum
            ) > true_sum * masker.epsilon
            num_extreme_values += float(extreme_value_present)
        empirical_delta = num_extreme_values / total_runs
        print(
            f"budget: {budget}, empirical_delta: {empirical_delta} , masker.delta: {masker.delta}"
        )
        torch.testing.assert_close(empirical_delta, masker.delta, rtol=0.2, atol=0.05)

    def test_add_mask_early_exit(self, masker, sample_tensors):
        """Test early exit when previous mask is full."""
        keys, queries, values, attention_mask = sample_tensors

        # Create a full mask
        full_mask = Mask.create_full_mask(
            (2, 4, 8, 16), dtype=torch.float32, device=torch.device("cpu")
        )

        result = masker.add_mask(
            keys,
            queries,
            values,
            attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data={},
            previous_mask=full_mask,
        )

        assert result is full_mask

    def test_add_mask_with_zero_base_rate_sampling(self, sample_tensors):
        """Test that add_mask returns previous mask when base_rate_sampling is 0."""
        keys, queries, values, attention_mask = sample_tensors

        # Create a masker with base_rate_sampling=0
        config_zero = AdaptiveSamplingMaskerConfig(
            base_rate_sampling=0,
            epsilon=0.1,
            delta=0.05,
            init_offset=0,
            local_offset=0,
        )
        masker_zero = AdaptiveSamplingMasker(config_zero)

        # Create an empty mask
        empty_mask = Mask.create_empty_mask(
            (2, 4, 8, 16), dtype=torch.float32, device=torch.device("cpu")
        )

        # Call add_mask and verify it returns the previous mask unchanged
        result = masker_zero.add_mask(
            keys,
            queries,
            values,
            attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data={},
            previous_mask=empty_mask,
        )

        # Verify that the result is the same as the previous mask
        assert result is empty_mask
        assert result.is_empty

    def test_add_mask_with_zero_float_base_rate_sampling(self, sample_tensors):
        """Test that add_mask returns previous mask when base_rate_sampling is 0.0."""
        keys, queries, values, attention_mask = sample_tensors

        # Create a masker with base_rate_sampling=0.0
        config_zero = AdaptiveSamplingMaskerConfig(
            base_rate_sampling=0.0,
            epsilon=0.1,
            delta=0.05,
            init_offset=0,
            local_offset=0,
        )
        masker_zero = AdaptiveSamplingMasker(config_zero)

        # Create a random mask with ~30% sparsity
        torch.manual_seed(42)
        shape: tuple[int, int, int, int] = (2, 4, 8, 16)
        mask_tensor: torch.Tensor = (torch.rand(shape) > 0.3).to(torch.float32)
        # Add some random weights
        mask_tensor = mask_tensor * torch.rand(shape, dtype=torch.float32)
        previous_mask = Mask.create_mask_from_dense_mask(
            shape, mask_tensor, dtype=torch.float32
        )

        # Call add_mask and verify it returns the previous mask unchanged
        result = masker_zero.add_mask(
            keys,
            queries,
            values,
            attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data={},
            previous_mask=previous_mask,
        )

        # Verify that the result is the same as the previous mask
        assert result is previous_mask

    def test_add_mask_basic(self, masker, sample_tensors):
        """Test basic add_mask functionality."""
        keys, queries, values, attention_mask = sample_tensors

        # Create an empty mask
        empty_mask = Mask.create_empty_mask(
            (2, 4, 8, 16), dtype=torch.float32, device=torch.device("cpu")
        )

        result = masker.add_mask(
            keys,
            queries,
            values,
            attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data={},
            previous_mask=empty_mask,
        )

        assert isinstance(result, Mask)
        assert result.shape == (2, 4, 8, 16)
        assert not result.is_empty

    def test_create_from_config(self, config):
        """Test create_from_config factory method."""
        masker = AdaptiveSamplingMasker.create_from_config(config)
        assert isinstance(masker, AdaptiveSamplingMasker)
        assert masker.base_rate_sampling == 0.1

    def test_create_from_config_invalid(self):
        """Test create_from_config with invalid config type."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
            MaskerConfig,
        )

        invalid_config = MaskerConfig()

        with pytest.raises(ValueError, match="Invalid config type"):
            AdaptiveSamplingMasker.create_from_config(invalid_config)

    def test_device_consistency(self, masker, sample_tensors):
        """Test that all tensors are on the same device."""
        keys, queries, values, attention_mask = sample_tensors

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        keys = keys.to(device)
        queries = queries.to(device)
        values = values.to(device)
        attention_mask = attention_mask.to(device)

        empty_mask = Mask.create_empty_mask(
            (2, 4, 8, 16), dtype=torch.float32, device=torch.device("cpu")
        )

        result = masker.add_mask(
            keys,
            queries,
            values,
            attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data={},
            previous_mask=empty_mask,
        )

        # Check that result is on the same device
        assert result.get_dense_mask().device == keys.device

    def test_numerical_stability(self, masker, sample_tensors):
        """Test numerical stability with extreme values."""
        keys, queries, values, attention_mask = sample_tensors

        # Use very large values to test numerical stability
        keys = keys * 1000
        queries = queries * 1000

        empty_mask = Mask.create_empty_mask(
            (2, 4, 8, 16), dtype=torch.float32, device=torch.device("cpu")
        )

        result = masker.add_mask(
            keys,
            queries,
            values,
            attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data={},
            previous_mask=empty_mask,
        )

        # Should not have NaN or infinite values
        dense_mask = result.get_dense_mask()
        assert torch.all(torch.isfinite(dense_mask))
        assert not torch.any(torch.isnan(dense_mask))


@pytest.mark.unit
class TestImportanceSamplingEstimator:
    """The property the sampling modes exist for: an unbiased leftover estimate.

    Every assertion here is on the ESTIMATOR, not on shapes. Each one fails if
    the mask stores 1/pi instead of pi, if the proposal is left on the raw
    (unscaled) PQ axis, or if a repeated multinomial draw is counted twice.
    """

    SHAPE = (1, 2, 1, 512)
    START, END = 0, 512

    def _leftover_logits(self, spread: float, seed: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        return (
            torch.randn(self.SHAPE, generator=generator, dtype=torch.float32) * spread
        )

    def _masker(self, mode: str, temperature: float = 1.0):
        return AdaptiveSamplingMasker(
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.1,
                epsilon=0.1,
                delta=0.1,
                init_offset=0,
                local_offset=0,
                sampling_mode=mode,
                temperature=temperature,
            )
        )

    def _estimate(self, masker, logits, budget, trials):
        """Mean and sd of sum_{i in S} x_i / pi_i, as the attention path computes it."""
        x = torch.exp(logits - logits.max())
        budget_tensor = torch.full((*self.SHAPE[:-1], 1), budget, dtype=torch.long)
        estimates = []
        for trial in range(trials):
            torch.manual_seed(9000 + trial)
            mask = masker._create_importance_sampling_mask(
                logits, budget_tensor, self.SHAPE[-1], self.START, torch.float32
            )
            # apply_inv_mask is what turns the stored pi_i into the 1/pi_i weight
            estimates.append(float(mask.apply_inv_mask(x).sum()))
        stacked = torch.tensor(estimates)
        return float(stacked.mean()), float(stacked.std()), float(x.sum())

    @pytest.mark.parametrize("mode", ["gumbel", "multinomial"])
    def test_horvitz_thompson_estimator_is_unbiased(self, mode):
        """The whole point: E[sum x_i / pi_i] must equal sum x_i.

        With data = 1/pi the estimator collapses toward zero, and with the
        merge clamp on top it saturates at the truncated sum -- either way this
        is off by tens of percent, far outside the tolerance below.
        """
        logits = self._leftover_logits(spread=3.0, seed=11)
        mean, sd, truth = self._estimate(self._masker(mode), logits, 64, trials=120)
        relative_bias = (mean - truth) / truth
        standard_error = sd / (120**0.5) / truth
        assert abs(relative_bias) < max(0.05, 3 * standard_error), (
            f"{mode}: relative bias {relative_bias:+.3%} "
            f"(truth {truth:.3f}, mean {mean:.3f}, 1 s.e. {standard_error:.3%})"
        )

    @pytest.mark.parametrize("mode", ["gumbel", "multinomial"])
    def test_importance_sampling_beats_uniform_variance(self, mode):
        """Importance sampling is only worth its cost if it cuts the variance."""
        logits = self._leftover_logits(spread=3.0, seed=12)
        _, sd_importance, truth = self._estimate(
            self._masker(mode), logits, 64, trials=120
        )
        x = torch.exp(logits - logits.max())
        n = self.SHAPE[-1]
        uniform_estimates = []
        for trial in range(120):
            generator = torch.Generator().manual_seed(9000 + trial)
            drawn = torch.randint(0, n, (64,), generator=generator)
            uniform_estimates.append(
                float(x[..., torch.unique(drawn)].sum() / (64 / n))
            )
        sd_uniform = float(torch.tensor(uniform_estimates).std())
        assert sd_importance < 0.5 * sd_uniform, (
            f"{mode}: sd {sd_importance / truth:.2%} of truth vs uniform "
            f"{sd_uniform / truth:.2%}"
        )

    def test_mask_stores_inclusion_probability_not_its_reciprocal(self):
        """data must be pi_i in (0, 1]; apply_inv_mask is what makes it 1/pi_i.

        Storing 1/pi_i would (a) invert the correction and (b) be erased by
        merge_mask's clamp(m1 + m2, 0, 1), since every such value is >= 1.
        """
        logits = self._leftover_logits(spread=3.0, seed=13)
        budget_tensor = torch.full((*self.SHAPE[:-1], 1), 64, dtype=torch.long)
        for mode in ("gumbel", "multinomial"):
            torch.manual_seed(3)
            mask = self._masker(mode)._create_importance_sampling_mask(
                logits, budget_tensor, self.SHAPE[-1], self.START, torch.float32
            )
            dense = mask.get_dense_mask()
            selected = dense[dense > 0]
            assert selected.numel() > 0
            assert bool((selected > 0).all()) and bool((selected <= 1.0).all())
            assert bool((selected < 1.0).any()), f"{mode} produced a hard top-k"
            weights = mask.apply_inv_mask(torch.ones_like(dense))[dense > 0]
            assert bool((weights >= 1.0 - 1e-3).all()), "weights must up-weight"
            # and the clamp in merge_mask must leave them alone
            other = torch.zeros_like(dense)
            other[..., :4] = 1.0
            merged = Mask.create_mask_from_dense_mask(
                dense.shape, other, dtype=torch.float32
            ).merge_mask(mask, inplace=False)
            survived = merged.get_dense_mask()[..., 4:][dense[..., 4:] > 0]
            assert torch.allclose(survived, selected[-survived.numel() :], atol=1e-6)

    def test_multinomial_duplicate_draws_are_not_double_counted(self):
        """With replacement, one key can be drawn many times; pi_i applies once."""
        logits = torch.full(self.SHAPE, -20.0)
        logits[..., 7] = 20.0  # essentially all mass on a single key
        budget_tensor = torch.full((*self.SHAPE[:-1], 1), 32, dtype=torch.long)
        torch.manual_seed(5)
        mask = self._masker("multinomial")._create_importance_sampling_mask(
            logits, budget_tensor, self.SHAPE[-1], self.START, torch.float32
        )
        dense = mask.get_dense_mask()
        assert float(dense[..., 7].max()) <= 1.0
        assert float(dense[..., 7].min()) > 0.0

    def test_causally_masked_keys_are_never_sampled(self):
        """expwts == 0 (attention-masked) must be -inf in the proposal.

        log(clamp_min(1e-20)) floors those at a FINITE -46, which every
        isfinite() liveness test would happily sample.
        """
        masker = self._masker("gumbel")
        expwts = torch.rand(self.SHAPE) + 0.1
        expwts[..., 256:] = 0.0
        empty = Mask.create_empty_mask(
            self.SHAPE, dtype=torch.float32, device=torch.device("cpu")
        )
        leftover = masker._get_leftover_scores(
            expwts, empty, 0, self.SHAPE[-1], {}, {}, 1.0
        )
        assert bool(torch.isinf(leftover[..., 256:]).all())
        assert bool(torch.isfinite(leftover[..., :256]).all())

    def test_published_proposal_is_rescaled_onto_the_attention_logit_axis(self):
        """PQCache publishes raw q.k; the proposal must be scaling * q.k.

        Without this, temperature=1.0 means temperature=1/sqrt(head_dim) in
        attention space and the draw collapses onto the deterministic top-k.
        """
        masker = self._masker("gumbel")
        pq_scores = torch.randn(self.SHAPE) * 30.0
        meta = {"heavy_scores": {0: pq_scores}, "heavy_score_offset": {0: 0}}
        expwts = torch.rand(self.SHAPE) + 0.1
        empty = Mask.create_empty_mask(
            self.SHAPE, dtype=torch.float32, device=torch.device("cpu")
        )
        scaling = 0.0884
        leftover = masker._get_leftover_scores(
            expwts, empty, 0, self.SHAPE[-1], meta, {"layer_idx": 0}, scaling
        )
        assert torch.allclose(leftover, pq_scores * scaling, atol=1e-5)

    def test_stale_published_scores_from_another_step_are_not_reused(self):
        """The cache is keyed only by layer, so the leading dims must be checked."""
        masker = self._masker("gumbel")
        stale = torch.randn(1, 2, 8, self.SHAPE[-1])  # a prefill entry, q=8
        meta = {"heavy_scores": {0: stale}, "heavy_score_offset": {0: 0}}
        expwts = torch.rand(self.SHAPE) + 0.1  # decode, q=1
        empty = Mask.create_empty_mask(
            self.SHAPE, dtype=torch.float32, device=torch.device("cpu")
        )
        leftover = masker._get_leftover_scores(
            expwts, empty, 0, self.SHAPE[-1], meta, {"layer_idx": 0}, 1.0
        )
        assert leftover.shape == self.SHAPE
        assert torch.allclose(leftover, torch.log(expwts.clamp_min(1e-20)), atol=1e-5)

    def test_grouped_query_attention_shapes(self):
        """Llama-3.1-8B is 32 query heads over 8 KV heads; the mask is query-head."""
        num_q_heads, num_kv_heads, head_dim = 8, 2, 16
        seq_len_keys, seq_len_queries = 64, 1
        keys = torch.randn(1, num_kv_heads, seq_len_keys, head_dim)
        queries = torch.randn(1, num_q_heads, seq_len_queries, head_dim)
        values = torch.randn(1, num_kv_heads, seq_len_keys, head_dim)
        previous = torch.zeros(1, num_q_heads, seq_len_queries, seq_len_keys)
        previous[..., :8] = 1.0
        previous_mask = Mask.create_mask_from_dense_mask(
            previous.shape, previous, dtype=torch.float32
        )
        for mode in ("uniform", "gumbel", "multinomial"):
            result = self._masker(mode).add_mask(
                keys,
                queries,
                values,
                None,
                scaling=head_dim**-0.5,
                dropout=0.0,
                sparse_meta_data={},
                previous_mask=previous_mask,
                layer_idx=0,
            )
            assert result.get_dense_mask().shape == previous.shape

    def test_one_saturated_row_does_not_erase_other_rows_weights(self):
        """A whole-tensor -inf threshold would set pi = 1 everywhere."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations.utils.importance_sampling_utils import (  # noqa: E501
            gumbel_topk_with_inclusion,
        )

        n = 16
        scores = torch.randn(1, 1, 2, n) * 3.0
        budget = torch.tensor([[[[n], [4]]]]).reshape(1, 1, 2, 1)
        torch.manual_seed(2)
        _, inclusion, valid, _dense = gumbel_topk_with_inclusion(scores, budget, 1.0)
        saturated = inclusion[0, 0, 0][valid[0, 0, 0]]
        sampled = inclusion[0, 0, 1][valid[0, 0, 1]]
        assert bool((saturated == 1.0).all()), "a saturated row is a census"
        assert bool((sampled < 1.0).any()), "the other row must keep real weights"


@pytest.mark.unit
class TestSamplingModeConfig:
    """The flag itself."""

    def _config(self, **overrides):
        base = dict(
            base_rate_sampling=0.1,
            epsilon=0.1,
            delta=0.1,
            init_offset=0,
            local_offset=0,
        )
        base.update(overrides)
        return AdaptiveSamplingMaskerConfig(**base)

    def test_default_is_uniform(self):
        assert self._config().sampling_mode == "uniform"

    @pytest.mark.parametrize("mode", ["uniform", "gumbel", "multinomial"])
    def test_accepts_every_documented_mode(self, mode):
        assert self._config(sampling_mode=mode).sampling_mode == mode

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="sampling_mode must be one of"):
            self._config(sampling_mode="importance")

    def test_rejects_zero_temperature_for_importance_modes(self):
        with pytest.raises(ValueError, match="deterministic top-k"):
            self._config(sampling_mode="gumbel", temperature=0.0)

    def test_rejects_negative_temperature(self):
        with pytest.raises(ValueError, match="temperature must be >= 0"):
            self._config(temperature=-1.0)


@pytest.mark.unit
class TestImportanceSamplingIsHeavyMaskerAgnostic:
    """The sampling stage must not care which heavy masker preceded it.

    PQCache publishes its scores and they get reused as the proposal; every
    other heavy masker publishes nothing and the exact log(expwts) fallback
    takes over. Both must produce a valid mask, and neither may re-select a key
    the heavy stage already took.
    """

    HEAD_DIM = 32
    SHAPE = (1, 4, 1, 256)

    def _tensors(self):
        torch.manual_seed(17)
        batch, heads, queries, keys_len = self.SHAPE
        return (
            torch.randn(batch, heads, keys_len, self.HEAD_DIM),
            torch.randn(batch, heads, queries, self.HEAD_DIM),
            torch.randn(batch, heads, keys_len, self.HEAD_DIM),
        )

    def _heavy_masker(self, name):
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (  # noqa: E501
            OracleTopK,
            OracleTopKConfig,
            PQCache,
            PQCacheConfig,
        )

        if name == "oracle":
            return OracleTopK(OracleTopKConfig(heavy_size=16))
        return PQCache(
            PQCacheConfig(
                heavy_size=16,
                pq_group_factor=2,
                pq_bits=4,
                kmeans_iter=2,
                init_offset=8,
                metric="euclidean",
            )
        )

    @pytest.mark.parametrize("heavy", ["oracle", "pqcache"])
    @pytest.mark.parametrize("mode", ["gumbel", "multinomial"])
    def test_runs_behind_any_heavy_masker(self, heavy, mode):
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (  # noqa: E501
            LocalMasker,
            LocalMaskerConfig,
            SinkMasker,
            SinkMaskerConfig,
        )

        keys, queries, values = self._tensors()
        chain = [
            SinkMasker(SinkMaskerConfig(sink_size=8)),
            LocalMasker(LocalMaskerConfig(window_size=8)),
            self._heavy_masker(heavy),
            AdaptiveSamplingMasker(
                AdaptiveSamplingMaskerConfig(
                    base_rate_sampling=0.05,
                    epsilon=0.2,
                    delta=0.2,
                    init_offset=8,
                    local_offset=8,
                    sampling_mode=mode,
                    temperature=1.0,
                )
            ),
        ]
        meta: dict = {}
        mask = Mask.create_empty_mask(
            self.SHAPE, dtype=torch.float32, device=torch.device("cpu")
        )
        heavy_dense = None
        for masker in chain:
            mask = masker.add_mask(
                keys=keys,
                queries=queries,
                values=values,
                attention_mask=None,
                scaling=self.HEAD_DIM**-0.5,
                dropout=0.0,
                sparse_meta_data=meta,
                previous_mask=mask,
                layer_idx=0,
            )
            if masker is chain[2]:
                heavy_dense = mask.get_dense_mask().clone()

        dense = mask.get_dense_mask()
        assert dense.shape == self.SHAPE
        assert bool(torch.isfinite(dense).all())
        # the heavy stage's picks are untouched, and the sampler adds new keys
        # at a fractional inclusion probability rather than re-picking them
        assert bool((dense[heavy_dense > 0] == 1.0).all())
        fresh = (heavy_dense == 0) & (dense > 0)
        assert bool(fresh.any()), f"{heavy}/{mode} sampled nothing"
        assert bool((dense[fresh] > 0).all()) and bool((dense[fresh] <= 1.0).all())

        # PQCache offers its scores; OracleTopK offers none and takes the
        # exact log(expwts) fallback. Both must work.
        assert ("heavy_scores" in meta) == (heavy == "pqcache")

    def test_works_with_no_heavy_masker_at_all(self):
        """Sink + Local + AdaptiveSampling, nothing to reuse."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (  # noqa: E501
            SinkMasker,
            SinkMaskerConfig,
        )

        keys, queries, values = self._tensors()
        mask = Mask.create_empty_mask(
            self.SHAPE, dtype=torch.float32, device=torch.device("cpu")
        )
        meta: dict = {}
        for masker in (
            SinkMasker(SinkMaskerConfig(sink_size=8)),
            AdaptiveSamplingMasker(
                AdaptiveSamplingMaskerConfig(
                    base_rate_sampling=0.05,
                    epsilon=0.2,
                    delta=0.2,
                    init_offset=8,
                    local_offset=8,
                    sampling_mode="multinomial",
                )
            ),
        ):
            mask = masker.add_mask(
                keys=keys,
                queries=queries,
                values=values,
                attention_mask=None,
                scaling=self.HEAD_DIM**-0.5,
                dropout=0.0,
                sparse_meta_data=meta,
                previous_mask=mask,
                layer_idx=0,
            )
        dense = mask.get_dense_mask()
        assert "heavy_scores" not in meta
        assert bool((dense > 0).any()) and bool((dense <= 1.0).all())
