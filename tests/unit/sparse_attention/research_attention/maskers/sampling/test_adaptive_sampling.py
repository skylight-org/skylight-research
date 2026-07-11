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
class TestAdaptiveSamplingSoftcap:
    """Test softcap handling in AdaptiveSamplingMasker."""

    @pytest.fixture
    def masker(self):
        """Create an AdaptiveSamplingMasker instance for softcap testing."""
        config = AdaptiveSamplingMaskerConfig(
            base_rate_sampling=0.25,
            epsilon=0.1,
            delta=0.1,
            init_offset=0,
            local_offset=0,
        )
        return AdaptiveSamplingMasker(config)

    def test_compute_exp_scores_applies_softcap_exactly(self, masker):
        """Softcapped exp scores must equal exp(cap*tanh(s/cap) - rowmax)."""
        torch.manual_seed(0)
        queries = torch.randn(1, 1, 2, 4)
        keys = torch.randn(1, 1, 16, 4)
        scaling = 0.3
        cap = 2.0

        raw_scores = torch.matmul(queries, keys.transpose(-2, -1)) * scaling

        # softcap set: cap * tanh(s / cap), then row-wise max subtraction
        capped_scores = cap * torch.tanh(raw_scores / cap)
        expected_capped = torch.exp(
            capped_scores - capped_scores.max(dim=-1, keepdim=True)[0]
        )
        actual_capped = masker._compute_exp_attention_scores(
            queries, keys, scaling, None, softcap=cap
        )
        torch.testing.assert_close(actual_capped, expected_capped, atol=1e-6, rtol=0)

        # softcap=None: no tanh, plain exp(s - rowmax)
        expected_uncapped = torch.exp(
            raw_scores - raw_scores.max(dim=-1, keepdim=True)[0]
        )
        actual_uncapped = masker._compute_exp_attention_scores(
            queries, keys, scaling, None, softcap=None
        )
        torch.testing.assert_close(
            actual_uncapped, expected_uncapped, atol=1e-6, rtol=0
        )

    def test_softcap_changes_adaptive_budget(self, masker):
        """Softcap flattens a bimodal score gap, shrinking the adaptive budget.

        Uncapped, the exp scores are {1, e^-8}: high relative std and a small
        denominator concentrated on half the keys drive the adaptive budget to
        its maximum. With softcap=0.5 the scores become {1, e^-0.5}: the
        distribution is much flatter (smaller std, larger denominator), so the
        budget formula (delta_ppf * std * range / (eps * denom))^2 yields a
        smaller budget and thus a lower mask density.
        """
        batch_size, num_heads, seq_len_queries, seq_len_keys, head_dim = 1, 1, 1, 64, 2

        queries = torch.tensor([[1.0, 0.0]]).view(
            batch_size, num_heads, seq_len_queries, head_dim
        )
        keys = torch.zeros(batch_size, num_heads, seq_len_keys, head_dim)
        keys[:, :, : seq_len_keys // 2, 0] = 8.0  # bimodal: half at 8.0, half at 0.0
        values = torch.zeros(batch_size, num_heads, seq_len_keys, head_dim)
        scaling = 1.0
        attention_mask = None

        def make_empty_mask():
            return Mask.create_empty_mask(
                (batch_size, num_heads, seq_len_queries, seq_len_keys),
                dtype=torch.float32,
                device=torch.device("cpu"),
            )

        torch.manual_seed(0)
        mask_nocap = masker.add_mask(
            keys,
            queries,
            values,
            attention_mask,
            scaling=scaling,
            dropout=0.0,
            sparse_meta_data={},
            previous_mask=make_empty_mask(),
        )

        torch.manual_seed(0)
        mask_cap = masker.add_mask(
            keys,
            queries,
            values,
            attention_mask,
            scaling=scaling,
            dropout=0.0,
            sparse_meta_data={},
            previous_mask=make_empty_mask(),
            softcap=0.5,
        )

        # Flatter capped distribution => smaller sampled budget => lower density.
        assert mask_nocap.get_density() > mask_cap.get_density()

        # softcap=None must reproduce the no-kwarg result exactly under same seed.
        torch.manual_seed(0)
        mask_none = masker.add_mask(
            keys,
            queries,
            values,
            attention_mask,
            scaling=scaling,
            dropout=0.0,
            sparse_meta_data={},
            previous_mask=make_empty_mask(),
            softcap=None,
        )
        assert torch.equal(mask_none.get_dense_mask(), mask_nocap.get_dense_mask())
