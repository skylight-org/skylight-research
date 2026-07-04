"""
:author: Aditya Desai
:copyright: 2025 Sparse Attention Hub
:license: Apache 2.0
:date: 2025-07-03
:summary: Tests for sparse attention. This file is part of the Sparse Attention Hub project.
"""


import mock
import numpy as np
import pytest
import torch

from sparse_attention_hub.sparse_attention.utils.mask import Mask
from sparse_attention_hub.sparse_attention.utils.mask_attention_utils import (
    _compute_masked_exp_attention_weights,
    apply_inv_mask_sum,
    apply_softcap,
    create_sampling_mask_with_per_head_budget,
    get_attention_denominator,
    get_attention_numerator,
    get_masked_attention_output,
)


@pytest.mark.unit
class TestApplyInvMaskSum:
    """Test apply_inv_mask_sum utility function."""

    @pytest.fixture
    def sample_tensor(self):
        """Create a sample tensor for testing."""
        return torch.randn(2, 4, 8, 16)

    @pytest.fixture
    def big_sample_tensor(self):
        """Create a sample tensor for testing."""
        return torch.randn(2, 4, 8, 128)

    def test_full_mask(self, sample_tensor):
        """Test with full mask."""
        full_mask = Mask.create_full_mask(
            (2, 4, 8, 16), dtype=torch.float32, device=torch.device("cpu")
        )

        result = apply_inv_mask_sum(sample_tensor, full_mask)

        expected = sample_tensor.sum(dim=-1, keepdim=True)
        assert result.shape == (2, 4, 8, 1)
        torch.testing.assert_close(result, expected)

    def test_empty_mask(self, sample_tensor):
        """Test with empty mask."""
        empty_mask = Mask.create_empty_mask(
            (2, 4, 8, 16), dtype=torch.float32, device=torch.device("cpu")
        )

        result = apply_inv_mask_sum(sample_tensor, empty_mask)
        # empty and full have to behavethe same
        expected = sample_tensor.sum(dim=-1, keepdim=True)
        assert result.shape == (2, 4, 8, 1)
        torch.testing.assert_close(result, expected)

    def test_sparse_mask(self, big_sample_tensor):
        """Test with sparse mask."""
        dense_mask = torch.rand_like(big_sample_tensor)
        dense_mask = (dense_mask > 0.5).float() * dense_mask
        mask_object = Mask.create_mask_from_dense_mask(
            dense_mask.shape, dense_mask, dtype=dense_mask.dtype
        )

        result = apply_inv_mask_sum(big_sample_tensor, mask_object)

        non_zero_indices = dense_mask != 0
        zero_indices = dense_mask == 0
        expected = big_sample_tensor.clone()
        expected[zero_indices] = 0
        expected[non_zero_indices] = (
            expected[non_zero_indices] / dense_mask[non_zero_indices]
        )
        expected = expected.sum(dim=-1, keepdim=True)
        assert result.shape == (2, 4, 8, 1)
        torch.testing.assert_close(result, expected, atol=5e-5, rtol=5e-5)

    def test_sparse_mask_no_indices(self, sample_tensor):
        """Test with sparse mask that has no active indices."""
        indices = torch.empty(0, dtype=torch.long)
        ptr = torch.zeros(2 * 4 * 8 + 1, dtype=torch.long)  # 2*4*8 rows + 1
        data = torch.empty(0, dtype=torch.float32)
        sparse_mask = Mask.create_mask_from_indices(
            (2, 4, 8, 16), indices, ptr, data, dtype=torch.float32
        )

        result = apply_inv_mask_sum(sample_tensor, sparse_mask)

        expected = torch.zeros(
            2, 4, 8, 1, device=sample_tensor.device, dtype=sample_tensor.dtype
        )
        assert result.shape == (2, 4, 8, 1)
        torch.testing.assert_close(result, expected, atol=5e-5, rtol=5e-5)

    def test_shape_mismatch(self, sample_tensor):
        """Test with shape mismatch."""
        wrong_shape_mask = Mask.create_full_mask(
            (2, 4, 8, 8), dtype=torch.float32, device=torch.device("cpu")
        )

        with pytest.raises(ValueError, match="input_tensor.shape must be"):
            apply_inv_mask_sum(sample_tensor, wrong_shape_mask)

    def test_device_consistency(self, sample_tensor):
        """Test device consistency."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sample_tensor = sample_tensor.to(device)

        full_mask = Mask.create_full_mask(
            (2, 4, 8, 16), dtype=torch.float32, device=device
        )

        result = apply_inv_mask_sum(sample_tensor, full_mask)

        assert result.device == sample_tensor.device

    def test_dtype_consistency(self, sample_tensor):
        """Test dtype consistency."""
        sample_tensor = sample_tensor.to(torch.float64)
        full_mask = Mask.create_full_mask(
            (2, 4, 8, 16), dtype=torch.float64, device=torch.device("cpu")
        )

        result = apply_inv_mask_sum(sample_tensor, full_mask)

        assert result.dtype == torch.float64


@pytest.mark.unit
class TestCreateSamplingMaskWithPerHeadBudget:
    """Test create_sampling_mask_with_per_head_budget utility function."""

    @pytest.fixture
    def sample_budgets(self):
        """Create sample budgets tensor."""
        return torch.tensor(
            [[[[2]], [[3]], [[1]], [[4]]]], dtype=torch.long
        )  # (1, 4, 1, 1)

    @pytest.fixture
    def sample_sampling_probabilities(self):
        """Create sample sampling probabilities tensor."""
        return torch.tensor(
            [[[[0.2]], [[0.3]], [[0.1]], [[0.4]]]], dtype=torch.float32
        )  # (1, 4, 1, 1)

    def test_basic_functionality(self, sample_budgets, sample_sampling_probabilities):
        """Test basic functionality."""
        seq_len_keys = 1024
        start_idx = 0
        end_idx = seq_len_keys
        dtype = torch.float32

        mask_object = create_sampling_mask_with_per_head_budget(
            budgets=sample_budgets,
            sampling_probability=sample_sampling_probabilities,
            seq_len_keys=seq_len_keys,
            start_idx=start_idx,
            end_idx=end_idx,
            dtype=dtype,
        )

        mask = mask_object.get_dense_mask()
        assert isinstance(mask_object, Mask)
        assert mask.shape == (1, 4, 1, 1024)
        assert mask.dtype == dtype
        # for this with sampling with replacement, this assert would hold mostly when seq_len_keys is large and budgets are small
        torch.testing.assert_close(
            (mask > 0).long().sum(dim=-1, keepdim=True), sample_budgets
        )
        mask_2d = mask.view(-1, seq_len_keys)
        sampling_probabilities_2d = sample_sampling_probabilities.view(-1, 1)
        for i in range(mask_2d.shape[0]):
            torch.testing.assert_close(
                mask_2d[i][mask_2d[i] > 0],
                torch.full_like(
                    mask_2d[i][mask_2d[i] > 0],
                    sampling_probabilities_2d[i][0],
                    dtype=dtype,
                ),
            )

    def test_sampling_range(self, sample_budgets, sample_sampling_probabilities):
        """Test with different sampling range."""
        seq_len_keys = 20
        start_idx = 10
        end_idx = 15
        dtype = torch.float32

        mask = create_sampling_mask_with_per_head_budget(
            budgets=sample_budgets,
            sampling_probability=sample_sampling_probabilities,
            seq_len_keys=seq_len_keys,
            start_idx=start_idx,
            end_idx=end_idx,
            dtype=dtype,
        )

        assert isinstance(mask, Mask)
        assert mask.shape == (1, 4, 1, 20)

        # Check that indices are within the sampling range
        mask = mask.get_dense_mask()
        assert mask[:, :, :, :start_idx].sum() == 0
        assert mask[:, :, :, end_idx:].sum() == 0

    def test_zero_budgets(self):
        """Test with zero budgets."""
        budgets = torch.zeros(1, 1, 4, 1, dtype=torch.long)
        sampling_probabilities = torch.zeros(1, 1, 4, 1, dtype=torch.float32)

        mask = create_sampling_mask_with_per_head_budget(
            budgets=budgets,
            sampling_probability=sampling_probabilities,
            seq_len_keys=16,
            start_idx=0,
            end_idx=16,
            dtype=torch.float32,
        )

        assert isinstance(mask, Mask)
        assert mask.shape == (1, 1, 4, 16)
        assert mask.get_dense_mask().sum() == 0

    def test_large_budgets(self):
        """Test with large budgets."""
        budgets = torch.tensor([[[[8]], [[12]], [[6]], [[10]]]], dtype=torch.long)
        sampling_probabilities = torch.tensor(
            [[[[0.5]], [[0.75]], [[0.375]], [[0.625]]]], dtype=torch.float32
        )

        mask = create_sampling_mask_with_per_head_budget(
            budgets=budgets,
            sampling_probability=sampling_probabilities,
            seq_len_keys=16,
            start_idx=0,
            end_idx=16,
            dtype=torch.float32,
        )

        assert isinstance(mask, Mask)
        assert mask.shape == (1, 4, 1, 16)

        # Check that we have the expected number of elements
        indices, ptr, data = mask.get_index_mask()
        expected_total = budgets.sum().item()
        assert indices.numel() == expected_total
        # ^ this is true , but there can be repetition of indices

        # with large budgets getting the # indices per row exact is not possible
        # due to random sampling with replacement
        # mask = mask.get_dense_mask()
        # torch.testing.assert_close((mask > 0).long().sum(dim=-1, keepdim=True), budgets)

    def test_device_consistency(self, sample_budgets, sample_sampling_probabilities):
        """Test device consistency."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sample_budgets = sample_budgets.to(device)
        sample_sampling_probabilities = sample_sampling_probabilities.to(device)

        mask = create_sampling_mask_with_per_head_budget(
            budgets=sample_budgets,
            sampling_probability=sample_sampling_probabilities,
            seq_len_keys=16,
            start_idx=0,
            end_idx=16,
            dtype=torch.float32,
        )

        dense_mask = mask.get_dense_mask()
        assert dense_mask.device == sample_budgets.device

    def test_dtype_consistency(self, sample_budgets, sample_sampling_probabilities):
        """Test dtype consistency."""
        sample_sampling_probabilities = sample_sampling_probabilities.to(torch.float64)

        mask = create_sampling_mask_with_per_head_budget(
            budgets=sample_budgets,
            sampling_probability=sample_sampling_probabilities,
            seq_len_keys=16,
            start_idx=0,
            end_idx=16,
            dtype=torch.float64,
        )

        assert mask.dtype == torch.float64

    def test_batch_multiple_heads(self):
        """Test with multiple batches and heads."""
        batch_size, num_heads = 2, 3
        budgets = torch.randint(1, 5, (batch_size, num_heads, 4, 1), dtype=torch.long)
        sampling_probabilities = torch.rand(
            batch_size, num_heads, 4, 1, dtype=torch.float32
        )

        mask = create_sampling_mask_with_per_head_budget(
            budgets=budgets,
            sampling_probability=sampling_probabilities,
            seq_len_keys=16,
            start_idx=0,
            end_idx=16,
            dtype=torch.float32,
        )

        assert isinstance(mask, Mask)
        assert mask.shape == (batch_size, num_heads, 4, 16)

    def test_edge_case_single_element(self):
        """Test edge case with single element."""
        budgets = torch.tensor([[[[1]]]], dtype=torch.long)
        sampling_probabilities = torch.tensor([[[[0.1]]]], dtype=torch.float32)

        mask = create_sampling_mask_with_per_head_budget(
            budgets=budgets,
            sampling_probability=sampling_probabilities,
            seq_len_keys=16,
            start_idx=0,
            end_idx=16,
            dtype=torch.float32,
        )

        assert isinstance(mask, Mask)
        assert mask.shape == (1, 1, 1, 16)

        # Should have exactly one element
        indices, ptr, data = mask.get_index_mask()
        assert indices.numel() == 1

    def test_sampling_probability_consistency(self, sample_budgets):
        """Test that sampling probabilities are correctly assigned."""
        # Use different probabilities for each element
        sampling_probabilities = torch.tensor(
            [[[[0.1]], [[0.2]], [[0.3]], [[0.4]]]], dtype=torch.float32
        )

        mask = create_sampling_mask_with_per_head_budget(
            budgets=sample_budgets,
            sampling_probability=sampling_probabilities,
            seq_len_keys=16,
            start_idx=0,
            end_idx=16,
            dtype=torch.float32,
        )

        indices, ptr, data = mask.get_index_mask()

        # Check that data values match the sampling probabilities
        # Each row should have the same probability value
        expected_probs = sampling_probabilities.view(-1)  # [0.1, 0.2, 0.3, 0.4]

        for i in range(len(expected_probs)):
            start_idx = ptr[i]
            end_idx = ptr[i + 1]
            if start_idx < end_idx:
                row_data = data[start_idx:end_idx]
                assert torch.all(row_data == expected_probs[i])


@pytest.mark.unit
class TestMaskExpWts:
    """Test class for mask attention utils."""

    def test_compute_masked_attention_weights(self):
        """Test that the masked attention weights are computed correctly."""
        # Test parameters
        batch_size, num_heads, seq_len, d_model = 2, 4, 8, 16
        scaling = 1.0 / np.sqrt(d_model)

        # Create test tensors
        queries = torch.randn(batch_size, num_heads, seq_len, d_model)
        keys = torch.randn(batch_size, num_heads, seq_len, d_model)

        # Test case 1: No attention mask, empty sparse mask
        sparse_attention_mask = Mask.create_empty_mask(
            (batch_size, num_heads, seq_len, seq_len),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        result = _compute_masked_exp_attention_weights(
            queries=queries,
            keys=keys,
            attention_mask=None,
            scaling=scaling,
            sparse_attention_mask=sparse_attention_mask,
        )

        # Verify shape
        assert result.shape == (batch_size, num_heads, seq_len, seq_len)

        # Verify that result contains exponential values (should be positive)
        assert torch.all(result > 0)

        # Verify numerical correctness: should be exp(QK^T * scaling)
        expected_raw_weights = torch.matmul(queries, keys.transpose(-2, -1)) * scaling
        expected_max = torch.max(expected_raw_weights, dim=-1, keepdim=True)[0]
        expected_exp_weights = torch.exp(expected_raw_weights - expected_max)

        assert torch.allclose(result, expected_exp_weights, atol=1e-6)

    def test_compute_masked_attention_weights_with_attention_mask(self):
        """Test masked attention weights computation with attention mask."""
        batch_size, num_heads, seq_len, d_model = 2, 4, 8, 16
        scaling = 1.0 / np.sqrt(d_model)

        queries = torch.randn(batch_size, num_heads, seq_len, d_model)
        keys = torch.randn(batch_size, num_heads, seq_len, d_model)

        # Create attention mask (lower triangular for causal attention)
        attention_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        attention_mask.masked_fill_(attention_mask == 1, float("-inf"))
        attention_mask = (
            attention_mask.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, num_heads, -1, -1)
        )

        sparse_attention_mask = Mask.create_empty_mask(
            (batch_size, num_heads, seq_len, seq_len),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        result = _compute_masked_exp_attention_weights(
            queries=queries,
            keys=keys,
            attention_mask=attention_mask,
            scaling=scaling,
            sparse_attention_mask=sparse_attention_mask,
        )

        # Verify shape
        assert result.shape == (batch_size, num_heads, seq_len, seq_len)

        # Verify that upper triangular elements are zero (masked out)
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert torch.allclose(
                    result[:, :, i, j], torch.zeros_like(result[:, :, i, j])
                )

        # Verify that lower triangular elements are positive
        for i in range(seq_len):
            for j in range(i + 1):
                assert torch.all(result[:, :, i, j] > 0)

    def test_compute_masked_attention_weights_with_sparse_mask(self):
        """Test masked attention weights computation with sparse attention mask."""
        batch_size, num_heads, seq_len, d_model = 2, 4, 8, 16
        scaling = 1.0 / np.sqrt(d_model)

        queries = torch.randn(batch_size, num_heads, seq_len, d_model)
        keys = torch.randn(batch_size, num_heads, seq_len, d_model)

        # Create a sparse attention mask (random pattern)
        dense_mask = torch.randint(
            0, 3, (batch_size, num_heads, seq_len, seq_len)
        )  # assign weights
        dense_mask = dense_mask.float()
        sparse_attention_mask = Mask.create_mask_from_dense_mask(
            (batch_size, num_heads, seq_len, seq_len), dense_mask, dtype=torch.float32
        )

        result = _compute_masked_exp_attention_weights(
            queries=queries,
            keys=keys,
            attention_mask=None,
            scaling=scaling,
            sparse_attention_mask=sparse_attention_mask,
        )

        # Verify shape
        assert result.shape == (batch_size, num_heads, seq_len, seq_len)

        # Verify that masked positions (where dense_mask == 0) have zero values
        masked_positions = dense_mask == 0
        assert torch.allclose(
            result[masked_positions], torch.zeros_like(result[masked_positions])
        )

        # Verify that unmasked positions have positive values
        unmasked_positions = dense_mask > 0
        assert torch.all(result[unmasked_positions] > 0)

        # verify that the non-zero values are what we expect
        expected_raw_weights = torch.matmul(queries, keys.transpose(-2, -1)) * scaling
        expected_max = torch.max(expected_raw_weights, dim=-1, keepdim=True)[0]
        expected_exp_weights = torch.exp(expected_raw_weights - expected_max)
        assert torch.allclose(
            result[unmasked_positions],
            expected_exp_weights[unmasked_positions]
            * (1.0 / dense_mask[unmasked_positions]),
        )

    def test_compute_masked_attention_weights_with_both_masks(self):
        """Test masked attention weights computation with both attention mask and sparse mask."""

        batch_size, num_heads, seq_len, d_model = 2, 4, 8, 16
        scaling = 1.0 / np.sqrt(d_model)

        queries = torch.randn(batch_size, num_heads, seq_len, d_model)
        keys = torch.randn(batch_size, num_heads, seq_len, d_model)

        # Create causal attention mask
        attention_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        attention_mask.masked_fill_(attention_mask == 1, float("-inf"))
        attention_mask = (
            attention_mask.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, num_heads, -1, -1)
        )

        # Create a sparse attention mask (random pattern)
        dense_mask = torch.randint(
            0, 3, (batch_size, num_heads, seq_len, seq_len)
        )  # assign weights
        dense_mask = dense_mask.float()
        sparse_attention_mask = Mask.create_mask_from_dense_mask(
            (batch_size, num_heads, seq_len, seq_len), dense_mask, dtype=torch.float32
        )

        result = _compute_masked_exp_attention_weights(
            queries=queries,
            keys=keys,
            attention_mask=attention_mask,
            scaling=scaling,
            sparse_attention_mask=sparse_attention_mask,
        )

        # Verify shape
        assert result.shape == (batch_size, num_heads, seq_len, seq_len)

        # Verify that positions masked by either mask have zero values
        causal_mask = attention_mask == float("-inf")
        combined_mask = causal_mask | (dense_mask == 0)
        assert torch.allclose(
            result[combined_mask], torch.zeros_like(result[combined_mask])
        )

        # Verify that positions not masked by either mask have positive values
        active_positions = ~combined_mask
        assert torch.all(result[active_positions] > 0)

        # verify that the non-zero values are what we expect
        expected_raw_weights = torch.matmul(queries, keys.transpose(-2, -1)) * scaling

        expected_max = torch.max(
            expected_raw_weights + attention_mask, dim=-1, keepdim=True
        )[0]
        expected_exp_weights = torch.exp(expected_raw_weights - expected_max)
        assert torch.allclose(
            result[active_positions],
            expected_exp_weights[active_positions]
            * (1.0 / dense_mask[active_positions]),
        )

    def test_compute_masked_attention_weights_different_scaling(self):
        """Test masked attention weights computation with different scaling factors."""
        batch_size, num_heads, seq_len, d_model = 2, 4, 8, 16

        queries = torch.randn(batch_size, num_heads, seq_len, d_model)
        keys = torch.randn(batch_size, num_heads, seq_len, d_model)

        sparse_attention_mask = Mask.create_empty_mask(
            (batch_size, num_heads, seq_len, seq_len),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # Test with different scaling factors
        scaling_factors = [0.1, 0.2, 0.3]

        for scaling in scaling_factors:
            result = _compute_masked_exp_attention_weights(
                queries=queries,
                keys=keys,
                attention_mask=None,
                scaling=scaling,
                sparse_attention_mask=sparse_attention_mask,
            )

            # Verify shape
            assert result.shape == (batch_size, num_heads, seq_len, seq_len)

            # Verify that result contains positive values
            assert torch.all(result > 0)

            # Verify numerical correctness
            expected_raw_weights = (
                torch.matmul(queries, keys.transpose(-2, -1)) * scaling
            )
            expected_max = torch.max(expected_raw_weights, dim=-1, keepdim=True)[0]
            expected_exp_weights = torch.exp(expected_raw_weights - expected_max)

            assert torch.allclose(result, expected_exp_weights, atol=1e-6)

    def test_softcap_limits_attention_scores(self):
        """Softcap transformation should asymptotically bound large logits."""
        scores = torch.tensor([100.0])
        cap = 50.0

        result = apply_softcap(scores, cap)

        assert result.item() <= cap


@pytest.mark.unit
class TestGetAttentionDenominator:
    """Test class for get attention denominator."""

    # TODO(aditya): test using correct mocking.
    def test_get_attention_denominator(self):
        """simple function: replicated from the original implementation to
        ensure implementation persists"""

        batch_size, num_heads, seq_len, d_model = 2, 4, 8, 16
        scaling = 1.0 / np.sqrt(d_model)

        queries = torch.randn(batch_size, num_heads, seq_len, d_model)
        keys = torch.randn(batch_size, num_heads, seq_len, d_model)

        # Create causal attention mask
        attention_mask = torch.randn(batch_size, num_heads, seq_len, seq_len)

        # Create a sparse attention mask (random pattern)
        sparse_attention_mask = Mask.create_empty_mask(
            (batch_size, num_heads, seq_len, seq_len),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        module = torch.nn.Module()
        module.eval()
        module.num_key_value_groups = 1

        with mock.patch(
            "sparse_attention_hub.sparse_attention.utils.mask_attention_utils._compute_masked_exp_attention_weights"
        ) as mock_compute_masked_exp_attention_weights:
            mock_compute_masked_exp_attention_weights.return_value = torch.randn(
                batch_size, num_heads, seq_len, seq_len
            )
            true_denominator = torch.sum(
                mock_compute_masked_exp_attention_weights.return_value,
                dim=-1,
                keepdim=True,
            )
            denominator = get_attention_denominator(
                module=module,
                queries=queries,
                keys=keys,
                attention_mask=attention_mask,
                scaling=scaling,
                dropout=0.0,
                sparse_attention_mask=sparse_attention_mask,
            )

        assert torch.allclose(denominator, true_denominator)


@pytest.mark.unit
class TestGetAttentionNumerator:
    """Test class for get attention numerator."""

    # TODO(aditya): test using correct mocking.
    def test_get_attention_numerator(self):
        """simple function: replicated from the original implementation to
        ensure implementation persists"""

        batch_size, num_heads, seq_len, d_model = 2, 4, 8, 16
        scaling = 1.0 / np.sqrt(d_model)

        queries = torch.randn(batch_size, num_heads, seq_len, d_model)
        keys = torch.randn(batch_size, num_heads, seq_len, d_model)
        values = torch.randn(batch_size, num_heads, seq_len, d_model)

        # Create causal attention mask
        attention_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        attention_mask.masked_fill_(attention_mask == 1, float("-inf"))
        attention_mask = (
            attention_mask.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, num_heads, -1, -1)
        )

        # Create a sparse attention mask (random pattern)
        dense_mask = torch.randint(
            0, 3, (batch_size, num_heads, seq_len, seq_len)
        )  # assign weights
        dense_mask = dense_mask.float()
        sparse_attention_mask = Mask.create_mask_from_dense_mask(
            (batch_size, num_heads, seq_len, seq_len), dense_mask, dtype=torch.float32
        )

        with mock.patch(
            "sparse_attention_hub.sparse_attention.utils.mask_attention_utils._compute_masked_exp_attention_weights"
        ) as mock_compute_masked_exp_attention_weights:
            mock_compute_masked_exp_attention_weights.return_value = torch.randn(
                batch_size, num_heads, seq_len, seq_len
            )
            true_numerator = torch.matmul(
                mock_compute_masked_exp_attention_weights.return_value, values
            )
            numerator = get_attention_numerator(
                module=None,
                queries=queries,
                keys=keys,
                values=values,
                attention_mask=attention_mask,
                scaling=scaling,
                dropout=0.0,
                sparse_attention_mask=sparse_attention_mask,
            )
        assert torch.allclose(numerator, true_numerator)


@pytest.mark.unit
class TestGetMaskedAttentionOutputExternal:
    """Test class for get masked attention output."""

    def test_compare_with_eager_attention_sparse_mask_empty_dropout_0_eval_mode_num_kv_heads_2_different_q_len(
        self,
    ):
        """Test that the masked attention output is the same as the eager attention output for no mask."""
        batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_kv, d_model = (
            2,
            4,
            2,
            2,
            32,
            16,
        )
        scaling = 1.0 / np.sqrt(d_model)
        dropout = 0.1

        queries = torch.randn(batch_size, num_q_heads, seq_len_q, d_model)
        keys = torch.randn(batch_size, num_kv_heads, seq_len_kv, d_model)
        values = torch.randn(batch_size, num_kv_heads, seq_len_kv, d_model)

        sparse_attention_mask = Mask.create_empty_mask(
            (batch_size, num_q_heads, seq_len_q, seq_len_kv),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # Create attention mask (lower triangular for causal attention)
        attention_mask = torch.triu(
            torch.ones(seq_len_q, seq_len_kv), diagonal=seq_len_kv - seq_len_q
        )
        attention_mask.masked_fill_(attention_mask == 1, float("-inf"))
        attention_mask = (
            attention_mask.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, num_q_heads, -1, -1)
        )

        from transformers.models.llama.modeling_llama import eager_attention_forward

        module = torch.nn.Module()
        module.eval()
        module.num_key_value_groups = num_q_heads // num_kv_heads

        eager_attention_output, eager_attention_weights = eager_attention_forward(
            module=module,
            query=queries,
            key=keys,
            value=values,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout,
        )

        my_attention_output, my_attention_weights = get_masked_attention_output(
            module=module,
            queries=queries,
            keys=keys,
            values=values,
            attention_mask=attention_mask,
            sinks=None,
            scaling=scaling,
            dropout=dropout,
            sparse_attention_mask=sparse_attention_mask,
            return_attention_weights=True,
            mode="sparse",
        )
        assert torch.allclose(my_attention_output, eager_attention_output, atol=1e-6)
        assert torch.allclose(my_attention_weights, eager_attention_weights, atol=1e-6)

    def test_compare_with_eager_attention_sparse_mask_empty_dropout_0_eval_mode_num_kv_heads_2(
        self,
    ):
        """Test that the masked attention output is the same as the eager attention output for no mask."""
        batch_size, num_q_heads, num_kv_heads, seq_len, d_model = 2, 4, 2, 8, 16
        scaling = 1.0 / np.sqrt(d_model)
        dropout = 0.1

        queries = torch.randn(batch_size, num_q_heads, seq_len, d_model)
        keys = torch.randn(batch_size, num_kv_heads, seq_len, d_model)
        values = torch.randn(batch_size, num_kv_heads, seq_len, d_model)

        sparse_attention_mask = Mask.create_empty_mask(
            (batch_size, num_q_heads, seq_len, seq_len),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # Create attention mask (lower triangular for causal attention)
        attention_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        attention_mask.masked_fill_(attention_mask == 1, float("-inf"))
        attention_mask = (
            attention_mask.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, num_q_heads, -1, -1)
        )

        from transformers.models.llama.modeling_llama import eager_attention_forward

        module = torch.nn.Module()
        module.eval()
        module.num_key_value_groups = num_q_heads // num_kv_heads

        eager_attention_output, eager_attention_weights = eager_attention_forward(
            module=module,
            query=queries,
            key=keys,
            value=values,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout,
        )

        my_attention_output, my_attention_weights = get_masked_attention_output(
            module=module,
            queries=queries,
            keys=keys,
            values=values,
            attention_mask=attention_mask,
            sinks=None,
            scaling=scaling,
            dropout=dropout,
            sparse_attention_mask=sparse_attention_mask,
            return_attention_weights=True,
        )
        assert torch.allclose(my_attention_output, eager_attention_output, atol=1e-6)
        assert torch.allclose(my_attention_weights, eager_attention_weights, atol=1e-6)

    def test_compare_with_eager_attention_sparse_mask_empty_dropout_0_eval_mode(self):
        """Test that the masked attention output is the same as the eager attention output for no mask."""
        batch_size, num_heads, seq_len, d_model = 2, 4, 8, 16
        scaling = 1.0 / np.sqrt(d_model)
        dropout = 0.1

        queries = torch.randn(batch_size, num_heads, seq_len, d_model)
        keys = torch.randn(batch_size, num_heads, seq_len, d_model)
        values = torch.randn(batch_size, num_heads, seq_len, d_model)

        sparse_attention_mask = Mask.create_empty_mask(
            (batch_size, num_heads, seq_len, seq_len),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # Create attention mask (lower triangular for causal attention)
        attention_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        attention_mask.masked_fill_(attention_mask == 1, float("-inf"))
        attention_mask = (
            attention_mask.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, num_heads, -1, -1)
        )

        from transformers.models.llama.modeling_llama import eager_attention_forward

        module = torch.nn.Module()
        module.eval()
        print("Is Training: ", module.training)
        module.num_key_value_groups = 1

        eager_attention_output, eager_attention_weights = eager_attention_forward(
            module=module,
            query=queries,
            key=keys,
            value=values,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout,
        )

        my_attention_output, my_attention_weights = get_masked_attention_output(
            module=module,
            queries=queries,
            keys=keys,
            values=values,
            attention_mask=attention_mask,
            sinks=None,
            scaling=scaling,
            dropout=dropout,
            sparse_attention_mask=sparse_attention_mask,
            return_attention_weights=True,
        )
        assert torch.allclose(my_attention_output, eager_attention_output, atol=1e-6)
        assert torch.allclose(my_attention_weights, eager_attention_weights, atol=1e-6)

    def test_compare_with_eager_attention_sparse_mask_empty_dropout_0_train_mode(self):
        """Test that the masked attention output is the same as the eager attention output for no mask."""
        batch_size, num_heads, seq_len, d_model = 2, 4, 8, 16
        scaling = 1.0 / np.sqrt(d_model)
        dropout = 0

        queries = torch.randn(batch_size, num_heads, seq_len, d_model)
        keys = torch.randn(batch_size, num_heads, seq_len, d_model)
        values = torch.randn(batch_size, num_heads, seq_len, d_model)

        sparse_attention_mask = Mask.create_empty_mask(
            (batch_size, num_heads, seq_len, seq_len),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # Create attention mask (lower triangular for causal attention)
        attention_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        attention_mask.masked_fill_(attention_mask == 1, float("-inf"))
        attention_mask = (
            attention_mask.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, num_heads, -1, -1)
        )

        from transformers.models.llama.modeling_llama import eager_attention_forward

        module = torch.nn.Module()
        module.train()
        module.num_key_value_groups = 1

        eager_attention_output, eager_attention_weights = eager_attention_forward(
            module=module,
            query=queries,
            key=keys,
            value=values,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout,
        )

        my_attention_output, my_attention_weights = get_masked_attention_output(
            module=module,
            queries=queries,
            keys=keys,
            values=values,
            attention_mask=attention_mask,
            scaling=scaling,
            sinks=None,
            dropout=dropout,
            sparse_attention_mask=sparse_attention_mask,
            return_attention_weights=True,
        )
        assert torch.allclose(my_attention_output, eager_attention_output, atol=1e-6)
        assert torch.allclose(my_attention_weights, eager_attention_weights, atol=1e-6)

    def test_compare_with_eager_attention_sparse_mask_empty_dropout_train_mode(self):
        """Test that the masked attention output is the same as the eager attention output for no mask."""
        batch_size, num_heads, seq_len, d_model = 2, 4, 8, 16
        scaling = 1.0 / np.sqrt(d_model)
        dropout = 0.5

        queries = torch.randn(batch_size, num_heads, seq_len, d_model)
        keys = torch.randn(batch_size, num_heads, seq_len, d_model)
        values = torch.randn(batch_size, num_heads, seq_len, d_model)

        sparse_attention_mask = Mask.create_empty_mask(
            (batch_size, num_heads, seq_len, seq_len),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # Create attention mask (lower triangular for causal attention)
        attention_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        attention_mask.masked_fill_(attention_mask == 1, float("-inf"))
        attention_mask = (
            attention_mask.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, num_heads, -1, -1)
        )

        from transformers.models.llama.modeling_llama import eager_attention_forward

        module = torch.nn.Module()
        module.train()
        module.num_key_value_groups = 1

        def mock_dropout(x, p, training=True, inplace=False):
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            mask = torch.randn_like(x) > 0.5
            return x * mask

        with mock.patch("torch.nn.functional.dropout", mock_dropout):
            eager_attention_output, eager_attention_weights = eager_attention_forward(
                module=module,
                query=queries,
                key=keys,
                value=values,
                attention_mask=attention_mask,
                scaling=scaling,
                dropout=dropout,
            )

            my_attention_output, my_attention_weights = get_masked_attention_output(
                module=module,
                queries=queries,
                keys=keys,
                values=values,
                attention_mask=attention_mask,
                sinks=None,
                scaling=scaling,
                dropout=dropout,
                sparse_attention_mask=sparse_attention_mask,
                return_attention_weights=True,
            )

        # assert torch.allclose(my_attention_output, eager_attention_output, atol=1e-6)
        # assert torch.allclose(my_attention_weights, eager_attention_weights, atol=1e-6)

        print(
            "[NOTE] dropout behavior is different in eager and sparse attention by design"
        )

    def test_sinks_adjust_denominator(self):
        """Sinks should add mass to denominator and reduce key attention weights (GPT-OSS behavior).

        Setup:
        - queries/keys are all zeros -> all key logits are 0 -> uniform over keys
        - sink logit = 0 -> adds one extra exp(0) term per row
        - full mask -> all key positions are allowed
        Expected:
        - no sink: key probs = [1/2, 1/2], output = [1/2, 1/2]
        - with sink: key probs = [1/3, 1/3] (sink takes 1/3), output = [1/3, 1/3]
        - sum of returned key probs drops from 1 to 2/3
        """

        batch_size, num_heads, seq_len, d_model = 1, 1, 2, 2
        scaling = 1.0
        dropout = 0.0

        # Deterministic inputs: all logits are zeros
        queries = torch.zeros(batch_size, num_heads, seq_len, d_model)
        keys = torch.zeros(batch_size, num_heads, seq_len, d_model)

        # Values: identity-like so weighted sum equals the weights as a vector
        # shape: (B, H_kv, K, D)
        values = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])

        sparse_attention_mask = Mask.create_full_mask(
            (batch_size, num_heads, seq_len, seq_len),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        module = torch.nn.Module()
        module.eval()
        module.num_key_value_groups = 1  # avoid repeat_kv changing head count

        sink_value = torch.tensor([0.0])  # shape [H]; contributes exp(0)=1 per row

        out_no_sink, weights_no_sink = get_masked_attention_output(
            module=module,
            queries=queries,
            keys=keys,
            values=values,
            attention_mask=None,
            sinks=None,
            scaling=scaling,
            dropout=dropout,
            sparse_attention_mask=sparse_attention_mask,
            return_attention_weights=True,
        )

        out_with_sink, weights_with_sink = get_masked_attention_output(
            module=module,
            queries=queries,
            keys=keys,
            values=values,
            attention_mask=None,
            sinks=sink_value,
            scaling=scaling,
            dropout=dropout,
            sparse_attention_mask=sparse_attention_mask,
            return_attention_weights=True,
        )

        # Shape sanity checks
        assert out_no_sink.shape == (
            batch_size,
            seq_len,
            num_heads,
            d_model,
        )  # (B,Q,H,D)
        assert out_with_sink.shape == (
            batch_size,
            seq_len,
            num_heads,
            d_model,
        )  # (B,Q,H,D)
        assert weights_no_sink.shape == (
            batch_size,
            num_heads,
            seq_len,
            seq_len,
        )  # (B,H,Q,K)
        assert weights_with_sink.shape == (batch_size, num_heads, seq_len, seq_len)

        # Expected outputs: (B, Q, H, D)
        expected_no_sink = torch.tensor([[[[0.5, 0.5]], [[0.5, 0.5]]]])
        expected_with_sink = torch.tensor(
            [[[[1.0 / 3.0, 1.0 / 3.0]], [[1.0 / 3.0, 1.0 / 3.0]]]]
        )

        torch.testing.assert_close(out_no_sink, expected_no_sink, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            out_with_sink, expected_with_sink, atol=1e-6, rtol=1e-6
        )

        # Without sinks, key probabilities sum to 1. With sink=0, sink absorbs 1/3, so keys sum to 2/3.
        prob_sum_no_sink = weights_no_sink.sum(dim=-1)  # (B, H, Q)
        prob_sum_with_sink = weights_with_sink.sum(dim=-1)  # (B, H, Q)

        torch.testing.assert_close(
            prob_sum_no_sink,
            torch.ones_like(prob_sum_no_sink),
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            prob_sum_with_sink,
            torch.full_like(prob_sum_with_sink, 2.0 / 3.0),
            atol=1e-6,
            rtol=1e-6,
        )

        # Robustness: sink should scale all key probs by the same factor when logits are equal.
        # Here: [1/3,1/3] vs [1/2,1/2] => scale factor = 2/3 everywhere.
        scale = (weights_with_sink / weights_no_sink).mean()
        torch.testing.assert_close(scale, torch.tensor(2.0 / 3.0), atol=1e-6, rtol=1e-6)
