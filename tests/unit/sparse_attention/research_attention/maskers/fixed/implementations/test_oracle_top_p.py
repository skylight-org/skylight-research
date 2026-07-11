"""
:author: Aditya Desai
:copyright: 2025 Sparse Attention Hub
:license: Apache 2.0
:date: 2025-07-07
:summary: Tests for OracleTopPMasker implementation.
"""

import mock
import pytest
import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
    TopPMasker,
    TopPMaskerConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    OracleTopPMasker,
    OracleTopPMaskerConfig,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask


@pytest.mark.unit
class TestOracleTopPMaskerConfig:
    """Test OracleTopPMaskerConfig class."""

    def test_oracle_top_p_masker_config_creation(self):
        """Test that oracle top p masker config can be created."""
        config = OracleTopPMaskerConfig(top_p=0.8)
        assert config is not None
        assert config.top_p == 0.8

    def test_oracle_top_p_masker_config_inheritance(self):
        """Test that oracle top p masker config inherits from TopPMaskerConfig."""
        assert issubclass(OracleTopPMaskerConfig, TopPMaskerConfig)

    def test_oracle_top_p_masker_config_validation_valid(self):
        """Test that config validation works for valid top_p values."""
        # Test valid values
        valid_values = [0.0, 0.1, 0.5, 0.8, 1.0]
        for top_p in valid_values:
            config = OracleTopPMaskerConfig(top_p=top_p)
            assert config.top_p == top_p

    def test_oracle_top_p_masker_config_validation_invalid(self):
        """Test that config validation raises error for invalid top_p values."""
        # Test invalid values
        invalid_values = [-0.1, 1.1, 2.0, -1.0]
        for top_p in invalid_values:
            with pytest.raises(ValueError, match="top_p must be in range"):
                OracleTopPMaskerConfig(top_p=top_p)


@pytest.mark.unit
class TestOracleTopPMasker:
    """Test OracleTopPMasker class."""

    def test_oracle_top_p_masker_creation(self):
        """Test that oracle top p masker can be created."""
        config = OracleTopPMaskerConfig(top_p=0.8)
        masker = OracleTopPMasker(config)
        assert isinstance(masker, OracleTopPMasker)
        assert masker.config == config
        assert masker.top_p == 0.8

    def test_oracle_top_p_masker_creation_from_config(self):
        """Test that oracle top p masker can be created from a config."""
        config = OracleTopPMaskerConfig(top_p=0.8)
        masker = OracleTopPMasker.create_from_config(config)
        assert isinstance(masker, OracleTopPMasker)
        assert masker.config == config

    def test_oracle_top_p_masker_inheritance(self):
        """Test that oracle top p masker inherits from TopPMasker."""
        assert issubclass(OracleTopPMasker, TopPMasker)

    def test_oracle_top_p_masker_create_from_config_invalid_type(self):
        """Test that create_from_config raises error for invalid config type."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            FixedMaskerConfig,
        )

        config = FixedMaskerConfig()
        with pytest.raises(ValueError, match="Invalid config type"):
            OracleTopPMasker.create_from_config(config)


@pytest.mark.unit
class TestOracleTopPMaskerMethods:
    """Test OracleTopPMasker methods."""

    def test_should_use_full_attention_small_sequence(self):
        """Test _should_use_full_attention returns True for small sequences."""
        config = OracleTopPMaskerConfig(top_p=0.8)
        masker = OracleTopPMasker(config)

        # Create mock tensor dimensions
        from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
            AttentionTensorDimensions,
        )

        # seq_len_keys = 5, top_p = 0.8, effective_size = int(0.8 * 5) = 4
        # Since 5 <= 4 is False, should return False
        dims = AttentionTensorDimensions(
            batch_size=2, num_heads=4, seq_len_queries=3, seq_len_keys=5
        )
        result = masker._should_use_full_attention(dims)
        assert result is False

        # seq_len_keys = 3, top_p = 0.8, effective_size = int(0.8 * 3) = 2
        # Since 3 <= 2 is False, should return False
        dims = AttentionTensorDimensions(
            batch_size=2, num_heads=4, seq_len_queries=3, seq_len_keys=3
        )
        result = masker._should_use_full_attention(dims)
        assert result is False

        # seq_len_keys = 1, top_p = 0.8, effective_size = int(0.8 * 1) = 0
        # Since 1 <= 0 is False, should return False
        dims = AttentionTensorDimensions(
            batch_size=2, num_heads=4, seq_len_queries=3, seq_len_keys=1
        )
        result = masker._should_use_full_attention(dims)
        assert result is False

    def test_should_use_full_attention_large_sequence(self):
        """Test _should_use_full_attention returns False for large sequences."""
        config = OracleTopPMaskerConfig(top_p=0.5)
        masker = OracleTopPMasker(config)

        from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
            AttentionTensorDimensions,
        )

        # seq_len_keys = 10, top_p = 0.5, effective_size = int(0.5 * 10) = 5
        # Since 10 <= 5 is False, should return False
        dims = AttentionTensorDimensions(
            batch_size=2, num_heads=4, seq_len_queries=3, seq_len_keys=10
        )
        result = masker._should_use_full_attention(dims)
        assert result is False

    def test_compute_attention_scores(self):
        """Test _compute_attention_scores method."""
        config = OracleTopPMaskerConfig(top_p=0.8)
        masker = OracleTopPMasker(config)

        def max_normalized(x):
            _max_attention_score = x.max(dim=-1, keepdim=True)[0]
            return x - _max_attention_score

        batch_size, num_heads, seq_len_queries, seq_len_keys = 2, 4, 3, 5
        dim = 64

        keys = torch.randn(batch_size, num_heads, seq_len_keys, dim)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, dim)

        scores = masker._compute_exp_attention_scores(
            keys,
            queries,
            previous_dense_mask=torch.zeros(
                batch_size, num_heads, seq_len_queries, seq_len_keys
            ),
            attention_mask=None,
            scaling=1.0,
        )

        expected_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        assert (
            scores.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {scores.shape}"

        # Verify it's the correct computation
        expected_scores = torch.exp(max_normalized(queries @ keys.transpose(-2, -1)))
        assert torch.allclose(scores, expected_scores)

    def test_compute_top_p_thresholds_2d(self):
        """Test _compute_top_p_thresholds with 2D tensor."""
        config = OracleTopPMaskerConfig(top_p=0.8)
        masker = OracleTopPMasker(config)

        # Test with 2D tensor (2 queries, 5 keys)
        scores = torch.tensor(
            [
                [0.1, 0.3, 0.2, 0.4, 0.1],  # Query 1
                [0.2, 0.1, 0.5, 0.1, 0.1],  # Query 2
            ],
            dtype=torch.float32,
        )

        thresholds = masker._compute_top_p_thresholds(scores, 0.8)

        # Expected: top_p=0.8 means we want 80% of cumulative sum
        # Query 1: [0.1, 0.3, 0.2, 0.4, 0.1] -> sorted: [0.4, 0.3, 0.2, 0.1, 0.1]
        #          cumsum: [0.4, 0.7, 0.9, 1.0, 1.1] -> normalized: [0.36, 0.64, 0.82, 0.91, 1.0]
        #          threshold at 0.82 >= 0.8 -> position 2 -> value 0.2
        # Query 2: [0.2, 0.1, 0.5, 0.1, 0.1] -> sorted: [0.5, 0.2, 0.1, 0.1, 0.1]
        #          cumsum: [0.5, 0.7, 0.8, 0.9, 1.0] -> normalized: [0.5, 0.7, 0.8, 0.9, 1.0]
        #          threshold at 0.8 >= 0.8 -> position 2 -> value 0.1

        expected_thresholds = torch.tensor([[0.2], [0.1]], dtype=torch.float32)
        assert torch.allclose(thresholds, expected_thresholds, atol=1e-6)

    def test_compute_top_p_thresholds_4d(self):
        """Test _compute_top_p_thresholds with 4D tensor."""
        config = OracleTopPMaskerConfig(top_p=0.6)
        masker = OracleTopPMasker(config)

        # Test with 4D tensor (batch=1, heads=1, queries=2, keys=3)
        scores = torch.tensor(
            [[[[0.1, 0.3, 0.2], [0.2, 0.1, 0.5]]]],  # Query 1  # Query 2
            dtype=torch.float32,
        )

        thresholds = masker._compute_top_p_thresholds(scores, 0.6)

        # Expected shape: (1, 1, 2, 1)
        expected_shape = (1, 1, 2, 1)
        assert thresholds.shape == expected_shape

        # Query 1: [0.1, 0.3, 0.2] -> sorted: [0.3, 0.2, 0.1]
        #          cumsum: [0.3, 0.5, 0.6] -> normalized: [0.5, 0.83, 1.0]
        #          threshold at 0.83 >= 0.6 -> position 1 -> value 0.2
        # Query 2: [0.2, 0.1, 0.5] -> sorted: [0.5, 0.2, 0.1]
        #          cumsum: [0.5, 0.7, 0.8] -> normalized: [0.625, 0.875, 1.0]
        #          threshold at 0.625 >= 0.6 -> position 0 -> value 0.5

        expected_thresholds = torch.tensor([[[[0.2], [0.5]]]], dtype=torch.float32)
        assert torch.allclose(thresholds, expected_thresholds, atol=1e-6)

    def test_compute_top_p_thresholds_edge_cases(self):
        """Test _compute_top_p_thresholds with edge cases."""
        config = OracleTopPMaskerConfig(top_p=0.8)
        masker = OracleTopPMasker(config)
        # Test with all same values
        scores = torch.ones(2, 3, dtype=torch.float32)
        thresholds = masker._compute_top_p_thresholds(scores, 0.8)
        assert torch.allclose(thresholds, torch.ones(2, 1, dtype=torch.float32))

        # Test with top_p = 1.0 (should select all)
        scores = torch.tensor([[0.1, 0.3, 0.2]], dtype=torch.float32)
        thresholds = masker._compute_top_p_thresholds(scores, 1.0)
        # Should select minimum value
        assert torch.allclose(thresholds, torch.tensor([[0.1]], dtype=torch.float32))

        # Test with top_p = 0.0 (should select maximum value)
        scores = torch.tensor([[0.1, 0.3, 0.2]], dtype=torch.float32)
        thresholds = masker._compute_top_p_thresholds(scores, 0.0)
        # Should select maximum value
        assert torch.allclose(thresholds, torch.tensor([[0.3]], dtype=torch.float32))


@pytest.mark.unit
class TestOracleTopPMaskerAddMask:
    """Test OracleTopPMasker add_mask method."""

    def test_add_mask_full_previous_mask(self):
        """Test that OracleTopPMasker returns full mask when previous mask is full."""
        config = OracleTopPMaskerConfig(top_p=0.8)
        masker = OracleTopPMasker(config)

        batch_size, num_heads, seq_len_queries, seq_len_keys = 2, 4, 5, 8

        # Create mock inputs
        keys = torch.randn(batch_size, num_heads, seq_len_keys, 64)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, 64)
        values = torch.randn(batch_size, num_heads, seq_len_keys, 64)
        attention_mask = torch.ones(batch_size, seq_len_keys)

        # Create full mask as previous mask
        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        full_previous_mask = Mask.create_full_mask(
            mask_shape, dtype=torch.float32, device=torch.device("cpu")
        )

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data={},
            previous_mask=full_previous_mask,
        )

        assert result.is_full_mask()
        assert result.shape == mask_shape

    def test_add_mask_small_sequence(self):
        """Test that OracleTopPMasker returns full mask when sequence is small enough."""
        config = OracleTopPMaskerConfig(top_p=0.9)  # 90% of sequence
        masker = OracleTopPMasker(config)

        batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 2, 3, 5

        # Create mock inputs
        keys = torch.randn(batch_size, num_heads, seq_len_keys, 32)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, 32)
        values = torch.randn(batch_size, num_heads, seq_len_keys, 32)
        attention_mask = None

        # Create empty mask as previous mask
        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        empty_previous_mask = Mask.create_empty_mask(
            mask_shape, dtype=torch.float32, device=torch.device("cpu")
        )

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data={},
            previous_mask=empty_previous_mask,
        )

        # With top_p=0.9 and seq_len_keys=5, effective_size = int(0.9 * 5) = 4
        # Since 5 <= 4 is False, should NOT return full mask
        assert not result.is_full_mask()
        assert result.shape == mask_shape

    def test_add_mask_top_p_selection(self):
        """Test that OracleTopPMasker correctly selects top-p tokens."""
        config = OracleTopPMaskerConfig(top_p=0.6)  # 60% of sequence
        masker = OracleTopPMasker(config)

        batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 1, 2, 1000

        # Create deterministic inputs for predictable selection
        torch.manual_seed(42)
        keys = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, 16)
        values = torch.randn(batch_size, num_heads, seq_len_keys, 16)

        # Create empty mask as previous mask
        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        empty_previous_mask = Mask.create_empty_mask(
            mask_shape, dtype=torch.float32, device=torch.device("cpu")
        )
        scores = torch.rand(batch_size, num_heads, seq_len_queries, seq_len_keys)
        with mock.patch.object(
            masker, "_compute_exp_attention_scores", return_value=scores
        ):
            result = masker.add_mask(
                keys=keys,
                queries=queries,
                values=values,
                attention_mask=None,
                scaling=1.0,
                dropout=0.0,
                sparse_meta_data={},
                previous_mask=empty_previous_mask,
            )

        # Convert to dense to check pattern
        result_dense = result.get_dense_mask()
        assert result_dense.shape == mask_shape

        expected_result = torch.ones_like(scores[:, :, :, 0]) * 0.6

        thresholds = masker._compute_top_p_thresholds(scores, 0.6)
        achieved_result = torch.sum(
            scores * (scores >= thresholds), dim=-1
        ) / torch.sum(scores, dim=-1)

        assert torch.allclose(achieved_result, expected_result, rtol=0.1)

    def test_add_mask_edge_case_top_p_zero(self):
        """Test OracleTopPMasker with top_p = 0.0."""
        config = OracleTopPMaskerConfig(top_p=0.0)
        masker = OracleTopPMasker(config)

        batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 1, 1, 5

        keys = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, 16)
        values = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        attention_mask = None

        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        empty_previous_mask = Mask.create_empty_mask(
            mask_shape, dtype=torch.float32, device=torch.device("cpu")
        )

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data={},
            previous_mask=empty_previous_mask,
        )

        result_dense = result.get_dense_mask()
        # With top_p=0.0, should select only the highest scoring position
        total_active = torch.sum(result_dense[0, 0, 0] != 0).item()
        assert total_active == 1, f"Should have 1 active position, got {total_active}"

    def test_add_mask_edge_case_top_p_one(self):
        """Test OracleTopPMasker with top_p = 1.0."""
        config = OracleTopPMaskerConfig(top_p=1.0)
        masker = OracleTopPMasker(config)

        batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 1, 1, 5

        keys = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, 16)
        values = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        attention_mask = torch.ones(batch_size, seq_len_keys)

        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        empty_previous_mask = Mask.create_empty_mask(
            mask_shape, dtype=torch.float32, device=torch.device("cpu")
        )

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data={},
            previous_mask=empty_previous_mask,
        )

        result_dense = result.get_dense_mask()
        # With top_p=1.0, should select all positions
        total_active = torch.sum(result_dense[0, 0, 0] != 0).item()
        assert (
            total_active == seq_len_keys
        ), f"Should have {seq_len_keys} active positions, got {total_active}"


@pytest.mark.unit
class TestOracleTopPMaskerIntegration:
    """Test OracleTopPMasker integration scenarios."""

    def test_oracle_top_p_masker_registry_registration(self):
        """Test that OracleTopPMasker is properly registered."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
            ResearchMasker,
        )

        config = OracleTopPMaskerConfig(top_p=0.8)
        masker = ResearchMasker.create_masker_from_config(config)
        assert isinstance(masker, OracleTopPMasker)
        assert masker.top_p == 0.8

    def test_oracle_top_p_masker_with_different_shapes(self):
        """Test OracleTopPMasker with different tensor shapes."""
        config = OracleTopPMaskerConfig(top_p=0.7)
        masker = OracleTopPMasker(config)

        # Test with different shapes
        test_shapes = [
            (1, 1, 3, 5),  # Small
            (2, 4, 8, 16),  # Medium
            (4, 8, 16, 32),  # Large
        ]

        for batch_size, num_heads, seq_len_queries, seq_len_keys in test_shapes:
            keys = torch.randn(batch_size, num_heads, seq_len_keys, 64)
            queries = torch.randn(batch_size, num_heads, seq_len_queries, 64)
            values = torch.randn(batch_size, num_heads, seq_len_keys, 64)
            attention_mask = None

            mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
            empty_previous_mask = Mask.create_empty_mask(
                mask_shape, dtype=torch.float32, device=torch.device("cpu")
            )

            result = masker.add_mask(
                keys=keys,
                queries=queries,
                values=values,
                attention_mask=attention_mask,
                scaling=1.0,
                dropout=0.0,
                sparse_meta_data={},
                previous_mask=empty_previous_mask,
            )

            assert result.shape == mask_shape
            assert not result.is_full_mask()  # Should not be full for these shapes

    def test_oracle_top_p_masker_device_consistency(self):
        """Test that OracleTopPMasker works on different devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        config = OracleTopPMaskerConfig(top_p=0.8)
        masker = OracleTopPMasker(config)

        batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 1, 2, 5

        # Test on CPU
        keys_cpu = torch.randn(batch_size, num_heads, seq_len_keys, 16, device="cpu")
        queries_cpu = torch.randn(
            batch_size, num_heads, seq_len_queries, 16, device="cpu"
        )
        values_cpu = torch.randn(batch_size, num_heads, seq_len_keys, 16, device="cpu")
        attention_mask_cpu = None

        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        empty_previous_mask_cpu = Mask.create_empty_mask(
            mask_shape, dtype=torch.float32, device=torch.device("cpu")
        )

        result_cpu = masker.add_mask(
            keys=keys_cpu,
            queries=queries_cpu,
            values=values_cpu,
            attention_mask=attention_mask_cpu,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data={},
            previous_mask=empty_previous_mask_cpu,
        )

        # Test on GPU
        keys_gpu = keys_cpu.cuda()
        queries_gpu = queries_cpu.cuda()
        values_gpu = values_cpu.cuda()
        attention_mask_gpu = None
        empty_previous_mask_gpu = Mask.create_empty_mask(
            mask_shape, dtype=torch.float32, device=torch.device("cuda")
        )

        result_gpu = masker.add_mask(
            keys=keys_gpu,
            queries=queries_gpu,
            values=values_gpu,
            attention_mask=attention_mask_gpu,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data={},
            previous_mask=empty_previous_mask_gpu,
        )

        # Results should be equivalent (after moving to same device)
        result_cpu_dense = result_cpu.get_dense_mask()
        result_gpu_dense = result_gpu.get_dense_mask().cpu()
        assert torch.allclose(result_cpu_dense, result_gpu_dense)


@pytest.mark.unit
class TestOracleTopPSoftcap:
    """Test softcap handling in OracleTopPMasker."""

    def test_softcap_changes_top_p_selection(self):
        """Softcap flattens logits and flips the top-p selection deterministically.

        Setup: d=2, B=H=1, Q=1, K=8; queries=[[1, 0]], keys[i]=[logit_i, 0]
        with logits [10, 5, 0, 0, 0, 0, 0, 0], scaling=1.0, top_p=0.9.
        (int(0.9 * 8) = 7 < 8, so the full-attention shortcut is not taken.)

        Without softcap: exp scores after row-max subtraction are
        [1, e^-5, e^-10 x6]; normalized cumulative mass of the top score
        alone is ~0.993 >= 0.9, so only 1 position is selected.

        With softcap=1.0: capped logits tanh(10)~=1.0, tanh(5)~=0.9999,
        tanh(0)=0 give exp adjusted scores [1, ~1, e^-1 x6] (total ~4.21);
        normalized cumsum is [0.238, 0.475, 0.563, 0.650, 0.738, 0.825,
        0.913, 1.0], which crosses 0.9 only at index 6, so the threshold is
        e^-1 and all 8 positions are selected.
        """
        config = OracleTopPMaskerConfig(top_p=0.9)
        masker = OracleTopPMasker(config)

        batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 1, 1, 8

        logits = torch.tensor([10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        queries = torch.tensor([[[[1.0, 0.0]]]])
        keys = torch.zeros(batch_size, num_heads, seq_len_keys, 2)
        keys[0, 0, :, 0] = logits
        values = torch.zeros(batch_size, num_heads, seq_len_keys, 2)

        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)

        def run(**extra_kwargs):
            previous_mask = Mask.create_empty_mask(
                mask_shape, dtype=torch.float32, device=torch.device("cpu")
            )
            return masker.add_mask(
                keys=keys,
                queries=queries,
                values=values,
                attention_mask=None,
                scaling=1.0,
                dropout=0.0,
                sparse_meta_data={},
                previous_mask=previous_mask,
                **extra_kwargs,
            )

        result_no_softcap = run()
        result_softcap = run(softcap=1.0)

        dense_no_softcap = result_no_softcap.get_dense_mask()
        dense_softcap = result_softcap.get_dense_mask()

        active_no_softcap = torch.sum(dense_no_softcap != 0).item()
        active_softcap = torch.sum(dense_softcap != 0).item()

        assert active_no_softcap == 1, (
            f"Without softcap only the top score should be selected, "
            f"got {active_no_softcap} active positions"
        )
        assert active_softcap == seq_len_keys, (
            f"With softcap=1.0 all {seq_len_keys} positions should be selected, "
            f"got {active_softcap} active positions"
        )
        assert not torch.equal(dense_no_softcap, dense_softcap)

    def test_compute_exp_scores_applies_softcap_exactly(self):
        """_compute_exp_attention_scores matches the inline softcap formula."""
        config = OracleTopPMaskerConfig(top_p=0.8)
        masker = OracleTopPMasker(config)

        torch.manual_seed(0)
        queries = torch.randn(1, 1, 2, 4)
        keys = torch.randn(1, 1, 5, 4)
        previous_dense_mask = torch.zeros(1, 1, 2, 5)
        scaling = 0.3
        cap = 2.0

        raw_scores = torch.matmul(queries, keys.transpose(2, 3)) * scaling

        # With softcap: exp(cap * tanh(s * scaling / cap) - rowmax)
        scores_softcap = masker._compute_exp_attention_scores(
            keys,
            queries,
            previous_dense_mask,
            attention_mask=None,
            scaling=scaling,
            softcap=cap,
        )
        capped = cap * torch.tanh(raw_scores / cap)
        expected_softcap = torch.exp(capped - capped.max(dim=-1, keepdim=True)[0])
        assert torch.allclose(scores_softcap, expected_softcap, atol=1e-6)

        # Regression: softcap=None must match the inline formula WITHOUT tanh
        scores_none = masker._compute_exp_attention_scores(
            keys,
            queries,
            previous_dense_mask,
            attention_mask=None,
            scaling=scaling,
            softcap=None,
        )
        expected_none = torch.exp(raw_scores - raw_scores.max(dim=-1, keepdim=True)[0])
        assert torch.allclose(scores_none, expected_none, atol=1e-6)
