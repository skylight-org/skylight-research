"""
:author: Aditya Desai
:copyright: 2025 Sparse Attention hub
:license: Apache 2.0
:date: 2025-06-29
:summary: Tests for masker base classes, configs, and create_from_config methods. This file is part of the Sparse Attention Hub project.
"""

import pytest


@pytest.mark.unit
class TestMaskerConfigImports:
    """Test class for masker config imports."""

    def test_masker_config_imports(self):
        """Test that all masker configs can be imported."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
            MaskerConfig,
            ResearchMasker,
        )

        assert MaskerConfig is not None
        assert ResearchMasker is not None

        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            CausalMasker,
            DoubleSparsityTopKMasker,
            DoubleSparsityTopKMaskerConfig,
            FixedMasker,
            FixedMaskerConfig,
            HashAttentionTopKMasker,
            HashAttentionTopKMaskerConfig,
            LocalMasker,
            LocalMaskerConfig,
            OracleTopK,
            OracleTopKConfig,
            PQCache,
            PQCacheConfig,
            ScaNNTopKMasker,
            ScaNNTopKMaskerConfig,
            SinkMasker,
            SinkMaskerConfig,
        )

        assert CausalMasker is not None
        assert DoubleSparsityTopKMasker is not None
        assert DoubleSparsityTopKMaskerConfig is not None
        assert FixedMasker is not None
        assert FixedMaskerConfig is not None
        assert HashAttentionTopKMasker is not None
        assert HashAttentionTopKMaskerConfig is not None
        assert LocalMasker is not None
        assert LocalMaskerConfig is not None
        assert OracleTopK is not None
        assert OracleTopKConfig is not None
        assert PQCache is not None
        assert PQCacheConfig is not None
        assert ScaNNTopKMasker is not None
        assert ScaNNTopKMaskerConfig is not None
        assert SinkMasker is not None
        assert SinkMaskerConfig is not None

        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling import (
            SamplingMasker,
            SamplingMaskerConfig,
        )

        assert SamplingMasker is not None
        assert SamplingMaskerConfig is not None


@pytest.mark.unit
class TestMaskerConfigCreation:
    """Test class for masker config creation."""

    def test_base_masker_config_creation(self):
        """Test that all base masker configs can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers import (
            MaskerConfig,
        )

        config = MaskerConfig()
        assert config is not None


@pytest.mark.unit
class TestMaskerImports:
    """Test class for masker imports."""

    def test_base_masker_imports(self):
        """Test that all base masker classes can be imported."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers import (
            MaskerConfig,
            ResearchMasker,
        )

        assert ResearchMasker is not None
        assert MaskerConfig is not None


@pytest.mark.unit
class TestConcreteMaskerCreation:
    """Test class for masker configs and create_from_config methods."""

    def test_masker_creation_from_config_for_local_masker(self):
        """Test that LocalMasker can be created from a config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers import (
            ResearchMasker,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            LocalMasker,
            LocalMaskerConfig,
        )

        config = LocalMaskerConfig(window_size=10)
        masker = ResearchMasker.create_masker_from_config(config)
        assert type(masker) is LocalMasker
        assert masker.config == config

    def test_masker_creation_from_config_for_causal_masker(self):
        """Test that CausalMasker can be created from a config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers import (
            ResearchMasker,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            CausalMasker,
            FixedMaskerConfig,
        )

        config = FixedMaskerConfig()
        masker = ResearchMasker.create_masker_from_config(config)
        assert type(masker) is CausalMasker
        assert masker.config == config

    def test_masker_creation_from_config_for_sink_masker(self):
        """Test that SinkMasker can be created from a config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers import (
            ResearchMasker,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            SinkMasker,
            SinkMaskerConfig,
        )

        config = SinkMaskerConfig(sink_size=5)
        masker = ResearchMasker.create_masker_from_config(config)
        assert type(masker) is SinkMasker
        assert masker.config == config

    def test_masker_creation_from_config_for_oracle_top_k_masker(self):
        """Test that OracleTopK can be created from a config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers import (
            ResearchMasker,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            OracleTopK,
            OracleTopKConfig,
        )

        config = OracleTopKConfig(heavy_size=100)
        masker = ResearchMasker.create_masker_from_config(config)
        assert type(masker) is OracleTopK
        assert masker.config == config

    def test_masker_creation_from_config_for_pq_cache_masker(self):
        """Test that PQCache can be created from a config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers import (
            ResearchMasker,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            PQCache,
            PQCacheConfig,
        )

        config = PQCacheConfig(
            heavy_size=100,
            pq_group_factor=2,  # Assuming head_dim=16, pq_sub_dim=8, so pq_group_factor=2
            pq_bits=4,
            kmeans_iter=10,
            init_offset=0,
            metric="euclidean",
        )
        masker = ResearchMasker.create_masker_from_config(config)
        assert type(masker) is PQCache
        assert masker.config == config

    def test_masker_creation_from_config_for_hash_attention_top_k_masker(self):
        """Test that HashAttentionTopKMasker can be created from a config."""
        import torch

        from sparse_attention_hub.sparse_attention.research_attention.maskers import (
            ResearchMasker,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            HashAttentionTopKMasker,
            HashAttentionTopKMaskerConfig,
        )

        # Create sample weight tensors
        sample_weights = {
            0: {
                "key_matrix": [torch.randn(2, 64, 32), torch.randn(2, 32, 8)],
                "key_bias": [torch.randn(2, 32), torch.randn(2, 8)],
                "query_matrix": [torch.randn(2, 64, 32), torch.randn(2, 32, 8)],
                "query_bias": [torch.randn(2, 32), torch.randn(2, 8)],
            }
        }

        config = HashAttentionTopKMaskerConfig(
            heavy_size=100,
            hat_bits=8,
            hat_mlp_layers=2,
            hat_mlp_hidden_size=64,
            hat_mlp_activation="relu",
            hat_weights=sample_weights,
        )
        masker = ResearchMasker.create_masker_from_config(config)
        assert type(masker) is HashAttentionTopKMasker
        assert masker.config == config

    def test_masker_creation_from_config_for_double_sparsity_top_k_masker(self):
        """Test that DoubleSparsityTopKMasker can be created from a config."""
        import os
        import tempfile

        from sparse_attention_hub.sparse_attention.research_attention.maskers import (
            ResearchMasker,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            DoubleSparsityTopKMasker,
            DoubleSparsityTopKMaskerConfig,
        )

        # Create a temporary file for sorted_channel_file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as temp_file:
            temp_file.write('{"test": "data"}')
            temp_file_path = temp_file.name

        try:
            config = DoubleSparsityTopKMaskerConfig(
                heavy_size=100,
                group_factor=4,
                label_bits=8,
                sorted_channel_file=temp_file_path,
            )
            masker = ResearchMasker.create_masker_from_config(config)
            assert type(masker) is DoubleSparsityTopKMasker
            assert masker.config == config
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def test_masker_creation_from_config_for_random_sampling_masker(self):
        """Test that RandomSamplingMasker can be created from a config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers import (
            ResearchMasker,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling import (
            RandomSamplingMasker,
            RandomSamplingMaskerConfig,
        )

        config = RandomSamplingMaskerConfig(sampling_rate=0.5)
        masker = ResearchMasker.create_masker_from_config(config)
        assert type(masker) is RandomSamplingMasker
        assert masker.config == config

    def test_masker_creation_from_config_for_magic_pig_masker(self):
        """Test that MagicPig can be created from a config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers import (
            ResearchMasker,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling import (
            MagicPig,
            MagicPigConfig,
        )

        config = MagicPigConfig(lsh_l=4, lsh_k=8)
        masker = ResearchMasker.create_masker_from_config(config)
        assert type(masker) is MagicPig
        assert masker.config == config
