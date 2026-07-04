"""
:author: Aditya Desai
:copyright: 2025 Sparse Attention Hub
:license: Apache 2.0
:date: 2025-07-03
:summary: Tests for research attention. This file is part of the Sparse Attention Hub project.
"""

from unittest.mock import patch

import pytest
import torch


@pytest.mark.unit
class TestImports:
    """Test class for imports."""

    def test_imports(self):
        """Test that all imports are working."""
        from sparse_attention_hub.sparse_attention.research_attention import (
            ResearchAttention,
        )

        assert ResearchAttention is not None


@pytest.mark.unit
class TestResearchAttentionAndConfigCreation:
    """Test class for research attention and config creation."""

    def test_research_attention_creation(self):
        """Test that research attention can be created."""
        from sparse_attention_hub.sparse_attention.research_attention import (
            ResearchAttention,
            ResearchAttentionConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            LocalMasker,
            LocalMaskerConfig,
            OracleTopK,
            OracleTopKConfig,
            SinkMasker,
            SinkMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling import (
            RandomSamplingMasker,
            RandomSamplingMaskerConfig,
        )

        masker_configs = [
            SinkMaskerConfig(sink_size=10),
            LocalMaskerConfig(window_size=10),
            OracleTopKConfig(heavy_size=10),
            RandomSamplingMaskerConfig(sampling_rate=0.5),
        ]

        config = ResearchAttentionConfig(masker_configs=masker_configs)
        assert config is not None
        attention = ResearchAttention.create_from_config(config)
        assert attention is not None
        assert len(attention.maskers) == len(masker_configs)
        assert isinstance(attention.maskers[0], SinkMasker)
        assert isinstance(attention.maskers[1], LocalMasker)
        assert isinstance(attention.maskers[2], OracleTopK)
        assert isinstance(attention.maskers[3], RandomSamplingMasker)


@pytest.mark.unit
class TestInheritance:
    """Test class for inheritance."""

    def test_inheritance(self):
        """Test that research attention inherits from sparse attention."""
        from sparse_attention_hub.sparse_attention import (
            SparseAttention,
            SparseAttentionConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention import (
            ResearchAttention,
            ResearchAttentionConfig,
        )

        assert issubclass(ResearchAttention, SparseAttention)
        assert issubclass(ResearchAttentionConfig, SparseAttentionConfig)


@pytest.mark.unit
class TestSamplingMaskerValidation:
    """Test class for sampling masker validation."""

    def test_single_sampling_masker_allowed(self):
        """Test that a single sampling masker is allowed."""
        from sparse_attention_hub.sparse_attention import SparseAttentionConfig
        from sparse_attention_hub.sparse_attention.research_attention import (
            ResearchAttention,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            LocalMasker,
            LocalMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling import (
            RandomSamplingMasker,
            RandomSamplingMaskerConfig,
        )

        # Create maskers: one fixed, one sampling
        local_masker = LocalMasker.create_from_config(LocalMaskerConfig(window_size=10))
        sampling_masker = RandomSamplingMasker.create_from_config(
            RandomSamplingMaskerConfig(sampling_rate=0.5)
        )

        # This should not raise an error
        config = SparseAttentionConfig()
        attention = ResearchAttention(config, [local_masker, sampling_masker])
        assert attention is not None
        assert len(attention.maskers) == 2

    def test_multiple_sampling_maskers_rejected(self):
        """Test that multiple sampling maskers are rejected."""
        from sparse_attention_hub.sparse_attention import SparseAttentionConfig
        from sparse_attention_hub.sparse_attention.research_attention import (
            ResearchAttention,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling import (
            MagicPig,
            MagicPigConfig,
            RandomSamplingMasker,
            RandomSamplingMaskerConfig,
        )

        # Create two sampling maskers
        random_masker = RandomSamplingMasker.create_from_config(
            RandomSamplingMaskerConfig(sampling_rate=0.5)
        )
        magic_pig_masker = MagicPig.create_from_config(
            MagicPigConfig(lsh_l=4, lsh_k=16)
        )

        # This should raise an error
        config = SparseAttentionConfig()
        with pytest.raises(
            ValueError,
            match="Only one sampling masker supported for efficiency; consider implementing all sampling logic in one masker",
        ):
            ResearchAttention(config, [random_masker, magic_pig_masker])

    def test_multiple_sampling_maskers_via_config(self):
        """Test that multiple sampling maskers are rejected when created via config."""
        from sparse_attention_hub.sparse_attention.research_attention import (
            ResearchAttention,
            ResearchAttentionConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling import (
            MagicPigConfig,
            RandomSamplingMaskerConfig,
        )

        # Create config with two sampling maskers
        masker_configs = [
            RandomSamplingMaskerConfig(sampling_rate=0.5),
            MagicPigConfig(lsh_l=4, lsh_k=16),
        ]
        config = ResearchAttentionConfig(masker_configs=masker_configs)

        # This should raise an error
        with pytest.raises(
            ValueError,
            match="Only one sampling masker supported for efficiency; consider implementing all sampling logic in one masker",
        ):
            ResearchAttention.create_from_config(config)

    def test_no_sampling_maskers_allowed(self):
        """Test that no sampling maskers is allowed."""
        from sparse_attention_hub.sparse_attention import SparseAttentionConfig
        from sparse_attention_hub.sparse_attention.research_attention import (
            ResearchAttention,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            LocalMasker,
            LocalMaskerConfig,
            SinkMasker,
            SinkMaskerConfig,
        )

        # Create only fixed maskers
        local_masker = LocalMasker.create_from_config(LocalMaskerConfig(window_size=10))
        sink_masker = SinkMasker.create_from_config(SinkMaskerConfig(sink_size=5))

        # This should not raise an error
        config = SparseAttentionConfig()
        attention = ResearchAttention(config, [local_masker, sink_masker])
        assert attention is not None
        assert len(attention.maskers) == 2


@pytest.mark.unit
class TestSoftcapPlumbing:
    """Test class for softcap propagation through research attention."""

    def test_softcap_forwarded_to_masked_attention_when_passed_in_kwargs(self):
        """Empty maskers should pass call-site softcap to masked attention compute path."""
        from sparse_attention_hub.sparse_attention.research_attention import (
            ResearchAttention,
            ResearchAttentionConfig,
        )

        config = ResearchAttentionConfig(masker_configs=[])
        attention = ResearchAttention.create_from_config(config)

        queries = torch.randn(1, 1, 2, 4)
        keys = torch.randn(1, 1, 2, 4)
        values = torch.randn(1, 1, 2, 4)

        with patch(
            "sparse_attention_hub.sparse_attention.research_attention.base.get_masked_attention_output"
        ) as mock_get_masked_attention_output:
            mock_get_masked_attention_output.return_value = (
                torch.zeros_like(queries),
                torch.zeros(1, 1, 2, 2),
            )

            module = torch.nn.Module()
            module.training = False
            module.layer_type = "full_attention"

            attention.custom_attention(
                module=module,
                queries=queries,
                keys=keys,
                values=values,
                attention_mask=None,
                scaling=1.0,
                dropout=0.0,
                sparse_meta_data={},
                layer_idx=0,
                softcap=30.0,
            )

            assert mock_get_masked_attention_output.call_count == 1
            assert mock_get_masked_attention_output.call_args.kwargs["softcap"] == 30.0

    def test_softcap_override_respected_for_gemma_style_attention(self):
        """Caller-provided softcap must override Gemma layer_type default."""
        from sparse_attention_hub.sparse_attention.research_attention import (
            ResearchAttention,
            ResearchAttentionConfig,
        )

        config = ResearchAttentionConfig(masker_configs=[])
        attention = ResearchAttention.create_from_config(config)

        queries = torch.randn(1, 1, 2, 4)
        keys = torch.randn(1, 1, 2, 4)
        values = torch.randn(1, 1, 2, 4)

        with patch(
            "sparse_attention_hub.sparse_attention.research_attention.base.get_masked_attention_output"
        ) as mock_get_masked_attention_output:
            mock_get_masked_attention_output.return_value = (
                torch.zeros_like(queries),
                torch.zeros(1, 1, 2, 2),
            )

            module = torch.nn.Module()
            module.training = False
            module.layer_type = "full_attention"

            attention.custom_attention(
                module=module,
                queries=queries,
                keys=keys,
                values=values,
                attention_mask=None,
                scaling=1.0,
                dropout=0.0,
                sparse_meta_data={},
                layer_idx=0,
                softcap=45.0,
            )

            assert mock_get_masked_attention_output.call_count == 1
            assert mock_get_masked_attention_output.call_args.kwargs["softcap"] == 45.0


@pytest.mark.unit
class TestDensityLayerFiltering:
    """Test density metric filtering by attention layer type."""

    def test_density_logged_for_full_attention_layer(self):
        """Density should be logged when layer type is full_attention."""
        from sparse_attention_hub.sparse_attention.research_attention import (
            ResearchAttention,
            ResearchAttentionConfig,
        )

        attention = ResearchAttention.create_from_config(
            ResearchAttentionConfig(masker_configs=[])
        )

        queries = torch.randn(1, 1, 2, 4)
        keys = torch.randn(1, 1, 2, 4)
        values = torch.randn(1, 1, 2, 4)

        with patch(
            "sparse_attention_hub.sparse_attention.research_attention.base.get_masked_attention_output"
        ) as mock_get_masked_attention_output, patch(
            "sparse_attention_hub.sparse_attention.research_attention.base.MicroMetricLogger.is_metric_enabled"
        ) as mock_is_metric_enabled, patch(
            "sparse_attention_hub.sparse_attention.research_attention.base.MicroMetricLogger.log"
        ) as mock_log:
            mock_get_masked_attention_output.return_value = (
                torch.zeros_like(queries),
                torch.zeros(1, 1, 2, 2),
            )
            mock_is_metric_enabled.side_effect = (
                lambda metric_name: metric_name == "research_attention_density"
            )

            module = torch.nn.Module()
            module.training = False

            attention.custom_attention(
                module=module,
                queries=queries,
                keys=keys,
                values=values,
                attention_mask=None,
                scaling=1.0,
                dropout=0.0,
                sparse_meta_data={},
                layer_idx=0,
                layer_type="full_attention",
            )

            assert any(
                call.args[0] == "research_attention_density"
                for call in mock_log.call_args_list
            )

    def test_density_not_logged_for_sliding_attention_layer(self):
        """Density should not be logged when layer type is not full_attention."""
        from sparse_attention_hub.sparse_attention.research_attention import (
            ResearchAttention,
            ResearchAttentionConfig,
        )

        attention = ResearchAttention.create_from_config(
            ResearchAttentionConfig(masker_configs=[])
        )

        queries = torch.randn(1, 1, 2, 4)
        keys = torch.randn(1, 1, 2, 4)
        values = torch.randn(1, 1, 2, 4)

        with patch(
            "sparse_attention_hub.sparse_attention.research_attention.base.get_masked_attention_output"
        ) as mock_get_masked_attention_output, patch(
            "sparse_attention_hub.sparse_attention.research_attention.base.MicroMetricLogger.is_metric_enabled"
        ) as mock_is_metric_enabled, patch(
            "sparse_attention_hub.sparse_attention.research_attention.base.MicroMetricLogger.log"
        ) as mock_log:
            mock_get_masked_attention_output.return_value = (
                torch.zeros_like(queries),
                torch.zeros(1, 1, 2, 2),
            )
            mock_is_metric_enabled.side_effect = (
                lambda metric_name: metric_name == "research_attention_density"
            )

            module = torch.nn.Module()
            module.training = False

            attention.custom_attention(
                module=module,
                queries=queries,
                keys=keys,
                values=values,
                attention_mask=None,
                scaling=1.0,
                dropout=0.0,
                sparse_meta_data={},
                layer_idx=0,
                layer_type="sliding_attention",
            )

            density_calls = [
                call
                for call in mock_log.call_args_list
                if call.args[0] == "research_attention_density"
            ]
            assert len(density_calls) == 0

    def test_density_logged_for_unknown_layer(self):
        """Density SHOULD be logged for 'unknown' layers (e.g. Llama, which exposes
        no per-layer attention-type metadata). Regression guard for PR #76."""
        from sparse_attention_hub.sparse_attention.research_attention import (
            ResearchAttention,
            ResearchAttentionConfig,
        )

        attention = ResearchAttention.create_from_config(
            ResearchAttentionConfig(masker_configs=[])
        )

        queries = torch.randn(1, 1, 2, 4)
        keys = torch.randn(1, 1, 2, 4)
        values = torch.randn(1, 1, 2, 4)

        with patch(
            "sparse_attention_hub.sparse_attention.research_attention.base.get_masked_attention_output"
        ) as mock_get_masked_attention_output, patch(
            "sparse_attention_hub.sparse_attention.research_attention.base.MicroMetricLogger.is_metric_enabled"
        ) as mock_is_metric_enabled, patch(
            "sparse_attention_hub.sparse_attention.research_attention.base.MicroMetricLogger.log"
        ) as mock_log:
            mock_get_masked_attention_output.return_value = (
                torch.zeros_like(queries),
                torch.zeros(1, 1, 2, 2),
            )
            mock_is_metric_enabled.side_effect = (
                lambda metric_name: metric_name == "research_attention_density"
            )

            module = torch.nn.Module()
            module.training = False

            attention.custom_attention(
                module=module,
                queries=queries,
                keys=keys,
                values=values,
                attention_mask=None,
                scaling=1.0,
                dropout=0.0,
                sparse_meta_data={},
                layer_idx=0,
                layer_type="unknown",
            )

            assert any(
                call.args[0] == "research_attention_density"
                for call in mock_log.call_args_list
            )

    def test_unrecognized_layer_type_does_not_crash(self):
        """An out-of-whitelist layer_type (e.g. 'chunked_attention' on Llama4) must
        NOT raise. Regression guard for the removed assert in PR #76."""
        from sparse_attention_hub.sparse_attention.research_attention import (
            ResearchAttention,
            ResearchAttentionConfig,
        )

        attention = ResearchAttention.create_from_config(
            ResearchAttentionConfig(masker_configs=[])
        )

        queries = torch.randn(1, 1, 2, 4)
        keys = torch.randn(1, 1, 2, 4)
        values = torch.randn(1, 1, 2, 4)

        with patch(
            "sparse_attention_hub.sparse_attention.research_attention.base.get_masked_attention_output"
        ) as mock_get_masked_attention_output:
            mock_get_masked_attention_output.return_value = (
                torch.zeros_like(queries),
                torch.zeros(1, 1, 2, 2),
            )

            module = torch.nn.Module()
            module.training = False

            # Must not raise (assert on layer_type was removed in PR #76).
            attention.custom_attention(
                module=module,
                queries=queries,
                keys=keys,
                values=values,
                attention_mask=None,
                scaling=1.0,
                dropout=0.0,
                sparse_meta_data={},
                layer_idx=0,
                layer_type="chunked_attention",
            )
