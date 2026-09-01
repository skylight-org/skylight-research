"""
:author: Aditya Desai
:copyright: 2025 Sparse Attention Hub
:license: Apache 2.0
:date: 2025-06-29
:summary: Tests for masker implementations imports.
"""

import pytest


@pytest.mark.unit
class TestImplementationsImports:
    def test_implementations_imports(self):
        """Test that all implementation classes can be imported."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            CausalMasker,
            DoubleSparsityTopKMasker,
            DoubleSparsityTopKMaskerConfig,
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

        # Test masker classes
        assert LocalMasker is not None
        assert CausalMasker is not None
        assert SinkMasker is not None
        assert OracleTopK is not None
        assert PQCache is not None
        assert ScaNNTopKMasker is not None
        assert HashAttentionTopKMasker is not None
        assert DoubleSparsityTopKMasker is not None

        # Test config classes
        assert LocalMaskerConfig is not None
        assert SinkMaskerConfig is not None
        assert OracleTopKConfig is not None
        assert PQCacheConfig is not None
        assert ScaNNTopKMaskerConfig is not None
        assert HashAttentionTopKMaskerConfig is not None
        assert DoubleSparsityTopKMaskerConfig is not None
