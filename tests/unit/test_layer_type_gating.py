"""Unit tests for ResearchAttentionConfig.apply_to_layer_types layer-type gating."""

from types import SimpleNamespace

import pytest
import torch

from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttention,
    ResearchAttentionConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig,
    SinkMaskerConfig,
)
from sparse_attention_hub.sparse_attention.utils.mask_attention_utils import (
    get_true_attention_output,
)

LAYER_TYPES = [
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "full_attention",
]


def _make_attention(apply_to_layer_types):
    config = ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=2),
            LocalMaskerConfig(window_size=2),
        ],
        apply_to_layer_types=apply_to_layer_types,
    )
    return ResearchAttention.create_from_config(config)


def _make_module(layer_idx, layer_types=LAYER_TYPES):
    """Stand-in for a HF attention module: needs .config and .training."""
    return SimpleNamespace(
        config=SimpleNamespace(layer_types=layer_types),
        layer_idx=layer_idx,
        training=False,
    )


# Keys must greatly exceed queries: LocalMasker falls back to full attention when
# seq_len_keys <= window_size + seq_len_queries (basic_fixed.py:_should_use_full_attention),
# which is exactly the decode-shaped regime the sweep runs in anyway.
Q_LEN = 4
K_LEN = 64


def _tensors(q_len=Q_LEN, k_len=K_LEN, heads=2, dim=8):
    torch.manual_seed(0)
    return (
        torch.randn((1, heads, q_len, dim), dtype=torch.float32),
        torch.randn((1, heads, k_len, dim), dtype=torch.float32),
        torch.randn((1, heads, k_len, dim), dtype=torch.float32),
    )


def _causal_mask(q_len=Q_LEN, k_len=K_LEN):
    """Causal mask for a q_len-token chunk appended to a k_len-q_len KV cache."""
    offset = k_len - q_len
    rows = torch.arange(q_len).unsqueeze(1) + offset
    cols = torch.arange(k_len).unsqueeze(0)
    mask = torch.zeros(1, 1, q_len, k_len, dtype=torch.float32)
    mask.masked_fill_(
        (cols > rows).unsqueeze(0).unsqueeze(0), torch.finfo(torch.float32).min
    )
    return mask


def _run(attention, module, q, k, v, mask):
    return attention.custom_attention(
        module=module,
        queries=q,
        keys=k,
        values=v,
        attention_mask=mask,
        scaling=1.0,
        dropout=0.0,
        sparse_meta_data={},
        layer_idx=module.layer_idx,
    )


class TestLayerTypeGating:
    def test_gated_layer_returns_exact_dense_output(self):
        """A sliding layer must return bitwise the dense SDPA output."""
        q, k, v = _tensors()
        mask = _causal_mask()
        attention = _make_attention(("full_attention",))
        module = _make_module(layer_idx=0)  # sliding_attention

        out, weights = _run(attention, module, q, k, v, mask)
        expected, _ = get_true_attention_output(module, q, k, v, mask, 1.0, 0.0)

        assert torch.equal(out, expected)
        assert weights is None

    def test_ungated_layer_is_sparse(self):
        """A full layer must go through the maskers and differ from dense."""
        q, k, v = _tensors()
        mask = _causal_mask()
        attention = _make_attention(("full_attention",))
        module = _make_module(layer_idx=1)  # full_attention

        out, _ = _run(attention, module, q, k, v, mask)
        dense, _ = get_true_attention_output(module, q, k, v, mask, 1.0, 0.0)

        # sink=2 + local=2 out of 64 keys is a real restriction.
        assert not torch.allclose(out, dense, atol=1e-4)

    def test_default_none_is_sparse_everywhere(self):
        """Legacy behaviour: no gating configured => every layer is sparse."""
        q, k, v = _tensors()
        mask = _causal_mask()
        attention = _make_attention(None)
        module = _make_module(0)
        dense, _ = get_true_attention_output(module, q, k, v, mask, 1.0, 0.0)

        out, _ = _run(attention, module, q, k, v, mask)
        assert not torch.allclose(out, dense, atol=1e-4)

    def test_model_without_layer_types_is_unaffected(self):
        """Llama-style models expose no layer_types => never gated."""
        q, k, v = _tensors()
        mask = _causal_mask()
        attention = _make_attention(("full_attention",))
        module = SimpleNamespace(config=SimpleNamespace(), layer_idx=0, training=False)

        out, _ = _run(attention, module, q, k, v, mask)
        dense, _ = get_true_attention_output(module, q, k, v, mask, 1.0, 0.0)
        assert not torch.allclose(out, dense, atol=1e-4)

    def test_out_of_range_layer_idx_is_not_gated(self):
        q, k, v = _tensors()
        mask = _causal_mask()
        attention = _make_attention(("full_attention",))
        module = _make_module(layer_idx=99)

        out, _ = _run(attention, module, q, k, v, mask)
        dense, _ = get_true_attention_output(module, q, k, v, mask, 1.0, 0.0)
        assert not torch.allclose(out, dense, atol=1e-4)

    def test_attention_sink_model_raises_when_gated(self):
        """s_aux has no dense-fallback equivalent, so gating must refuse it."""
        q, k, v = _tensors()
        mask = _causal_mask()
        attention = _make_attention(("full_attention",))
        module = _make_module(layer_idx=0)

        with pytest.raises(NotImplementedError, match="attention-sink"):
            attention.custom_attention(
                module=module,
                queries=q,
                keys=k,
                values=v,
                attention_mask=mask,
                scaling=1.0,
                dropout=0.0,
                sparse_meta_data={},
                layer_idx=0,
                s_aux=torch.zeros(q.shape[1], dtype=torch.float32),
            )

    def test_olmo3_pattern_selects_expected_layers(self):
        """The Olmo-3 3:1 pattern must gate everything but [3,7,...]."""
        layer_types = [
            "full_attention" if (i + 1) % 4 == 0 else "sliding_attention"
            for i in range(32)
        ]
        attention = _make_attention(("full_attention",))
        selected = [
            i
            for i in range(32)
            if attention._applies_to_layer(
                _make_module(i, layer_types), {"layer_idx": i}
            )
        ]
        assert selected == [3, 7, 11, 15, 19, 23, 27, 31]
