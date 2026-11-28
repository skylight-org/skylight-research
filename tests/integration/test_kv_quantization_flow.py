"""End-to-end KV quantization through ModelAdapterHF on a tiny random Qwen3 (CPU, offline).

The model and a word-level tokenizer are built in a temporary folder, so nothing is
downloaded. These are plumbing checks (the hook runs on every layer, the store fills
and is freed, fake and packed agree), not accuracy measurements.
"""

from typing import Any, Dict, List, Optional

import pytest
import torch

pytest.importorskip("tokenizers")

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import (  # noqa: E402
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from sparse_attention_hub.adapters import ModelAdapterHF, Request  # noqa: E402
from sparse_attention_hub.kv_quantization import (  # noqa: E402
    CACHE_KEY,
    DefaultCachePolicyConfig,
    IntQuantizerConfig,
    KVQuantizationConfig,
    RegionConfig,
)
from sparse_attention_hub.sparse_attention.research_attention import (  # noqa: E402
    ResearchAttentionConfig,
)

pytestmark = pytest.mark.integration

WORDS = [f"w{i}" for i in range(60)]
NUM_LAYERS = 2
CONTEXT = " ".join(WORDS[i % len(WORDS)] for i in range(80))
QUESTION = "w1 w2 w3 w4"
MAX_NEW_TOKENS = 6


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("tiny_qwen3")
    vocab = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2}
    vocab.update({word: i + 3 for i, word in enumerate(WORDS)})
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    tokenizer.save_pretrained(path)

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=len(vocab),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=512,
        bos_token_id=1,
        eos_token_id=2,
    )
    Qwen3ForCausalLM(config).save_pretrained(path)
    return str(path)


def _kv_config(store_codes: bool) -> KVQuantizationConfig:
    return KVQuantizationConfig(
        key_quantizer=IntQuantizerConfig(bits=2),
        value_quantizer=IntQuantizerConfig(bits=2),
        cache_policy=DefaultCachePolicyConfig(
            key_sink=RegionConfig(tokens=2),
            value_sink=RegionConfig(tokens=2),
            key_local=RegionConfig(tokens=16),
            value_local=RegionConfig(tokens=16),
            key_block_tokens=16,
            value_block_tokens=16,
        ),
        store_codes=store_codes,
    )


def _run(model_dir: str, kv_config: Optional[KVQuantizationConfig]) -> Dict[str, Any]:
    """Answer one question; capture the store's stats just before it is freed."""
    adapter = ModelAdapterHF(
        model_name=model_dir,
        sparse_attention_config=ResearchAttentionConfig(
            masker_configs=[], kv_quantization_config=kv_config
        ),
        device="cpu",
    )
    captured: List[Optional[Dict[str, Any]]] = []
    metas: List[Dict[str, Any]] = []
    original_clear = adapter.sparse_attention.clear_request_state

    def spy(meta: Dict[str, Any]) -> None:
        store = meta.get(CACHE_KEY)
        captured.append(None if store is None else store.stats())
        metas.append(meta)
        original_clear(meta)

    adapter.sparse_attention.clear_request_state = spy
    response = adapter.process_request(
        Request(context=CONTEXT, questions=QUESTION, answer_prefix=""),
        generation_kwargs={"max_new_tokens": MAX_NEW_TOKENS},
        request_kwargs={},
    )
    return {"response": response.responses, "stats": captured, "metas": metas}


class TestKVQuantizationFlow:
    def test_off_creates_no_store(self, tiny_model_dir: str) -> None:
        result = _run(tiny_model_dir, None)
        assert result["stats"] == [None]

    def test_store_fills_on_every_layer_and_is_freed(self, tiny_model_dir: str) -> None:
        result = _run(tiny_model_dir, _kv_config(store_codes=False))
        (stats,) = result["stats"]
        assert stats is not None
        assert sorted(stats["layers"]) == list(range(NUM_LAYERS))
        seen = {layer["seen_tokens"] for layer in stats["layers"].values()}
        assert len(seen) == 1, "every layer must see the same history"
        for layer in stats["layers"].values():
            assert layer["key_sink_tokens"] == 2
            assert layer["key_compressed_tokens"] > 0
            assert layer["value_compressed_tokens"] > 0
            assert (
                layer["key_sink_tokens"]
                + layer["key_compressed_tokens"]
                + layer["key_recent_tokens"]
                == layer["seen_tokens"]
            )
        assert (
            CACHE_KEY not in result["metas"][0]
        ), "store must be freed after the question"

    def test_fake_and_packed_generate_the_same_text(self, tiny_model_dir: str) -> None:
        fake = _run(tiny_model_dir, _kv_config(store_codes=False))
        packed = _run(tiny_model_dir, _kv_config(store_codes=True))
        assert fake["response"] == packed["response"]
        assert fake["stats"][0]["layers"] == packed["stats"][0]["layers"]
        assert packed["stats"][0]["store_bytes"] < fake["stats"][0]["store_bytes"]
