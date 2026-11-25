"""Quest Top-K masker implementation."""

from dataclasses import dataclass, field
from typing import Any, Dict

import torch
try:
    from ray import tune
except ImportError:
    class _TuneStub:
        @staticmethod
        def grid_search(values):
            return values

    tune = _TuneStub()

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    AttentionTensorDimensions,
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.utils.kv_utils import (
    _get_num_key_value_groups,
    repeat_kv,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask

from ..base import TopKMasker, TopKMaskerConfig
from .utils.common_utils import pseudo_quantize
from .utils.quest_utils import (
    attention_mask_to_allowed_prob,
    compute_page_min_max,
    pages_to_token_mask,
    pages_valid,
    quest_page_scores,
)


@dataclass
class QuestTopKMaskerConfig(TopKMaskerConfig):
    """Configuration for QuestTopKMasker."""

    page_size: int
    label_bits: int = 16
    search_space: Dict[str, Any] = field(
        default_factory=lambda: {
            "heavy_size": tune.grid_search(
                [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
            ),
            "page_size": tune.grid_search([64, 128, 256]),
        }
    )

    def __post_init__(self) -> None:
        """Validate post-initialization constraints."""
        super().__post_init__()
        if not (0 < self.label_bits <= 16):
            raise ValueError(
                f"label_bits must be in range (0, 16], got {self.label_bits}"
            )
        if self.page_size <= 0:
            raise ValueError("page_size must be greater than 0")


@MaskerRegistry.register(QuestTopKMaskerConfig)
class QuestTopKMasker(TopKMasker):
    """Quest page-Top-K masker."""

    page_size: int
    label_bits: int

    def __init__(self, config: QuestTopKMaskerConfig) -> None:
        super().__init__(config)

        if config.page_size <= 0:
            raise ValueError("page_size must be a positive integer")
        self.page_size = int(config.page_size)
        self.heavy_size = config.heavy_size
        self.label_bits = config.label_bits

    def add_mask(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
        scaling: float,
        dropout: float,
        sparse_meta_data: Dict[Any, Any],
        previous_mask: Mask,
        **kwargs: Dict[str, Any],
    ) -> Mask:
        """Add Quest page-Top-K sparse mask."""
        if previous_mask.is_full_mask():
            return previous_mask

        dims: AttentionTensorDimensions = self._extract_tensor_dimensions(keys, queries)
        effective_heavy_size: int = self._calculate_effective_size(
            self.heavy_size, dims.seq_len_keys
        )

        if self._should_use_full_attention(dims, effective_heavy_size):
            return self._create_full_mask(
                dims, previous_mask.dtype, previous_mask.device
            )

        return self._create_quest_page_topk_mask(
            dims,
            effective_heavy_size,
            keys,
            queries,
            attention_mask,
            previous_mask,
        )

    def _create_quest_page_topk_mask(
        self,
        dims: AttentionTensorDimensions,
        heavy_tokens: int,
        keys: torch.Tensor,
        queries: torch.Tensor,
        attention_mask: torch.Tensor,
        previous_mask: Mask,
    ) -> Mask:
        """
        Steps:
          1) Repeat KV for GQA/MQA -> MHA.
          2) Page keys; compute per-page min/max: [B,H,P,D].
          3) Score pages with Quest bound: [B,H,Q,P].
          4) Pick Top-K pages where K_pages ~= heavy_tokens / page_size.
          5) Convert chosen pages to token-level {0,1} (float), then probabilistic-OR (max) with previous probabilities.
          6) Gate by external attention_mask probabilities (multiply).
        """
        # 1) Handle GQA/MQA
        ngroups = _get_num_key_value_groups(queries, keys)
        keys_rep = repeat_kv(keys, ngroups)  # [B,H,K,D]

        B, H, K, _ = keys_rep.shape
        _, _, Q, _ = queries.shape

        page_size = self.page_size
        num_pages = (K + page_size - 1) // page_size

        # 2) Per-page min/max (vectorized) via shared utility
        page_min, page_max = compute_page_min_max(
            keys_rep, page_size, num_pages
        )  # [B,H,P,D]

        # Quantize page_min and page_max if label_bits < 16
        if self.label_bits < 16:
            page_min = pseudo_quantize(page_min, self.label_bits)
            page_max = pseudo_quantize(page_max, self.label_bits)

        # 3) Quest page scores via shared utility
        page_scores = quest_page_scores(queries, page_min, page_max)  # [B,H,Q,P]

        # Convert external attention mask to allowed probabilities in [0,1]
        # and compute which pages are valid (have any allowed tokens).
        allowed_prob = None
        if attention_mask is not None:
            allowed_prob = attention_mask_to_allowed_prob(
                attention_mask, K
            )  # [B,1,*,K] float
            page_any_valid = pages_valid(
                allowed_prob, page_size, num_pages
            )  # [B,Q,P] bool
            page_any_valid = page_any_valid.unsqueeze(1).expand(
                B, H, Q, num_pages
            )  # [B,H,Q,P]
            # Invalidate pages with no allowed tokens by pushing scores to -inf
            page_scores = torch.where(
                page_any_valid,
                page_scores,
                torch.finfo(page_scores.dtype).min,
            )

        # 4) Select top-k pages per (B,H,Q)
        k_pages = min(num_pages, max(3, heavy_tokens // page_size))
        if k_pages <= 0:
            k_pages = 1
        topk_pages = torch.topk(
            page_scores, k=k_pages, dim=-1, largest=True
        ).indices  # [B,H,Q,Kp]

        # Previous dense mask as probabilities in [0,1] (no boolean casts)
        dense_prev = previous_mask.get_dense_mask()  # [B,H,Q,K]
        if not dense_prev.dtype.is_floating_point:
            dense_prev = dense_prev.to(page_scores.dtype)
        dense_prev = dense_prev.clamp_(0.0, 1.0)

        # 5) Build {0,1} (float) token mask from selected pages, then probabilistic-OR via max
        page_union = pages_to_token_mask(
            topk_pages, K, page_size, device=dense_prev.device, dtype=dense_prev.dtype
        )  # [B,H,Q,K], values in {0,1} (float)
        dense_mask = torch.maximum(dense_prev, page_union)

        # 6) Gate by external attention mask via multiplication with allowed probs
        if allowed_prob is not None:
            ap = allowed_prob.to(dense_mask.dtype)
            dense_mask = dense_mask * ap.expand_as(dense_mask)

        mask_shape = (B, H, Q, K)
        return Mask.create_mask_from_dense_mask(
            mask_shape, dense_mask, dtype=previous_mask.dtype
        )

    def _should_use_full_attention(
        self, dims: AttentionTensorDimensions, heavy_tokens: int
    ) -> bool:
        """Full attention if the sequence is within the token budget."""
        return dims.seq_len_keys <= max(1, heavy_tokens)

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "QuestTopKMasker":
        if not isinstance(config, QuestTopKMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
