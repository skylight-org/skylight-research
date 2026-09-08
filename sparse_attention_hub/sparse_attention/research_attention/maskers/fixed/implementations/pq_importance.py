from dataclasses import dataclass
from typing import Tuple

import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    AttentionTensorDimensions,
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.gumbel_utils import (
    sample_categorical_inclusion,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask

from .pq_top_k import PQCache, PQCacheConfig


@dataclass
class PQImportanceConfig(PQCacheConfig):
    """Configuration for the PQImportance masker.

    Attributes:
        temperature: Softmax temperature for the PQ proposal
            ``p = softmax(s / T)``. 0.0 is plain PQCache top-k with unit
            inclusion probabilities.
    """

    temperature: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")


@MaskerRegistry.register(PQImportanceConfig)
class PQImportance(PQCache):
    """PQCache that samples the heavy budget from a PQ softmax proposal.

    Draws ``m = heavy_size`` keys independently with replacement from
    ``p = softmax(s / T)``, keeps the unique set, and stores inclusion
    probabilities ``pi_i = 1 - (1 - p_i)^m``. Downstream attention applies
    Horvitz-Thompson weights ``1 / pi_i`` to exact ``q k`` logits.
    """

    def __init__(self, config: PQImportanceConfig) -> None:
        super().__init__(config)
        self.temperature = config.temperature

    def _create_pq_mask(
        self,
        dims: AttentionTensorDimensions,
        scores: torch.Tensor,
        effective_heavy_size: int,
        previous_mask: Mask,
        device: torch.device,
    ) -> Mask:
        """Write inclusion probabilities on the unique sampled set."""
        previous_dense_pq: torch.Tensor = previous_mask.get_dense_mask()[
            :, :, :, self.init_offset : self.init_offset + scores.shape[3]
        ]
        masked_scores: torch.Tensor = scores.clone()
        masked_scores[previous_dense_pq != 0] = float("-inf")

        num_scored: int = scores.shape[-1]
        if effective_heavy_size >= num_scored:
            return super()._create_pq_mask(
                dims, scores, effective_heavy_size, previous_mask, device
            )

        _indices, _inclusion, _valid, dense_pi = sample_categorical_inclusion(
            masked_scores, effective_heavy_size, self.temperature
        )
        full: torch.Tensor = torch.zeros(
            dims.batch_size,
            dims.num_heads,
            dims.seq_len_queries,
            dims.seq_len_keys,
            device=device,
            dtype=previous_mask.dtype,
        )
        end_idx: int = self.init_offset + dense_pi.shape[-1]
        full[..., self.init_offset : end_idx] = dense_pi.to(previous_mask.dtype)
        return Mask.create_mask_from_dense_mask(
            full.shape, full, dtype=previous_mask.dtype
        )

    def _select_from_scores(
        self, scores: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample k keys with replacement from the PQ proposal and return pi."""
        num_scored: int = scores.shape[-1]

        if k >= num_scored:
            indices = torch.arange(num_scored, device=scores.device).expand(
                *scores.shape[:-1], num_scored
            )
            return indices, torch.ones_like(indices, dtype=scores.dtype)

        indices, _inclusion, valid, dense = sample_categorical_inclusion(
            scores, k, self.temperature
        )
        inclusion = dense.gather(dim=-1, index=indices)
        inclusion = torch.where(valid, inclusion, torch.zeros_like(inclusion))
        return indices, inclusion.to(scores.dtype)

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "PQImportance":
        if not isinstance(config, PQImportanceConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
