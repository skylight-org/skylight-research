from dataclasses import dataclass
from typing import Tuple

import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.gumbel_utils import (
    sample_gumbel_noise,
)

from .pq_top_k import PQCache, PQCacheConfig

_UNIFORM_EPS: float = 1e-6
_MIN_INCLUSION_PROBABILITY: float = 1e-4

# existing tests import this name
_sample_gumbel_noise = sample_gumbel_noise


@dataclass
class PQImportanceConfig(PQCacheConfig):
    """Configuration for the PQImportance masker.

    Attributes:
        temperature: Scale of the Gumbel noise added to the PQ scores. 0.0 is
            plain PQCache top-k with unit weights; larger values sample
            further down the ranking.
    """

    temperature: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")


@MaskerRegistry.register(PQImportanceConfig)
class PQImportance(PQCache):
    """PQCache whose top-k is a Gumbel sample, weighted by 1 / inclusion prob.

    The vAttention production path is PQCache (deterministic top-k) plus
    AdaptiveSampling (Gumbel on leftovers). This class Gumbel-samples the
    heavy budget itself and is kept for isolated experiments.
    """

    def __init__(self, config: PQImportanceConfig) -> None:
        super().__init__(config)
        self.temperature = config.temperature

    def _select_from_scores(
        self, scores: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample k keys from the PQ scores and return their mask weights."""
        num_scored: int = scores.shape[-1]

        if k >= num_scored:
            indices = torch.arange(num_scored, device=scores.device).expand(
                *scores.shape[:-1], num_scored
            )
            return indices, torch.ones_like(indices, dtype=scores.dtype)

        if self.temperature == 0.0:
            indices = torch.topk(scores, k=k, dim=-1).indices
            return indices, torch.ones_like(indices, dtype=scores.dtype)

        logits: torch.Tensor = scores.to(torch.float32)
        perturbed: torch.Tensor = logits + self.temperature * sample_gumbel_noise(
            logits
        )

        top_values, top_indices = torch.topk(perturbed, k=k + 1, dim=-1, sorted=True)
        indices = top_indices[..., :k]
        threshold: torch.Tensor = top_values[..., k : k + 1]
        sampled_logits: torch.Tensor = torch.gather(logits, dim=-1, index=indices)

        exponent: torch.Tensor = (sampled_logits - threshold) / self.temperature
        inclusion_probabilities: torch.Tensor = -torch.expm1(-torch.exp(exponent))
        inclusion_probabilities = torch.where(
            torch.isfinite(exponent),
            inclusion_probabilities,
            torch.ones_like(inclusion_probabilities),
        ).clamp(min=_MIN_INCLUSION_PROBABILITY, max=1.0)

        weights: torch.Tensor = 1.0 / inclusion_probabilities
        return indices, weights.to(scores.dtype)

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "PQImportance":
        if not isinstance(config, PQImportanceConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
