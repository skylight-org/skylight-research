from dataclasses import dataclass
from typing import Tuple

import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    MaskerConfig,
    MaskerRegistry,
)

from .pq_top_k import PQCache, PQCacheConfig

# keeps -log(-log(u)) finite at both ends of the uniform sample
_UNIFORM_EPS: float = 1e-6
# floor on pi_i so the 1/pi_i weight cannot blow up on a near-zero probability
_MIN_INCLUSION_PROBABILITY: float = 1e-4


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

    Selection is Gumbel-top-k on the PQ scores, i.e. sampling without
    replacement from p proportional to exp(score / temperature). Each selected
    key carries a Horvitz-Thompson weight so the sparse attention sum stays
    unbiased.
    """

    def __init__(self, config: PQImportanceConfig) -> None:
        super().__init__(config)
        self.temperature = config.temperature

    def _select_from_scores(
        self, scores: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample k keys from the PQ scores and return their mask weights.

        Args:
            scores: [..., num_scored] PQ scores, already masked (-inf) for
                positions taken by earlier maskers.
            k: number of keys to select.

        Returns:
            indices: [..., k] selected positions.
            weights: [..., k] values to write into the mask.
        """
        num_scored: int = scores.shape[-1]

        # k+1 does not exist: the budget covers everything, so keep it all.
        if k >= num_scored:
            indices = torch.arange(num_scored, device=scores.device).expand(
                *scores.shape[:-1], num_scored
            )
            return indices, torch.ones_like(indices, dtype=scores.dtype)

        # temperature 0 is deterministic top-k, every pi_i is 1.
        if self.temperature == 0.0:
            indices = torch.topk(scores, k=k, dim=-1).indices
            return indices, torch.ones_like(indices, dtype=scores.dtype)

        logits: torch.Tensor = scores.to(torch.float32)
        perturbed: torch.Tensor = logits + self.temperature * _sample_gumbel_noise(
            logits
        )

        top_values, top_indices = torch.topk(perturbed, k=k + 1, dim=-1, sorted=True)
        indices: torch.Tensor = top_indices[..., :k]
        threshold: torch.Tensor = top_values[..., k : k + 1]

        sampled_logits: torch.Tensor = torch.gather(logits, dim=-1, index=indices)

        # pi_i = P(s_i + T * G > threshold) = 1 - exp(-exp((s_i - threshold) / T)).
        # The softmax normalizer cancels: threshold is on the same scale as s_i.
        exponent: torch.Tensor = (sampled_logits - threshold) / self.temperature
        inclusion_probabilities: torch.Tensor = -torch.expm1(-torch.exp(exponent))

        # A -inf threshold means fewer than k+1 live positions; those rows are
        # fully selected, so pi_i = 1.
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


def _sample_gumbel_noise(reference: torch.Tensor) -> torch.Tensor:
    """Draw standard Gumbel(0, 1) noise shaped like ``reference``, in float32."""
    uniform: torch.Tensor = torch.rand(
        reference.shape, device=reference.device, dtype=torch.float32
    ).clamp_(min=_UNIFORM_EPS, max=1.0 - _UNIFORM_EPS)
    return -torch.log(-torch.log(uniform))
