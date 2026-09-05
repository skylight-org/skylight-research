"""PQ sampling masker: PQCache with Gumbel noise added to the scores.

PQCache picks the ``heavy_size`` keys with the largest PQ scores. This masker
perturbs those scores with Gumbel(0, 1) noise before the top-k, which turns the
selection into a sample of ``heavy_size`` distinct keys drawn without
replacement from ``softmax(scores / temperature)`` (the Gumbel-top-k trick).
Everything else -- quantization, scoring, budget, mask construction -- is
inherited from PQCache unchanged.

``temperature -> 0`` recovers PQCache exactly; larger values explore further
down the score ranking, which is what makes the selection robust to PQ
approximation error.
"""

from dataclasses import dataclass

import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    MaskerConfig,
    MaskerRegistry,
)

from .pq_top_k import PQCache, PQCacheConfig

# keeps -log(-log(u)) finite at both ends of the uniform sample
_UNIFORM_EPS: float = 1e-6


@dataclass
class PQImportanceConfig(PQCacheConfig):
    """Configuration for the PQImportance masker.

    Attributes:
        temperature: Scale of the Gumbel noise added to the PQ scores, i.e. the
            temperature of the distribution being sampled from. 0.0 is plain
            PQCache top-k; larger values sample further down the ranking.
    """

    temperature: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        super().__post_init__()

        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")


@MaskerRegistry.register(PQImportanceConfig)
class PQImportance(PQCache):
    """PQCache whose top-k is a Gumbel sample rather than a deterministic pick.

    Example:
        >>> config = PQImportanceConfig(
        ...     heavy_size=0.02, temperature=1.0, pq_group_factor=2,
        ...     pq_bits=6, kmeans_iter=10, init_offset=128, metric="euclidean",
        ... )
        >>> masker = PQImportance(config)
    """

    def __init__(self, config: PQImportanceConfig) -> None:
        """Initialize PQ sampling masker with configuration."""
        super().__init__(config)
        self.temperature = config.temperature

    def _compute_pq_scores(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        centroids: torch.Tensor,
        codebook: torch.Tensor,
    ) -> torch.Tensor:
        """Compute PQ scores and perturb them with Gumbel noise.

        The noise is added here, before PQCache masks out already-selected
        positions, so it can never revive a key another masker has taken.
        """
        scores: torch.Tensor = super()._compute_pq_scores(
            queries, keys, centroids, codebook
        )
        return scores + self.temperature * _sample_gumbel_noise(scores)

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "PQImportance":
        """Create PQImportance instance from configuration."""
        if not isinstance(config, PQImportanceConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)


def _sample_gumbel_noise(reference: torch.Tensor) -> torch.Tensor:
    """Draw standard Gumbel(0, 1) noise shaped and typed like ``reference``.

    Sampled in float32 regardless of the score dtype: the double logarithm is
    too lossy in half precision.
    """
    uniform: torch.Tensor = torch.rand(
        reference.shape, device=reference.device, dtype=torch.float32
    ).clamp_(min=_UNIFORM_EPS, max=1.0 - _UNIFORM_EPS)
    return (-torch.log(-torch.log(uniform))).to(reference.dtype)
