"""PQ importance sampling masker.

Uses the exact same product quantization as PQCache, but the PQ scores drive a
multinomial sample instead of a top-k, and the mask stores each selected key's
inclusion probability. Because get_masked_attention_output divides by the mask
value, the result is a weighted numerator / weighted denominator (Horvitz-
Thompson) estimate of dense attention as in vAttention, instead of a truncation
of it.

Kept as a fixed masker rather than a SamplingMasker so that it can be stacked
with AdaptiveSamplingMasker, which ResearchAttention allows only one of.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    AttentionTensorDimensions,
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask

from .pq_top_k import PQCache, PQCacheConfig

#keeps a fully masked row a valid distribution, and bounds the 1 / pi weight
_PROB_FLOOR: float = 1e-9
_MIN_INCLUSION_PROBABILITY: float = 1e-4


@dataclass
class PQImportanceConfig(PQCacheConfig):
    """Configuration for the PQImportance masker.

    Attributes:
        sample_size: Number of (or fraction of keys as) importance samples
            drawn from the keys left after the heavy_size stratum. May be 0,
            which reduces this masker to plain PQCache.
        temperature: Temperature of the proposal distribution. > 1.0 flattens
            it (more exploration, less sensitivity to PQ error), < 1.0 sharpens
            it towards top-k.
    """

    sample_size: Union[float, int]
    temperature: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        #not calling super(): TopKMaskerConfig requires heavy_size > 0, but
        #heavy_size == 0 (pure importance sampling) is valid here
        if self.heavy_size < 0:
            raise ValueError(f"heavy_size must be >= 0, got {self.heavy_size}")

        if self.sample_size < 0:
            raise ValueError(f"sample_size must be >= 0, got {self.sample_size}")

        if self.heavy_size == 0 and self.sample_size == 0:
            raise ValueError("at least one of heavy_size / sample_size must be > 0")

        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")

        if self.pq_group_factor <= 0:
            raise ValueError(f"pq_group_factor must be > 0, got {self.pq_group_factor}")

        if self.pq_bits <= 0:
            raise ValueError(f"pq_bits must be > 0, got {self.pq_bits}")

        if self.kmeans_iter <= 0:
            raise ValueError(f"kmeans_iter must be > 0, got {self.kmeans_iter}")

        if self.init_offset < 0:
            raise ValueError(f"init_offset must be >= 0, got {self.init_offset}")

        if self.metric not in ["euclidean", "ip"]:
            raise ValueError(f"metric must be 'euclidean' or 'ip', got '{self.metric}'")


@MaskerRegistry.register(PQImportanceConfig)
class PQImportance(PQCache):
    """PQ-scored importance sampling masker.

    Example:
        >>> config = PQImportanceConfig(
        ...     heavy_size=0.01, sample_size=0.02, pq_group_factor=2,
        ...     pq_bits=6, kmeans_iter=10, init_offset=128, metric="euclidean",
        ... )
        >>> masker = PQImportance(config)
    """

    def __init__(self, config: PQImportanceConfig) -> None:
        """Initialize PQ importance masker with configuration."""
        super().__init__(config)
        self.sample_size = config.sample_size
        self.temperature = config.temperature

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
        """Add a PQ importance sampling mask.

        Follows PQCache.add_mask; only the mask built from the PQ scores
        differs.
        """
        layer_idx: int = self._validate_inputs(sparse_meta_data, kwargs)

        if previous_mask.is_full_mask():
            return previous_mask

        tensor_dims: AttentionTensorDimensions = self._extract_tensor_dimensions(
            keys, queries
        )
        #budget is split across the two strata
        effective_heavy_size: int = self._calculate_effective_size(
            self.heavy_size, tensor_dims.seq_len_keys
        )
        effective_sample_size: int = self._calculate_effective_size(
            self.sample_size, tensor_dims.seq_len_keys
        )

        if self._should_use_full_attention(
            tensor_dims, effective_heavy_size + effective_sample_size
        ):
            return self._create_full_mask(
                tensor_dims, previous_mask.dtype, previous_mask.device
            )

        #quantization and scoring are inherited from PQCache unchanged
        self._initialize_pq_cache(sparse_meta_data, layer_idx)
        centroids: torch.Tensor
        codebook: torch.Tensor
        if sparse_meta_data["pq_centroids"][layer_idx] is None:
            centroids, codebook = self._perform_kmeans_clustering(
                keys, layer_idx, sparse_meta_data
            )
        else:
            centroids, codebook = self._handle_incremental_keys(
                keys, layer_idx, sparse_meta_data
            )

        scores: torch.Tensor = self._compute_pq_scores(
            queries, keys, centroids, codebook
        )

        pq_mask: Mask = self._create_importance_mask(
            dims=tensor_dims,
            scores=scores,
            num_heavy=effective_heavy_size,
            num_samples=effective_sample_size,
            previous_mask=previous_mask,
            attention_mask=attention_mask,
            scaling=scaling,
        )

        return previous_mask.merge_mask(pq_mask, inplace=False)

    def _create_importance_mask(
        self,
        dims: AttentionTensorDimensions,
        scores: torch.Tensor,
        num_heavy: int,
        num_samples: int,
        previous_mask: Mask,
        attention_mask: torch.Tensor,
        scaling: float,
    ) -> Mask:
        """Build a mask of top-k keys plus importance samples of the PQ scores.

        Mask values are inclusion probabilities: 1.0 for the top-k stratum, and
        the sampling probability for the sampled stratum.
        """
        num_scored: int = scores.shape[-1]
        key_slice: slice = slice(self.init_offset, self.init_offset + num_scored)

        #approximate logits of the true attention distribution
        logits: torch.Tensor = scores.to(torch.float32) * scaling
        if attention_mask is not None:
            logits = logits + attention_mask[:, :, :, key_slice].to(torch.float32)

        #positions already selected by earlier maskers are not candidates
        neg_inf: float = torch.finfo(logits.dtype).min
        previous_dense: torch.Tensor = previous_mask.get_dense_mask()[
            :, :, :, key_slice
        ]
        logits.masked_fill_(previous_dense != 0, neg_inf)

        row_wise_indices: List[torch.Tensor] = []
        row_wise_data: List[torch.Tensor] = []

        if num_heavy > 0:
            top_k_indices: torch.Tensor = torch.topk(
                logits, k=min(num_heavy, num_scored), dim=-1, largest=True
            ).indices
            #kept with probability 1 and removed from the sampling pool
            logits.scatter_(dim=-1, index=top_k_indices, value=neg_inf)
            row_wise_indices.append(top_k_indices)
            row_wise_data.append(
                torch.ones_like(top_k_indices, dtype=previous_mask.dtype)
            )

        if num_samples > 0:
            sampled_indices: torch.Tensor
            inclusion_probabilities: torch.Tensor
            sampled_indices, inclusion_probabilities = self._importance_sample(
                logits, num_samples
            )
            row_wise_indices.append(sampled_indices)
            row_wise_data.append(inclusion_probabilities.to(previous_mask.dtype))

        indices: torch.Tensor = torch.cat(row_wise_indices, dim=-1) + self.init_offset
        data: torch.Tensor = torch.cat(row_wise_data, dim=-1)

        mask_shape: Tuple[int, int, int, int] = (
            dims.batch_size,
            dims.num_heads,
            dims.seq_len_queries,
            dims.seq_len_keys,
        )
        #"dense" de-duplicates the repeated draws of with-replacement sampling
        return Mask.create_from_row_wise_idx(
            shape=mask_shape,
            row_wise_idx=indices,
            data=data,
            mask_type="dense",
            dtype=previous_mask.dtype,
        )

    def _importance_sample(
        self, logits: torch.Tensor, num_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw num_samples keys per row with replacement from softmax(logits).

        Returns the drawn indices and their inclusion probabilities.
        """
        batch_size, num_heads, seq_len_queries, num_scored = logits.shape

        probabilities: torch.Tensor = torch.softmax(logits / self.temperature, dim=-1)
        #a fully masked row softmaxes to nan; the floor makes it uniform instead
        probabilities = torch.nan_to_num(probabilities, nan=0.0) + _PROB_FLOOR
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)

        sampled_indices: torch.Tensor = torch.multinomial(
            probabilities.reshape(-1, num_scored), num_samples, replacement=True
        ).reshape(batch_size, num_heads, seq_len_queries, num_samples)

        sampled_probabilities: torch.Tensor = torch.gather(
            probabilities, dim=-1, index=sampled_indices
        ).clamp(min=0.0, max=1.0)

        #pi = 1 - (1 - p)^m over m draws, computed stably; p == 1 gives pi == 1
        inclusion_probabilities: torch.Tensor = -torch.expm1(
            num_samples * torch.log1p(-sampled_probabilities)
        )
        inclusion_probabilities = inclusion_probabilities.clamp(
            min=_MIN_INCLUSION_PROBABILITY, max=1.0
        )

        return sampled_indices, inclusion_probabilities

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "PQImportance":
        """Create PQImportance instance from configuration."""
        if not isinstance(config, PQImportanceConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
