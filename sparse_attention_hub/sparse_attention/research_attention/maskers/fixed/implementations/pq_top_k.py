"""PQ cache top-K masker implementation."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    AttentionTensorDimensions,
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.pq_utils import (
    ip2l2_augment,
    ip2l2_augment_queries,
    kmeans_batched_pytorch,
)
from sparse_attention_hub.sparse_attention.utils.kv_utils import repeat_kv
from sparse_attention_hub.sparse_attention.utils.mask import Mask

from ..base import TopKMasker, TopKMaskerConfig


@dataclass
class PQCacheConfig(TopKMaskerConfig):
    """Configuration for PQCache masker.

    Attributes:
        heavy_size: Number of top-K tokens to select (from TopKMaskerConfig)
        pq_group_factor: Group factor for product quantization (pq_sub_dim = head_dim // pq_group_factor)
        pq_bits: Number of bits for codebook (codebook size = 2^pq_bits)
        kmeans_iter: Number of K-means iterations for clustering
        init_offset: Number of sink tokens to skip from front
        metric: Distance metric - "euclidean" or "ip" (inner product)
    """

    pq_group_factor: int
    pq_bits: int
    kmeans_iter: int
    init_offset: int
    metric: str

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        super().__post_init__()

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


@MaskerRegistry.register(PQCacheConfig)
class PQCache(TopKMasker):
    """PQ cache-based top-K masker using product quantization for approximate attention."""

    def __init__(self, config: PQCacheConfig) -> None:
        """Initialize PQ cache masker with configuration."""
        super().__init__(config)
        self.heavy_size = config.heavy_size
        self.pq_group_factor = config.pq_group_factor
        self.pq_bits = config.pq_bits
        self.kmeans_iter = config.kmeans_iter
        self.init_offset = config.init_offset
        self.metric = config.metric

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
        """Add PQ cache mask to enable PQ-based attention selection.

        Args:
            keys: Key tensor [bsz, kv_heads, seq_len_keys, head_dim]
            queries: Query tensor [bsz, n_heads, seq_len_queries, head_dim]
            values: Value tensor [bsz, kv_heads, seq_len_keys, head_dim]
            attention_mask: Optional attention mask
            scaling: Attention scaling factor
            dropout: Dropout probability
            sparse_meta_data: Dictionary for storing layer-specific PQ data
            previous_mask: Mask from previous maskers
            **kwargs: Additional arguments (must contain layer_idx)

        Returns:
            Updated mask with PQ-based top-K selection
        """
        # Phase 1: Validation and early checks
        layer_idx: int = self._validate_inputs(sparse_meta_data, kwargs)

        if previous_mask.is_full_mask():
            return previous_mask

        tensor_dims: AttentionTensorDimensions = self._extract_tensor_dimensions(
            keys, queries
        )
        effective_heavy_size: int = self._calculate_effective_size(
            self.heavy_size, tensor_dims.seq_len_keys
        )

        # Check if should use full attention
        if self._should_use_full_attention(tensor_dims, effective_heavy_size):
            return self._create_full_mask(
                tensor_dims, previous_mask.dtype, previous_mask.device
            )

        # Initialize PQ cache structure
        self._initialize_pq_cache(sparse_meta_data, layer_idx)

        # Phase 2 & 3: Setup or update quantization
        if sparse_meta_data["pq_centroids"][layer_idx] is None:
            # First time: perform K-means clustering
            centroids, codebook = self._perform_kmeans_clustering(
                keys, layer_idx, sparse_meta_data
            )
        else:
            # Subsequent calls: handle incremental keys
            centroids, codebook = self._handle_incremental_keys(
                keys, layer_idx, sparse_meta_data
            )

        # Phase 4: Compute PQ-based scores
        scores: torch.Tensor = self._compute_pq_scores(
            queries, keys, centroids, codebook
        )

        # Phase 5: Create mask from scores
        pq_mask: Mask = self._create_pq_mask(
            tensor_dims, scores, effective_heavy_size, previous_mask, keys.device
        )

        # Phase 6: Merge and return
        return previous_mask.merge_mask(pq_mask, inplace=False)

    def _validate_inputs(
        self,
        sparse_meta_data: Dict[str, Dict[int, Optional[torch.Tensor]]],
        kwargs: Dict[str, Any],
    ) -> int:
        """Validate required inputs and return layer_idx."""
        if sparse_meta_data is None:
            raise ValueError("sparse_meta_data cannot be None")

        layer_idx: Optional[int] = kwargs.get("layer_idx")
        if layer_idx is None:
            raise ValueError("layer_idx must be provided in kwargs")

        return layer_idx

    def _should_use_full_attention(
        self, dims: AttentionTensorDimensions, heavy_size: int
    ) -> bool:
        """Determine if full attention should be used."""
        total_needed: int = (
            heavy_size + self.init_offset + dims.seq_len_queries + 2**self.pq_bits
        )
        return dims.seq_len_keys <= total_needed

    def _initialize_pq_cache(
        self, sparse_meta_data: Dict[str, Any], layer_idx: int
    ) -> None:
        """Initialize sparse_meta_data structure for PQ cache."""
        if "pq_centroids" not in sparse_meta_data:
            sparse_meta_data["pq_centroids"] = {}
        if "pq_codebook" not in sparse_meta_data:
            sparse_meta_data["pq_codebook"] = {}
        if "pq_ip2l2_phi" not in sparse_meta_data:
            sparse_meta_data["pq_ip2l2_phi"] = {}

        if layer_idx not in sparse_meta_data["pq_centroids"]:
            sparse_meta_data["pq_centroids"][layer_idx] = None
        if layer_idx not in sparse_meta_data["pq_codebook"]:
            sparse_meta_data["pq_codebook"][layer_idx] = None
        if layer_idx not in sparse_meta_data["pq_ip2l2_phi"]:
            sparse_meta_data["pq_ip2l2_phi"][layer_idx] = None

    def _perform_kmeans_clustering(
        self,
        keys: torch.Tensor,
        layer_idx: int,
        sparse_meta_data: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform K-means clustering on keys and store centroids + codebook.

        Args:
            keys: [bsz, num_heads, seq_len_keys, head_dim]
            layer_idx: Current layer index
            sparse_meta_data: Dictionary to store results

        Returns:
            centroids: [bsz, num_heads, n_subvec, cent_cnt, subvec_d]
            codebook: [bsz, n_quantized_keys, num_heads, n_subvec]
        """
        bsz, num_heads, seq_len_keys, head_dim = keys.shape

        # Calculate subvector parameters
        pq_sub_dim: int = head_dim // self.pq_group_factor
        n_subvec_per_head: int = self.pq_group_factor
        subvec_d: int = pq_sub_dim
        cent_cnt: int = 2**self.pq_bits

        # Extract keys to cluster (skip init_offset from front)
        keys_to_cluster: torch.Tensor = keys[:, :, self.init_offset :, :]
        n_keys: int = keys_to_cluster.shape[2]

        # Reshape for product quantization
        # [bsz, num_heads, n_keys, head_dim] → [bsz, num_heads, n_keys, n_subvec, subvec_d]
        # → [bsz, num_heads, n_subvec, n_keys, subvec_d]
        keys_reshaped: torch.Tensor = keys_to_cluster.reshape(
            bsz, num_heads, n_keys, n_subvec_per_head, subvec_d
        ).transpose(2, 3)

        # Reshape to [bsz * num_heads * n_subvec, n_keys, subvec_d]
        keys_flat: torch.Tensor = keys_reshaped.reshape(-1, n_keys, subvec_d)

        # If using IP metric, convert to L2 via augmentation
        ip2l2_phi: Optional[torch.Tensor] = None
        if self.metric == "ip":
            keys_flat, ip2l2_phi = ip2l2_augment(keys_flat)
            subvec_d += 1  # Augmented dimension

        # Use sklearn's KMeans (CPU-based, synchronous)
        centroids: torch.Tensor
        codes: torch.Tensor
        centroids, codes = kmeans_batched_pytorch(keys_flat, cent_cnt, self.kmeans_iter)
        # Reshape outputs
        # centroids: [bsz * num_heads * n_subvec, cent_cnt, subvec_d]
        #         → [bsz, num_heads, n_subvec, cent_cnt, subvec_d]
        centroids = centroids.reshape(
            bsz, num_heads, n_subvec_per_head, cent_cnt, subvec_d
        )

        # codes: [bsz * num_heads * n_subvec, n_keys]
        #     → [bsz, n_keys, num_heads, n_subvec]
        codes = codes.reshape(bsz, num_heads, n_subvec_per_head, n_keys).permute(
            0, 3, 1, 2
        )

        # Store in sparse_meta_data (on GPU)
        sparse_meta_data["pq_centroids"][layer_idx] = centroids
        sparse_meta_data["pq_codebook"][layer_idx] = codes
        if self.metric == "ip":
            sparse_meta_data["pq_ip2l2_phi"][layer_idx] = ip2l2_phi

        return centroids, codes

    def _handle_incremental_keys(
        self,
        keys: torch.Tensor,
        layer_idx: int,
        sparse_meta_data: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Handle incremental keys for subsequent calls.

        Returns:
            centroids: Cached centroids
            codebook: Updated codebook with new keys
        """
        centroids: torch.Tensor = sparse_meta_data["pq_centroids"][layer_idx]
        cached_codebook: Optional[torch.Tensor]
        new_keys: Optional[torch.Tensor]
        cached_codebook, new_keys = self._determine_new_keys(
            keys, sparse_meta_data, layer_idx
        )

        if new_keys is not None:
            # Quantize new keys
            new_codes: torch.Tensor = self._quantize_new_keys(
                new_keys, centroids, layer_idx, sparse_meta_data
            )
            # Update codebook
            if cached_codebook is None:
                codebook: torch.Tensor = new_codes
            else:
                codebook = torch.cat([cached_codebook, new_codes], dim=1)

            sparse_meta_data["pq_codebook"][layer_idx] = codebook
        else:
            codebook = cached_codebook

        return centroids, codebook

    def _determine_new_keys(
        self,
        keys: torch.Tensor,
        sparse_meta_data: Dict[str, Any],
        layer_idx: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Determine which keys are new and need quantization.

        Returns:
            (cached_codebook, new_keys):
            - If all keys are new: (None, keys_in_quantized_region)
            - If no new keys: (codebook, None)
            - If some new keys: (codebook, new_keys_in_quantized_region)
        """
        cached_codebook: Optional[torch.Tensor] = sparse_meta_data["pq_codebook"][
            layer_idx
        ]

        bsz, kv_heads, seq_len_keys, head_dim = keys.shape

        if cached_codebook is None:
            # First time - extract all keys in quantized region
            keys_in_quantized: torch.Tensor = keys[:, :, self.init_offset :, :]
            return None, keys_in_quantized

        cached_num_keys: int = cached_codebook.shape[1]

        # Total keys that should be quantized (excluding sink only)
        current_quantized_keys: int = seq_len_keys - self.init_offset

        if current_quantized_keys < cached_num_keys:
            raise ValueError(
                f"Quantized region shrunk: {current_quantized_keys} < {cached_num_keys}"
            )
        elif current_quantized_keys > cached_num_keys:
            # Extract only new keys in the quantized region
            new_start: int = self.init_offset + cached_num_keys
            new_keys: torch.Tensor = keys[:, :, new_start:, :]
            return cached_codebook, new_keys
        else:
            # No new keys in quantized region
            return cached_codebook, None

    def _quantize_new_keys(
        self,
        new_keys: torch.Tensor,
        centroids: torch.Tensor,
        layer_idx: int,
        sparse_meta_data: Dict[str, Any],
    ) -> torch.Tensor:
        """Predict codes for new keys using existing centroids.

        Args:
            new_keys: [bsz, kv_heads, n_new_keys, head_dim]
            centroids: [bsz, kv_heads, n_subvec, cent_cnt, subvec_d]

        Returns:
            new_codes: [bsz, n_new_keys, kv_heads, n_subvec]
        """
        bsz, kv_heads, n_new, head_dim = new_keys.shape
        _, _, n_subvec, cent_cnt, subvec_d_centroids = centroids.shape

        # Calculate expected subvec_d (before augmentation)
        base_subvec_d: int = head_dim // n_subvec

        # Reshape new_keys to subvectors
        new_keys_reshaped: torch.Tensor = new_keys.reshape(
            bsz, kv_heads, n_new, n_subvec, base_subvec_d
        ).transpose(2, 3)
        # [bsz, kv_heads, n_subvec, n_new, base_subvec_d]

        # If using IP metric, augment new_keys
        if self.metric == "ip":
            ip2l2_phi: torch.Tensor = sparse_meta_data["pq_ip2l2_phi"][layer_idx]
            # Reshape for augmentation: [bsz * kv_heads * n_subvec, n_new, base_subvec_d]
            new_keys_flat: torch.Tensor = new_keys_reshaped.reshape(
                -1, n_new, base_subvec_d
            )
            # Augment
            new_keys_flat_aug: torch.Tensor = ip2l2_augment_queries(
                new_keys_flat, ip2l2_phi
            )
            # [bsz * kv_heads * n_subvec, n_new, base_subvec_d + 1]
            # Reshape back
            new_keys_reshaped = new_keys_flat_aug.reshape(
                bsz, kv_heads, n_subvec, n_new, base_subvec_d + 1
            )

        # Compute distances to centroids
        # new_keys: [bsz, kv_heads, n_subvec, n_new, 1, subvec_d]
        # centroids: [bsz, kv_heads, n_subvec, 1, cent_cnt, subvec_d]
        new_keys_exp: torch.Tensor = new_keys_reshaped.unsqueeze(4)
        centroids_exp: torch.Tensor = centroids.unsqueeze(3)

        # Euclidean distance (works for both metrics after augmentation)
        distances: torch.Tensor = torch.sum((new_keys_exp - centroids_exp) ** 2, dim=-1)
        # [bsz, kv_heads, n_subvec, n_new, cent_cnt]

        # Get nearest centroid
        new_codes: torch.Tensor = torch.argmin(distances, dim=-1)
        # [bsz, kv_heads, n_subvec, n_new]
        new_codes = new_codes.permute(0, 3, 1, 2)
        # [bsz, n_new, kv_heads, n_subvec]

        return new_codes

    def _compute_pq_scores(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        centroids: torch.Tensor,
        codebook: torch.Tensor,
    ) -> torch.Tensor:
        """Compute approximate attention scores using PQ.

        Args:
            queries: [bsz, n_heads, seq_len_queries, head_dim]
            keys: [bsz, kv_heads, seq_len_keys, head_dim]
            centroids: [bsz, kv_heads, n_subvec, cent_cnt, subvec_d (or subvec_d+1 if augmented)]
            codebook: [bsz, n_clustered_keys, kv_heads, n_subvec]

        Returns:
            scores: [bsz, n_heads, seq_len_queries, n_clustered_keys]
        """
        bsz, n_heads, seq_len_q, head_dim = queries.shape
        _, kv_heads, seq_len_k, _ = keys.shape
        _, n_clustered, _, n_subvec = codebook.shape
        cent_cnt: int = centroids.shape[3]

        # Calculate GQA repeat factor
        num_key_value_groups: int = n_heads // kv_heads

        # Calculate subvec_d
        subvec_d: int = head_dim // n_subvec

        # Reshape queries: [bsz, n_heads, seq_len_q, n_subvec, subvec_d]
        queries_reshaped: torch.Tensor = queries.reshape(
            bsz, n_heads, seq_len_q, n_subvec, subvec_d
        )
        # → [bsz, n_heads, n_subvec, seq_len_q, subvec_d]
        queries_trans: torch.Tensor = queries_reshaped.transpose(2, 3)

        # Repeat centroids for GQA: [bsz, kv_heads, ...] → [bsz, n_heads, ...]
        # Note: centroids may be augmented if IP metric was used during clustering,
        # but we only use the first subvec_d dimensions for scoring
        # centroids: [bsz, kv_heads, n_subvec, cent_cnt, subvec_d_stored]
        if num_key_value_groups == 1:
            repeat_centroids: torch.Tensor = centroids
        else:
            # Manually repeat along head dimension for 5D tensor
            repeat_centroids = (
                centroids[:, :, None, :, :, :]
                .expand(bsz, kv_heads, num_key_value_groups, n_subvec, cent_cnt, -1)
                .reshape(bsz, kv_heads * num_key_value_groups, n_subvec, cent_cnt, -1)
            )
        # [bsz, n_heads, n_subvec, cent_cnt, subvec_d_stored]

        # Extract only the original dimensions (ignore augmented dimension if present)
        repeat_centroids = repeat_centroids[..., :subvec_d]
        # [bsz, n_heads, n_subvec, cent_cnt, subvec_d]

        # Transpose for matmul: [bsz, n_heads, n_subvec, subvec_d, cent_cnt]
        repeat_centroids = repeat_centroids.transpose(3, 4)

        # Compute Q @ Centroids.T (inner product scores)
        qk_table: torch.Tensor = torch.matmul(queries_trans, repeat_centroids)
        # [bsz, n_heads, n_subvec, seq_len_q, cent_cnt]

        # Repeat codebook for GQA
        repeat_codebook: torch.Tensor = repeat_kv(
            codebook.permute(0, 2, 3, 1), num_key_value_groups
        )
        # [bsz, n_heads, n_subvec, n_clustered]

        # Gather scores using codebook indices
        # Expand codebook for all queries: [bsz, n_heads, n_subvec, 1, n_clustered]
        repeat_codebook_exp: torch.Tensor = repeat_codebook.unsqueeze(3).expand(
            -1, -1, -1, seq_len_q, -1
        )

        # Gather: for each query, get score to each key's assigned centroid
        gathered_scores: torch.Tensor = torch.gather(
            qk_table, dim=4, index=repeat_codebook_exp
        )
        # [bsz, n_heads, n_subvec, seq_len_q, n_clustered]

        # Sum across subvectors
        scores: torch.Tensor = gathered_scores.sum(dim=2)
        # [bsz, n_heads, seq_len_q, n_clustered]

        return scores

    def _create_pq_mask(
        self,
        dims: AttentionTensorDimensions,
        scores: torch.Tensor,
        effective_heavy_size: int,
        previous_mask: Mask,
        device: torch.device,
    ) -> Mask:
        """Create mask from PQ scores.

        Args:
            dims: Attention tensor dimensions
            scores: [bsz, n_heads, seq_len_q, n_clustered]
            effective_heavy_size: Number of top-K to select
            previous_mask: Previous mask to check already-active positions
            device: Device for tensors

        Returns:
            Mask with top-K positions selected
        """
        # Mask out previous positions
        previous_dense: torch.Tensor = previous_mask.get_dense_mask()
        # Extract relevant portion for quantized keys
        previous_dense_pq: torch.Tensor = previous_dense[
            :, :, :, self.init_offset : self.init_offset + scores.shape[3]
        ]

        # Mask out already-active positions
        masked_scores: torch.Tensor = scores.clone()
        masked_scores[previous_dense_pq != 0] = torch.finfo(scores.dtype).min

        # Select top-K indices
        _, topk_indices = torch.topk(
            masked_scores, k=effective_heavy_size, dim=-1, largest=True
        )

        topk_indices_adjusted: torch.Tensor = topk_indices + self.init_offset
        # remove duplicates within each row
        topk_indices_adjusted = torch.sort(topk_indices_adjusted, dim=-1).values
        topk_indices_adjusted = torch.unique_consecutive(topk_indices_adjusted, dim=-1)
        # Create mask from indices
        return self._create_mask_from_rowise_indices(
            dims,
            topk_indices_adjusted,
            device,
            previous_mask.dtype,
        )

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "PQCache":
        """Create PQCache instance from configuration."""
        if not isinstance(config, PQCacheConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
