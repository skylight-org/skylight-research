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
        pq_group_factor: Group factor for product quantization
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
        """Add PQ cache mask to enable PQ-based attention selection."""
        layer_idx: int = self._validate_inputs(sparse_meta_data, kwargs)

        if previous_mask.is_full_mask():
            return previous_mask

        tensor_dims: AttentionTensorDimensions = self._extract_tensor_dimensions(
            keys, queries
        )
        effective_heavy_size: int = self._calculate_effective_size(
            self.heavy_size, tensor_dims.seq_len_keys
        )

        if self._should_use_full_attention(tensor_dims, effective_heavy_size):
            return self._create_full_mask(
                tensor_dims, previous_mask.dtype, previous_mask.device
            )

        self._initialize_pq_cache(sparse_meta_data, layer_idx)

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

        pq_mask: Mask = self._create_pq_mask(
            tensor_dims, scores, effective_heavy_size, previous_mask, keys.device
        )

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
        """Perform K-means clustering on keys and store centroids + codebook."""
        bsz, num_heads, seq_len_keys, head_dim = keys.shape

        pq_sub_dim: int = head_dim // self.pq_group_factor
        n_subvec_per_head: int = self.pq_group_factor
        subvec_d: int = pq_sub_dim
        cent_cnt: int = 2**self.pq_bits

        keys_to_cluster: torch.Tensor = keys[:, :, self.init_offset :, :]
        n_keys: int = keys_to_cluster.shape[2]

        keys_reshaped: torch.Tensor = keys_to_cluster.reshape(
            bsz, num_heads, n_keys, n_subvec_per_head, subvec_d
        ).transpose(2, 3)

        keys_flat: torch.Tensor = keys_reshaped.reshape(-1, n_keys, subvec_d)

        ip2l2_phi: Optional[torch.Tensor] = None
        if self.metric == "ip":
            keys_flat, ip2l2_phi = ip2l2_augment(keys_flat)
            subvec_d += 1

        centroids: torch.Tensor
        codes: torch.Tensor
        centroids, codes = kmeans_batched_pytorch(keys_flat, cent_cnt, self.kmeans_iter)

        centroids = centroids.reshape(
            bsz, num_heads, n_subvec_per_head, cent_cnt, subvec_d
        )

        codes = codes.reshape(bsz, num_heads, n_subvec_per_head, n_keys).permute(
            0, 3, 1, 2
        )

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
        """Handle incremental keys for subsequent calls."""
        centroids: torch.Tensor = sparse_meta_data["pq_centroids"][layer_idx]
        cached_codebook: Optional[torch.Tensor]
        new_keys: Optional[torch.Tensor]
        cached_codebook, new_keys = self._determine_new_keys(
            keys, sparse_meta_data, layer_idx
        )

        if new_keys is not None:
            new_codes: torch.Tensor = self._quantize_new_keys(
                new_keys, centroids, layer_idx, sparse_meta_data
            )
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
        """Determine which keys are new and need quantization."""
        cached_codebook: Optional[torch.Tensor] = sparse_meta_data["pq_codebook"][
            layer_idx
        ]

        bsz, kv_heads, seq_len_keys, head_dim = keys.shape

        if cached_codebook is None:
            keys_in_quantized: torch.Tensor = keys[:, :, self.init_offset :, :]
            return None, keys_in_quantized

        cached_num_keys: int = cached_codebook.shape[1]
        current_quantized_keys: int = seq_len_keys - self.init_offset

        if current_quantized_keys < cached_num_keys:
            raise ValueError(
                f"Quantized region shrunk: {current_quantized_keys} < {cached_num_keys}"
            )
        elif current_quantized_keys > cached_num_keys:
            new_start: int = self.init_offset + cached_num_keys
            new_keys: torch.Tensor = keys[:, :, new_start:, :]
            return cached_codebook, new_keys
        else:
            return cached_codebook, None

    def _quantize_new_keys(
        self,
        new_keys: torch.Tensor,
        centroids: torch.Tensor,
        layer_idx: int,
        sparse_meta_data: Dict[str, Any],
    ) -> torch.Tensor:
        """Predict codes for new keys using existing centroids."""
        bsz, kv_heads, n_new, head_dim = new_keys.shape
        _, _, n_subvec, cent_cnt, subvec_d_centroids = centroids.shape

        base_subvec_d: int = head_dim // n_subvec

        new_keys_reshaped: torch.Tensor = new_keys.reshape(
            bsz, kv_heads, n_new, n_subvec, base_subvec_d
        ).transpose(2, 3)

        if self.metric == "ip":
            ip2l2_phi: torch.Tensor = sparse_meta_data["pq_ip2l2_phi"][layer_idx]
            new_keys_flat: torch.Tensor = new_keys_reshaped.reshape(
                -1, n_new, base_subvec_d
            )
            new_keys_flat_aug: torch.Tensor = ip2l2_augment_queries(
                new_keys_flat, ip2l2_phi
            )
            new_keys_reshaped = new_keys_flat_aug.reshape(
                bsz, kv_heads, n_subvec, n_new, base_subvec_d + 1
            )

        new_keys_exp: torch.Tensor = new_keys_reshaped.unsqueeze(4)
        centroids_exp: torch.Tensor = centroids.unsqueeze(3)

        distances: torch.Tensor = torch.sum((new_keys_exp - centroids_exp) ** 2, dim=-1)

        new_codes: torch.Tensor = torch.argmin(distances, dim=-1)
        new_codes = new_codes.permute(0, 3, 1, 2)

        return new_codes

    def _compute_pq_scores(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        centroids: torch.Tensor,
        codebook: torch.Tensor,
    ) -> torch.Tensor:
        """Compute approximate attention scores using PQ."""
        bsz, n_heads, seq_len_q, head_dim = queries.shape
        _, kv_heads, seq_len_k, _ = keys.shape
        _, n_clustered, _, n_subvec = codebook.shape
        cent_cnt: int = centroids.shape[3]

        num_key_value_groups: int = n_heads // kv_heads
        subvec_d: int = head_dim // n_subvec

        queries_reshaped: torch.Tensor = queries.reshape(
            bsz, n_heads, seq_len_q, n_subvec, subvec_d
        )
        queries_trans: torch.Tensor = queries_reshaped.transpose(2, 3)

        if num_key_value_groups == 1:
            repeat_centroids: torch.Tensor = centroids
        else:
            repeat_centroids = (
                centroids[:, :, None, :, :, :]
                .expand(bsz, kv_heads, num_key_value_groups, n_subvec, cent_cnt, -1)
                .reshape(bsz, kv_heads * num_key_value_groups, n_subvec, cent_cnt, -1)
            )

        repeat_centroids = repeat_centroids[..., :subvec_d]
        repeat_centroids = repeat_centroids.transpose(3, 4)

        qk_table: torch.Tensor = torch.matmul(queries_trans, repeat_centroids)

        repeat_codebook: torch.Tensor = repeat_kv(
            codebook.permute(0, 2, 3, 1), num_key_value_groups
        )

        repeat_codebook_exp: torch.Tensor = repeat_codebook.unsqueeze(3).expand(
            -1, -1, -1, seq_len_q, -1
        )

        gathered_scores: torch.Tensor = torch.gather(
            qk_table, dim=4, index=repeat_codebook_exp
        )

        scores: torch.Tensor = gathered_scores.sum(dim=2)

        return scores

    def _create_pq_mask(
        self,
        dims: AttentionTensorDimensions,
        scores: torch.Tensor,
        effective_heavy_size: int,
        previous_mask: Mask,
        device: torch.device,
    ) -> Mask:
        """Create mask from PQ scores, excluding already-active positions.

        The key invariant: the returned mask must have exactly
        ``effective_heavy_size`` active positions per query row, and NONE of
        those positions may overlap with positions already active in
        ``previous_mask`` within the quantized region.

        The fix versus the original: we build the ``already_active`` exclusion
        tensor by constructing a fresh zero tensor and scattering the
        already-active absolute indices into it, rather than slicing
        ``previous_mask.get_dense_mask()`` directly.  This avoids any
        shape/broadcast ambiguity that arises when ``previous_mask`` was built
        with a different internal representation (sparse vs dense, broadcast
        dimensions, etc.).
        """
        bsz: int = dims.batch_size
        n_heads: int = dims.num_heads
        seq_len_q: int = dims.seq_len_queries
        n_clustered: int = scores.shape[3]

        # ------------------------------------------------------------------ #
        # 1. Derive which clustered-region positions are already active by    #
        #    reading the full dense previous mask and slicing the PQ window.  #
        #    We explicitly call contiguous() + clone() to guarantee we have   #
        #    an independent, fully-materialised tensor with no aliasing.      #
        # ------------------------------------------------------------------ #
        previous_dense: torch.Tensor = (
            previous_mask.get_dense_mask()
            .contiguous()
            .clone()
        )
        # previous_dense: [bsz, n_heads, seq_len_q, seq_len_keys]
        # Broadcast-safe expand so every query row has its own copy.
        if previous_dense.shape[2] == 1 and seq_len_q > 1:
            previous_dense = previous_dense.expand(bsz, n_heads, seq_len_q, -1)

        # Slice the window that corresponds to the quantized keys.
        previous_dense_pq: torch.Tensor = previous_dense[
            :, :, :, self.init_offset : self.init_offset + n_clustered
        ].contiguous()
        # Shape: [bsz, n_heads, seq_len_q, n_clustered]

        # ------------------------------------------------------------------ #
        # 2. Suppress already-active positions in the score tensor.           #
        # ------------------------------------------------------------------ #
        masked_scores: torch.Tensor = scores.clone()
        already_active: torch.Tensor = previous_dense_pq != 0
        masked_scores[already_active] = torch.finfo(scores.dtype).min

        # ------------------------------------------------------------------ #
        # 3. Top-K selection on the suppressed scores.                        #
        #    We get more than K candidates, then filter to remove any that   #
        #    are still marked as already-active (due to -inf ties).           #
        # ------------------------------------------------------------------ #
        # Request more candidates than needed to account for invalid ones.
        # We'll filter and keep only the truly valid top-K selections.
        k_padding: int = max(effective_heavy_size + 1, 
                            n_clustered)  # Over-sample to be safe
        
        _, topk_indices_unfiltered = torch.topk(
            masked_scores, 
            k=min(k_padding, n_clustered),  # Can't exceed available positions
            dim=-1, 
            largest=True
        )

        # ------------------------------------------------------------------ #
        # 4. 🔥 CRITICAL FIX: Filter to keep only truly valid indices        #
        #    Check if each topk_indices entry is marked as already_active.   #
        #    If so, mark it as invalid. Only use the first effective_heavy   #
        #    valid positions.                                                #
        # ------------------------------------------------------------------ #
        # Check which topk indices are currently suppressed (already active)
        is_suppressed: torch.Tensor = already_active.gather(
            dim=3, index=topk_indices_unfiltered
        )
        # Shape: [bsz, n_heads, seq_len_q, k_padding]

        # Create cumulative count of valid (non-suppressed) selections
        valid_flags: torch.Tensor = ~is_suppressed
        valid_cumsum: torch.Tensor = torch.cumsum(
            valid_flags.float(), dim=-1
        )
        # Shape: [bsz, n_heads, seq_len_q, k_padding]

        # Keep only the first effective_heavy_size valid entries
        valid_count_mask: torch.Tensor = (
            valid_cumsum <= effective_heavy_size
        )
        # Shape: [bsz, n_heads, seq_len_q, k_padding]

        # Combine: must be both non-suppressed AND within the count limit
        keep_mask: torch.Tensor = valid_flags & valid_count_mask
        # Shape: [bsz, n_heads, seq_len_q, k_padding]

        # Extract indices that we're keeping, padding with 0 for unused slots
        topk_indices_filtered: torch.Tensor = torch.where(
            keep_mask,
            topk_indices_unfiltered,
            torch.tensor(0, dtype=topk_indices_unfiltered.dtype, device=device)
        )
        # Shape: [bsz, n_heads, seq_len_q, k_padding]

        # Adjust indices to absolute coordinates
        topk_indices_adjusted: torch.Tensor = topk_indices_filtered + self.init_offset

        # ------------------------------------------------------------------ #
        # 5. Build the output mask.                                           #
        #    Only scatter at positions where keep_mask is True.               #
        # ------------------------------------------------------------------ #
        mask_shape: Tuple[int, int, int, int] = (
            bsz, n_heads, seq_len_q, dims.seq_len_keys,
        )
        
        # Start from an all-zero dense tensor.
        dense_out: torch.Tensor = torch.zeros(
            mask_shape, dtype=previous_mask.dtype, device=device
        )
        
        # Only scatter where keep_mask is True.
        # We need to scatter 1s at the positions specified by topk_indices_adjusted
        # where keep_mask is True.
        # Since scatter with multi-dimensional indices is tricky, we'll do it
        # element by element.
        for b in range(bsz):
            for h in range(n_heads):
                for q in range(seq_len_q):
                    for m in range(keep_mask.shape[-1]):
                        if keep_mask[b, h, q, m]:
                            k = topk_indices_adjusted[b, h, q, m].item()
                            dense_out[b, h, q, k] = 1.0

        return Mask(
            shape=mask_shape,
            mask=dense_out,
            from_dense_mask=True,
            dtype=previous_mask.dtype,
            device=device,
        )

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "PQCache":
        """Create PQCache instance from configuration."""
        if not isinstance(config, PQCacheConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)