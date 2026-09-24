"""ScaNN Top-K masker implementation.

ScaNN (Guo et al., "Accelerating Large-Scale Inference with Anisotropic Vector
Quantization", ICML 2020; google-research/scann) retrieves by maximum inner product in
three stages: a k-means tree, asymmetric hashing of the partition residuals, and an
optional exact rescoring pass. Its distinguishing idea is that the quantization error
that matters for MIPS is the component parallel to the datapoint, so codes are assigned
under ``eta * ||r_parallel||^2 + ||r_perpendicular||^2`` with ``eta > 1``.

Differences from google-research/scann, all deliberate:
  * Scores come from reconstructing ``c_pi(k) + z_tilde`` and one matmul rather than
    LUT16 SIMD gathers. Algebraically identical, and avoids a ``[B, H, M, Q, K]`` tensor.
  * Blocks are swept in one global order (descending mean residual norm) instead of an
    order sorted per datapoint, so every key is coordinate-descended in one kernel. The
    order is fixed when the index is built, so appended keys encode exactly as a full
    rebuild would encode them.
  * ``anisotropic_threshold`` is relative to each key's norm, which is ScaNN's behaviour
    on the unit-normalized data it is configured with. An absolute threshold makes
    ``1 - T^2/||k||^2`` non-positive for short keys, where ScaNN's ``eta`` is undefined.
  * ``anisotropic_threshold=None`` reproduces ``noise_shaping_threshold = NaN``.
  * Empty k-means clusters keep their previous center instead of being reinitialized.
  * Not implemented, and off by default in ScaNN too: AVQ centroid refit, SOAR spilling,
    upper trees, int8 reordering, PCA projection and variable-width blocks.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from ray import tune

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
from .utils.scann_utils import (
    anisotropic_eta,
    block_order,
    chunk_blocks,
    gather_rows,
    kmeans_l2,
    noise_shaped_codes,
    reconstruct,
    residual_stats,
    squared_distances,
    unit_directions,
)


@dataclass
class ScaNNTopKMaskerConfig(TopKMaskerConfig):
    """Configuration for :class:`ScaNNTopKMasker`.

    Defaults follow ScaNN's own builder: ``dimensions_per_block=2``, 16 codewords per
    block (LUT16), 12 partitioning iterations, 10 hashing iterations and ``T = 0.2``.
    ``num_leaves`` should be of the order of ``sqrt(num_keys)``.
    """

    num_leaves: int = 256
    num_leaves_to_search: int = 32
    dimensions_per_block: int = 2
    num_centers: int = 16
    anisotropic_threshold: Optional[float] = 0.2
    kmeans_iter: int = 12
    ah_kmeans_iter: int = 10
    noise_shaping_rounds: int = 10
    reorder_size: Union[float, int] = 0
    init_offset: int = 0
    seed: int = 123456789
    search_space: Dict[str, Any] = field(
        default_factory=lambda: {
            "heavy_size": tune.grid_search([0.01, 0.02, 0.05, 0.1]),
            "dimensions_per_block": tune.grid_search([2, 4]),
            "num_leaves_to_search": tune.grid_search([16, 32, 64]),
            "anisotropic_threshold": tune.grid_search([None, 0.1, 0.2, 0.3]),
        }
    )

    def __post_init__(self) -> None:
        """Validate post-initialization constraints."""
        super().__post_init__()
        if not 0 < self.num_leaves <= 32767:
            raise ValueError(f"num_leaves must be in (0, 32767], got {self.num_leaves}")
        if not 0 < self.num_leaves_to_search <= self.num_leaves:
            raise ValueError(
                "num_leaves_to_search must be in (0, num_leaves], got "
                f"{self.num_leaves_to_search}"
            )
        if self.dimensions_per_block <= 0:
            raise ValueError(
                f"dimensions_per_block must be > 0, got {self.dimensions_per_block}"
            )
        if not 1 < self.num_centers <= 256:
            raise ValueError(f"num_centers must be in (1, 256], got {self.num_centers}")
        if self.anisotropic_threshold is not None and not (
            0.0 < self.anisotropic_threshold < 1.0
        ):
            raise ValueError(
                "anisotropic_threshold must be None or in (0, 1), got "
                f"{self.anisotropic_threshold}"
            )
        if self.kmeans_iter <= 0 or self.ah_kmeans_iter <= 0:
            raise ValueError("kmeans_iter and ah_kmeans_iter must be > 0")
        if self.noise_shaping_rounds < 0:
            raise ValueError("noise_shaping_rounds must be >= 0")
        if self.reorder_size < 0:
            raise ValueError(f"reorder_size must be >= 0, got {self.reorder_size}")
        if self.init_offset < 0:
            raise ValueError(f"init_offset must be >= 0, got {self.init_offset}")


@MaskerRegistry.register(ScaNNTopKMaskerConfig)
class ScaNNTopKMasker(TopKMasker):
    """Tree + anisotropic asymmetric hashing Top-K masker."""

    def __init__(self, config: ScaNNTopKMaskerConfig) -> None:
        super().__init__(config)
        self.heavy_size = config.heavy_size
        self.num_leaves: int = int(config.num_leaves)
        self.num_leaves_to_search: int = int(config.num_leaves_to_search)
        self.dimensions_per_block: int = int(config.dimensions_per_block)
        self.num_centers: int = int(config.num_centers)
        self.anisotropic_threshold: Optional[float] = config.anisotropic_threshold
        self.kmeans_iter: int = int(config.kmeans_iter)
        self.ah_kmeans_iter: int = int(config.ah_kmeans_iter)
        self.noise_shaping_rounds: int = int(config.noise_shaping_rounds)
        self.reorder_size = config.reorder_size
        self.init_offset: int = int(config.init_offset)
        self.seed: int = int(config.seed)

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
        """Add a ScaNN Top-K sparse mask."""
        if previous_mask.is_full_mask():
            return previous_mask

        layer_idx: Any = kwargs.get("layer_idx")
        if layer_idx is None:
            raise ValueError("layer_idx must be provided in kwargs")
        if sparse_meta_data is None:
            raise ValueError("sparse_meta_data must be a dict, got None")

        dims: AttentionTensorDimensions = self._extract_tensor_dimensions(keys, queries)
        heavy_size: int = self._calculate_effective_size(
            self.heavy_size, dims.seq_len_keys
        )
        if self._should_use_full_attention(dims, heavy_size):
            return self._create_full_mask(
                dims, previous_mask.dtype, previous_mask.device
            )

        index: Dict[str, torch.Tensor] = self._update_index(
            keys, sparse_meta_data, layer_idx
        )
        return self._create_scann_mask(
            dims, heavy_size, keys, queries, attention_mask, previous_mask, index
        )

    def _should_use_full_attention(
        self, dims: AttentionTensorDimensions, heavy_size: int
    ) -> bool:
        """Full attention while the index cannot be trained or would cover the sequence.

        Both k-means stages need at least as many points as they have clusters.
        """
        clusters: int = max(self.num_leaves, self.num_centers)
        needed: int = heavy_size + self.init_offset + dims.seq_len_queries + clusters
        return dims.seq_len_keys <= needed

    def _update_index(
        self,
        keys: torch.Tensor,
        sparse_meta_data: Dict[Any, Any],
        layer_idx: Any,
    ) -> Dict[str, Any]:
        """Build the index on first use, then extend it with newly appended keys."""
        batch_size, num_kv_heads, seq_len_keys, head_dim = keys.shape
        if head_dim % self.dimensions_per_block != 0:
            raise ValueError(
                f"head_dim {head_dim} must be divisible by dimensions_per_block "
                f"{self.dimensions_per_block}"
            )

        store: Dict[Any, Dict[str, Any]] = sparse_meta_data.setdefault("scann", {})
        index: Optional[Dict[str, Any]] = store.get(layer_idx)
        cached: int = 0 if index is None else index["codes"].shape[1]
        available: int = seq_len_keys - self.init_offset
        if available < cached:
            raise ValueError(
                f"Key count shrank from {cached} to {available} for layer {layer_idx}"
            )
        if available == cached:
            return index

        points: torch.Tensor = (
            keys[:, :, self.init_offset + cached :, :]
            .reshape(batch_size * num_kv_heads, -1, head_dim)
            .float()
        )
        if index is None:
            index = self._build_index(points)
            store[layer_idx] = index
        else:
            self._extend_index(index, points)
        return index

    def _build_index(self, points: torch.Tensor) -> Dict[str, Any]:
        """Partition, residualize and quantize ``[G, N, D]`` keys.

        Leaf ids and codes are stored packed, so the resident index costs what the
        bits/token figure claims rather than eight bytes per code.
        """
        num_blocks: int = points.shape[-1] // self.dimensions_per_block
        centroids, leaves = kmeans_l2(
            points, self.num_leaves, self.kmeans_iter, self.seed
        )
        blocks: torch.Tensor = chunk_blocks(
            points - gather_rows(centroids, leaves), num_blocks
        )
        num_groups, num_points = points.shape[0], points.shape[1]
        codebooks, _ = kmeans_l2(
            blocks.transpose(1, 2).reshape(num_groups * num_blocks, num_points, -1),
            self.num_centers,
            self.ah_kmeans_iter,
            self.seed + 1,
        )
        codebooks = codebooks.view(
            num_groups, num_blocks, self.num_centers, self.dimensions_per_block
        )
        codes, order = self._encode(points, blocks, codebooks, None)
        return {
            "centroids": centroids,
            "leaves": leaves.to(torch.int16),
            "codebooks": codebooks,
            "codes": codes.to(torch.uint8),
            "order": order,
        }

    def _extend_index(self, index: Dict[str, Any], points: torch.Tensor) -> None:
        """Assign and quantize appended keys against the existing codebooks."""
        num_blocks: int = index["codebooks"].shape[1]
        leaves: torch.Tensor = squared_distances(points, index["centroids"]).argmin(-1)
        blocks: torch.Tensor = chunk_blocks(
            points - gather_rows(index["centroids"], leaves), num_blocks
        )
        codes, _ = self._encode(points, blocks, index["codebooks"], index["order"])
        index["leaves"] = torch.cat([index["leaves"], leaves.to(torch.int16)], dim=1)
        index["codes"] = torch.cat([index["codes"], codes.to(torch.uint8)], dim=1)

    def _encode(
        self,
        points: torch.Tensor,
        blocks: torch.Tensor,
        codebooks: torch.Tensor,
        order: Optional[List[int]],
    ) -> Tuple[torch.Tensor, List[int]]:
        """Noise-shaped codes for ``[G, N, D]`` keys, plus the block sweep order used.

        Reusing the build-time order keeps appended keys encoded exactly as a full
        rebuild would encode them.
        """
        directions: torch.Tensor = chunk_blocks(
            unit_directions(points), blocks.shape[2]
        )
        norms, parallel = residual_stats(blocks, directions, codebooks)
        eta: Optional[float] = (
            None
            if self.anisotropic_threshold is None
            else anisotropic_eta(self.anisotropic_threshold, points.shape[-1])
        )
        if order is None:
            order = block_order(norms)
        return (
            noise_shaped_codes(norms, parallel, eta, self.noise_shaping_rounds, order),
            order,
        )

    def _approximate_scores(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        index: Dict[str, Any],
    ) -> torch.Tensor:
        """Leaf-gated approximate inner products ``[B, H, Q, N]`` over the indexed keys.

        Keys outside the probed leaves drop to ``finfo.min``, which stays strictly above
        the ``-inf`` reserved for already-selected positions in every float dtype.
        """
        batch_size, num_kv_heads, _, head_dim = keys.shape
        groups: int = _get_num_key_value_groups(queries, keys)
        num_indexed: int = index["codes"].shape[1]

        approx: torch.Tensor = reconstruct(
            index["centroids"], index["leaves"], index["codebooks"], index["codes"]
        ).view(batch_size, num_kv_heads, num_indexed, head_dim)
        scores: torch.Tensor = torch.matmul(
            queries, repeat_kv(approx.to(queries.dtype), groups).transpose(-2, -1)
        )

        centroids: torch.Tensor = repeat_kv(
            index["centroids"]
            .view(batch_size, num_kv_heads, self.num_leaves, head_dim)
            .to(queries.dtype),
            groups,
        )
        leaf_scores: torch.Tensor = torch.matmul(queries, centroids.transpose(-2, -1))
        probed: torch.Tensor = torch.zeros_like(leaf_scores, dtype=torch.bool).scatter_(
            -1, leaf_scores.topk(self.num_leaves_to_search, dim=-1).indices, True
        )
        leaves: torch.Tensor = repeat_kv(
            index["leaves"].long().view(batch_size, num_kv_heads, num_indexed, 1),
            groups,
        ).squeeze(-1)
        gated: torch.Tensor = probed.gather(
            -1, leaves.unsqueeze(2).expand(-1, -1, queries.shape[2], -1)
        )
        return scores.masked_fill_(~gated, torch.finfo(scores.dtype).min)

    def _create_scann_mask(
        self,
        dims: AttentionTensorDimensions,
        heavy_size: int,
        keys: torch.Tensor,
        queries: torch.Tensor,
        attention_mask: torch.Tensor,
        previous_mask: Mask,
        index: Dict[str, Any],
    ) -> Mask:
        """Score, optionally rescore exactly, then take the Top-K over indexed keys."""
        num_indexed: int = index["codes"].shape[1]
        window: slice = slice(self.init_offset, self.init_offset + num_indexed)
        min_value: float = torch.finfo(queries.dtype).min

        dense: torch.Tensor = previous_mask.get_dense_mask()
        inactive: torch.Tensor = dense[..., window] == 0
        scores: torch.Tensor = self._mask_scores(
            self._approximate_scores(keys, queries, index),
            attention_mask,
            inactive,
            window,
        )

        reorder: int = self._calculate_effective_size(
            self.reorder_size, dims.seq_len_keys
        )
        if reorder > heavy_size:
            groups: int = _get_num_key_value_groups(queries, keys)
            exact: torch.Tensor = self._mask_scores(
                torch.matmul(
                    queries,
                    repeat_kv(keys[:, :, window, :], groups).transpose(-2, -1),
                ),
                attention_mask,
                inactive,
                window,
            )
            candidates: torch.Tensor = scores.topk(
                min(reorder, num_indexed), dim=-1
            ).indices
            scores = (
                torch.full_like(scores, min_value)
                .scatter_(-1, candidates, exact.gather(-1, candidates))
                .masked_fill_(~inactive, -torch.inf)
            )

        top_k: torch.Tensor = scores.topk(min(heavy_size, num_indexed), dim=-1).indices
        dense.scatter_(-1, top_k + self.init_offset, 1.0)
        return Mask.create_mask_from_dense_mask(
            dense.shape, dense, dtype=previous_mask.dtype
        )

    @staticmethod
    def _mask_scores(
        scores: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        inactive: torch.Tensor,
        window: slice,
    ) -> torch.Tensor:
        """Apply the external attention mask and drop already-selected positions.

        Already-selected positions get ``-inf`` while leaf-gated ones only get
        ``finfo.min``, so a Top-K that runs out of probed candidates falls back to
        unprobed keys rather than re-picking an active position.
        """
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, :, window]
        return scores.masked_fill_(~inactive, -torch.inf)

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "ScaNNTopKMasker":
        """Create a :class:`ScaNNTopKMasker` from its config."""
        if not isinstance(config, ScaNNTopKMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
