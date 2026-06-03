"""Magic pig sampling masker implementation.

This module provides a MagicPig sampling masker that uses Locality Sensitive Hashing (LSH)
to efficiently approximate maximum inner product search for attention computation. It combines
LSH-based similarity matching with probability-based sampling to create sparse attention patterns.

"""

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple

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
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.utils.kv_utils import (
    _get_num_key_value_groups,
    repeat_kv,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask

from ..base import SamplingMasker, SamplingMaskerConfig


@dataclass
class MagicPigConfig(SamplingMaskerConfig):
    """Configuration for MagicPig masker.

    This configuration class inherits from SamplingMaskerConfig and adds LSH-specific
    parameters.

    Attributes:
        lsh_l: Number of LSH tables. Higher values increase the chance of finding
            similar items but also increase computational cost.
        lsh_k: Number of bits per LSH table. Higher values make each table more
            precise but reduce the probability of collisions.
        center: bool, optional: If True, centers the keys and queries before LSH
            by subtracting the mean of the keys.
        packing: Literal["int64", "float32"]: The packing strategy for signatures.
             'int64' is more memory and compute-efficient.
        seed: int, optional: Random seed for reproducible LSH projections. Defaults to 42.
    """

    lsh_l: int  # number of LSH tables
    lsh_k: int  # number of bits per LSH table
    center: bool = True  # whether to center keys and queries before LSH
    packing: Literal["int64", "float32"] = "int64"  # packing strategy for signatures
    seed: Optional[int] = 42  # random seed for reproducible projections
    search_space: Dict[str, Any] = field(
        default_factory=lambda: {
            "lsh_l": tune.grid_search([16, 32, 64, 128]),
            "lsh_k": tune.grid_search([4, 8, 16, 32]),
        }
    )

    def __post_init__(self) -> None:
        """Validate LSH parameters after initialization."""
        if self.lsh_l <= 0:
            raise ValueError(f"lsh_l must be positive, got {self.lsh_l}")
        if self.lsh_k <= 0:
            raise ValueError(f"lsh_k must be positive, got {self.lsh_k}")
        if self.packing not in ["int64", "float32"]:
            raise ValueError(
                f"packing must be 'int64' or 'float32', got {self.packing}"
            )
        if self.packing == "int64" and self.lsh_k > 64:
            raise ValueError(
                f"For 'int64' packing, lsh_k must be <= 64, but got {self.lsh_k}"
            )
        if self.seed is None:
            raise ValueError("seed cannot be None")


@MaskerRegistry.register(MagicPigConfig)
class MagicPig(SamplingMasker):
    """Magic Pig masker using Locality Sensitive Hashing for attention computation.

    This masker implements LSH-based sampling of attention positions by using
    locality sensitive hashing to identify similar key-query pairs. The sampling
    is based on both LSH collision probability and actual LSH matches.

    Attributes:
        lsh_l: Number of LSH tables used for hashing.
        lsh_k: Number of bits per LSH table.
        center: Whether to center keys and queries before LSH (default is True).
        packing: Packing strategy for signatures, either 'int64' or 'float32'.

    Important Notes:
        - Uses the Neyshabur & Srebro technique to transform inner product search to cosine similarity
        - Computes theoretical LSH collision probabilities based on cosine similarity
        - Uses random signed projections for efficient LSH implementation
        - Ignores the sampling_rate parameter from the parent class
        - Merge operation adds the data in masks and clamps to 1.0

    Example:
        >>> config = MagicPigConfig(lsh_l=5, lsh_k=4)
        >>> masker = MagicPig(config)
        >>> # Use masker.add_mask() to apply LSH-based sampling to attention masks
    """

    def __init__(self, config: MagicPigConfig) -> None:
        """Initialize Magic Pig masker with configuration.

        Args:
            config: Configuration object containing LSH parameters.

        Raises:
            ValueError: If LSH parameters are invalid.
        """
        super().__init__(config)
        self.lsh_l = config.lsh_l
        self.lsh_k = config.lsh_k
        self.center = config.center
        self.packing = config.packing
        self.seed = config.seed

    def _pack_bits(self, signatures: torch.Tensor) -> torch.Tensor:
        """Packs binary signatures into int64 tensors."""
        binary_signatures = (signatures > 0).to(torch.int64)
        reshaped_signatures = binary_signatures.view(
            *signatures.shape[:-1], self.lsh_l, self.lsh_k
        )
        packer = 2 ** torch.arange(
            self.lsh_k, device=signatures.device, dtype=torch.int64
        )
        packed_signatures = (reshaped_signatures * packer).sum(dim=-1)
        return packed_signatures

    def _compute_signatures(
        self, vectors: torch.Tensor, sparse_meta_data: Dict[Any, Any], layer_idx: int
    ) -> torch.Tensor:
        """Computes signatures for given vectors, with optional packing."""
        total_bits: int = self.lsh_l * self.lsh_k
        batch_size, num_heads, seq_len, head_dim = vectors.shape
        vectors_flat: torch.Tensor = vectors.view(-1, head_dim)
        projection = sparse_meta_data["projections"][layer_idx]

        # 1. Compute signs (+1/-1)
        signs = torch.sign(torch.matmul(vectors_flat, projection))
        signs = signs.view(batch_size, num_heads, seq_len, total_bits)

        # 2. Pack if using int64 method
        if self.packing == "int64":
            return self._pack_bits(signs)

        # Otherwise, return signs as floats
        return signs.float()

    def _center_KQ(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        sparse_meta_data: Dict[Any, Any],
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Center K and Q tensors by subtracting the mean.

        Args:
            keys: Key tensor with shape (batch_size, num_heads, seq_len_keys, head_dim).
            queries: Query tensor with shape (batch_size, num_heads, seq_len_queries, head_dim).
            sparse_meta_data: Metadata dictionary containing additional information.
            layer_idx: Index of the layer for which centering is applied.

        Returns:
            Tuple of (centered_keys, centered_queries) tensors.
        """
        self._initialize_key_mean_cache(sparse_meta_data, layer_idx)
        if not self.center:
            return keys, queries

        # cache the key means if not already cached
        if sparse_meta_data["key_mean"][layer_idx] is None:
            sparse_meta_data["key_mean"][layer_idx] = keys.mean(dim=2, keepdim=True)

        key_mean: torch.Tensor = sparse_meta_data["key_mean"][layer_idx]

        centered_keys = keys - key_mean
        centered_queries = queries - key_mean

        return centered_keys, centered_queries

    def _initialize_key_mean_cache(
        self,
        sparse_meta_data: Dict[Any, Any],
        layer_idx: int,
    ) -> None:
        """Initialize the key mean cache for centering keys and queries.

        This method checks if the key mean cache exists in the sparse_meta_data.
        If not, it initializes it with zeros.

        Args:
            sparse_meta_data: Metadata dictionary containing additional information.
            layer_idx: Index of the layer for which the cache is initialized.
        """
        if "key_mean" not in sparse_meta_data:
            sparse_meta_data["key_mean"] = {}
        if layer_idx not in sparse_meta_data["key_mean"]:
            sparse_meta_data["key_mean"][layer_idx] = None

    def _initialize_projection_cache(
        self,
        sparse_meta_data: Dict[Any, Any],
        layer_idx: int,
    ) -> None:
        """Initialize the projection cache for LSH matches.

        This method checks if the projection cache exists in the sparse_meta_data.
        If not, it initializes it with zeros.

        Args:
            sparse_meta_data: Metadata dictionary containing additional information.
            layer_idx: Index of the layer for which the cache is initialized.
        """
        if "projections" not in sparse_meta_data:
            sparse_meta_data["projections"] = {}
        if layer_idx not in sparse_meta_data["projections"]:
            sparse_meta_data["projections"][layer_idx] = None

    def _initialize_key_signature_cache(
        self,
        sparse_meta_data: Dict[str, Dict[int, Optional[torch.Tensor]]],
        layer_idx: int,
    ) -> None:
        """Initialize the key signature cache structure in sparse_meta_data."""
        if "key_signature" not in sparse_meta_data:
            sparse_meta_data["key_signature"] = {}
        if layer_idx not in sparse_meta_data["key_signature"]:
            sparse_meta_data["key_signature"][layer_idx] = None

    def _determine_new_keys(
        self,
        keys: torch.Tensor,
        sparse_meta_data: Dict[str, Dict[int, Optional[torch.Tensor]]],
        layer_idx: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Determine which keys are new and need signature computation.

        Returns:
            (cached_signatures, new_keys):
            - If cached_signatures is None, all keys are new
            - If new_keys is None, no new keys to process (return cached_signatures)
        """
        cached_signatures: Optional[torch.Tensor] = sparse_meta_data["key_signature"][
            layer_idx
        ]

        if cached_signatures is None:
            return None, keys

        cached_num_keys: int = cached_signatures.shape[2]
        current_num_keys: int = keys.shape[2]

        if current_num_keys < cached_num_keys:
            raise ValueError(
                f"Current number of keys ({current_num_keys}) is less than cached number of keys ({cached_num_keys})"
            )
        elif current_num_keys > cached_num_keys:
            new_keys: torch.Tensor = keys[:, :, cached_num_keys:, :]
            return cached_signatures, new_keys
        else:
            return cached_signatures, None

    def _compute_key_signatures(
        self, keys: torch.Tensor, sparse_meta_data: Dict[Any, Any], layer_idx: int
    ) -> torch.Tensor:
        total_bits: int = self.lsh_l * self.lsh_k
        batch_size, num_heads, seq_len_keys, head_dim = keys.shape
        keys_flat: torch.Tensor = keys.view(-1, head_dim)
        projection = sparse_meta_data["projections"][layer_idx]
        keys_signatures = torch.sign(
            torch.matmul(keys_flat, projection)
        )  # (batch_size * num_heads * seq_len_keys, total_bits)
        # Reshape back to original dimensions
        keys_signatures = keys_signatures.view(
            batch_size, num_heads, seq_len_keys, total_bits
        )

        if self.packing == "int64":
            keys_signatures = keys_signatures.to(torch.int64)

        return keys_signatures

    def _update_and_return_key_signatures(
        self,
        cached_signatures: torch.Tensor,
        new_signatures: torch.Tensor,
        sparse_meta_data: Dict[Any, Any],
        layer_idx: int,
    ) -> torch.Tensor:
        """Update the cache with new signatures and return the complete signature tensor."""
        if cached_signatures is None:
            sparse_meta_data["key_signature"][layer_idx] = new_signatures
            return new_signatures
        else:
            concatenated_signatures: torch.Tensor = torch.cat(
                [cached_signatures, new_signatures], dim=2
            )
            sparse_meta_data["key_signature"][layer_idx] = concatenated_signatures
            return concatenated_signatures

    def _update_key_signatures(
        self,
        keys: torch.Tensor,
        sparse_meta_data: Dict[Any, Any],
        layer_idx: int,
    ) -> torch.Tensor:
        self._initialize_key_signature_cache(sparse_meta_data, layer_idx)

        cached_signatures: Optional[torch.Tensor]
        new_keys: Optional[torch.Tensor]
        cached_signatures, new_keys = self._determine_new_keys(
            keys, sparse_meta_data, layer_idx
        )

        if new_keys is None:
            assert (
                cached_signatures is not None
            ), "cached_signatures should not be None when new_keys is None"
            return cached_signatures

        new_signatures: torch.Tensor = self._compute_signatures(
            new_keys, sparse_meta_data, layer_idx
        )
        return self._update_and_return_key_signatures(
            cached_signatures, new_signatures, sparse_meta_data, layer_idx
        )

    def _transform_for_mips(
        self, keys: torch.Tensor, queries: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform keys and queries for maximum inner product search.

        Uses the technique from Neyshabur, Behnam and Srebro, Nathan.
        "On symmetric and asymmetric LSHs for inner product search."

        Args:
            keys: Key tensor with shape (batch_size, num_heads, seq_len_keys, head_dim).
            queries: Query tensor with shape (batch_size, num_heads, seq_len_queries, head_dim).

        Returns:
            Tuple of (transformed_keys, transformed_queries) with augmented dimensions.
        """
        # Normalize queries
        query_norms: torch.Tensor = torch.norm(queries, dim=-1, keepdim=True)
        queries_normalized: torch.Tensor = queries / (query_norms + 1e-8)

        # For keys, use the augmentation technique
        key_norms: torch.Tensor = torch.norm(keys, dim=-1, keepdim=True)
        max_key_norm: torch.Tensor = torch.max(key_norms)

        # Scale keys by max norm
        keys_scaled: torch.Tensor = keys / max_key_norm

        # Compute augmentation terms
        key_augmentation: torch.Tensor = (
            torch.sqrt(max_key_norm**2 - key_norms**2) / max_key_norm
        )
        query_augmentation: torch.Tensor = torch.zeros_like(queries_normalized[..., :1])

        # Concatenate augmentation terms
        keys_transformed: torch.Tensor = torch.cat(
            [keys_scaled, key_augmentation], dim=-1
        )
        queries_transformed: torch.Tensor = torch.cat(
            [queries_normalized, query_augmentation], dim=-1
        )

        return keys_transformed, queries_transformed

    def _compute_probabilities(
        self,
        keys_transformed: torch.Tensor,
        queries_transformed: torch.Tensor,
    ) -> torch.Tensor:
        """Compute LSH collision probabilities using cosine similarity on transformed vectors."""

        # Normalize the already-transformed vectors
        keys_norm: torch.Tensor = torch.norm(keys_transformed, dim=-1, keepdim=True)
        queries_norm: torch.Tensor = torch.norm(
            queries_transformed, dim=-1, keepdim=True
        )

        keys_normalized: torch.Tensor = keys_transformed / (keys_norm + 1e-8)
        queries_normalized: torch.Tensor = queries_transformed / (queries_norm + 1e-8)

        # Compute cosine similarities
        cosine_similarities: torch.Tensor = torch.matmul(
            queries_normalized, keys_normalized.transpose(-2, -1)
        )
        cosine_similarities = torch.clamp(cosine_similarities, -1.0, 1.0)

        # Convert to angles
        angles: torch.Tensor = torch.acos(cosine_similarities)

        # Compute LSH collision probability
        single_table_prob: torch.Tensor = (1 - angles / torch.pi) ** self.lsh_k
        collision_prob: torch.Tensor = 1 - (1 - single_table_prob) ** self.lsh_l

        return collision_prob

    def _compute_matches_float32(
        self, keys_signatures: torch.Tensor, queries_signatures: torch.Tensor
    ) -> torch.Tensor:
        """Compute LSH matches using float32 signatures.

        Args:
            keys_signatures: Key signatures tensor with shape (batch_size, num_heads, seq_len_keys, total_bits).
            queries_signatures: Query signatures tensor with shape (batch_size, num_heads, seq_len_queries, total_bits).

        Returns:
            Binary matches tensor with shape (batch_size, num_heads, seq_len_queries, seq_len_keys).
        """
        batch_size: int = keys_signatures.shape[0]
        num_heads: int = keys_signatures.shape[1]
        seq_len_queries: int = queries_signatures.shape[2]
        seq_len_keys: int = keys_signatures.shape[2]

        # Expand dimensions for broadcasting
        keys_signatures_expanded: torch.Tensor = keys_signatures.unsqueeze(2)
        queries_signatures_expanded: torch.Tensor = queries_signatures.unsqueeze(3)

        # Compute element-wise product, +1 if same, -1 if different
        signature_matches: torch.Tensor = (
            keys_signatures_expanded * queries_signatures_expanded
        )
        # Shape: (batch_size, num_heads, seq_len_queries, seq_len_keys, total_bits)

        # Reshape to group by LSH tables
        signature_matches_grouped: torch.Tensor = signature_matches.view(
            batch_size,
            num_heads,
            seq_len_queries,
            seq_len_keys,
            self.lsh_l,
            self.lsh_k,
        )

        # Check if at least one group (table) has all bits matching
        # Sum within each group - if sum == lsh_k, all bits match
        group_matches: torch.Tensor = (
            signature_matches_grouped.sum(dim=-1) == self.lsh_k
        ).int()
        # Shape: (batch_size, num_heads, seq_len_queries, seq_len_keys, lsh_l)

        # Check if at least one table has a match
        matches: torch.Tensor = (group_matches.sum(dim=-1) > 0).int()
        # Shape: (batch_size, num_heads, seq_len_queries, seq_len_keys)

        return matches

    def _compute_matches_int64(
        self, keys_signatures: torch.Tensor, queries_signatures: torch.Tensor
    ) -> torch.Tensor:
        """Compute LSH matches using int64 packed signatures."""
        keys_expanded = keys_signatures.unsqueeze(2)
        queries_expanded = queries_signatures.unsqueeze(3)
        table_matches = queries_expanded == keys_expanded
        matches = torch.any(table_matches, dim=-1)
        return matches.int()

    def _compute_lsh_matches(
        self,
        keys_transformed: torch.Tensor,
        queries_transformed: torch.Tensor,
        sparse_meta_data: Dict[Any, Any],
        layer_idx: int,
    ) -> torch.Tensor:
        """Compute LSH matches using bit-packed random signed projections on transformed vectors."""

        self._initialize_projection_cache(sparse_meta_data, layer_idx)

        if sparse_meta_data["projections"][layer_idx] is None:
            head_dim = keys_transformed.shape[-1]
            total_bits = self.lsh_k * self.lsh_l

            # Set seed for reproducible projections (seed is guaranteed to be non-None by validation)
            generator = torch.Generator(device=keys_transformed.device)
            generator.manual_seed(
                self.seed + layer_idx
            )  # Include layer_idx for different layers
            sparse_meta_data["projections"][layer_idx] = torch.randn(
                head_dim,
                total_bits,
                device=keys_transformed.device,
                dtype=keys_transformed.dtype,
                generator=generator,
            )

        keys_signatures: torch.Tensor = self._update_key_signatures(
            keys_transformed, sparse_meta_data, layer_idx
        )
        queries_signatures: torch.Tensor = (
            self._compute_signatures(  # Re-using the generic signature computer
                queries_transformed, sparse_meta_data, layer_idx
            )
        )

        # Call the correct matching function based on the config
        if self.packing == "int64":
            matches = self._compute_matches_int64(keys_signatures, queries_signatures)
        else:  # 'float32'
            matches = self._compute_matches_float32(keys_signatures, queries_signatures)

        return matches

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
        """Add Magic Pig mask to attention computation.

        This method implements the core LSH-based sampling logic. It computes
        LSH collision probabilities and matches, then combines them to create
        a sparse attention pattern that prioritizes similar content.

        Args:
            keys: Key tensor with shape (batch_size, num_heads, seq_len_keys, head_dim).
                The keys used in the attention computation.
            queries: Query tensor with shape (batch_size, num_heads, seq_len_queries, head_dim).
                The queries used in the attention computation.
            values: Value tensor with shape (batch_size, num_heads, seq_len_keys, head_dim).
                The values used in the attention computation.
            attention_mask: Attention mask tensor indicating which positions are valid.
                This is typically used for padding or causal masking.
            sparse_meta_data: Dictionary containing sparse attention metadata.
                This can include additional information needed for the masker.
            previous_mask: Previous attention mask to merge with the new Magic Pig mask.
                If this is a full mask, it will be returned unchanged.
            **kwargs: Additional keyword arguments that may be passed to the masker.
                These are not used in the current implementation but allow for
                future extensibility.

        Returns:
            A new Mask object representing the attention pattern after applying
            LSH-based sampling. The mask is created by merging the previous mask
            with the new Magic Pig mask.

        Note:
            - If previous_mask is a full mask, it is returned unchanged
            - The LSH parameters (lsh_k, lsh_l) control the sparsity and quality
            - Similar key-query pairs have higher probability of being attended to
            - The resulting mask is merged with the previous mask using the merge_mask method
        """
        layer_idx: int = self._validate_inputs(sparse_meta_data, kwargs)

        if previous_mask.is_full_mask():
            return previous_mask

        batch_size, num_heads, seq_len_queries, _ = queries.shape
        _, _, seq_len_keys, _ = keys.shape

        ngroups = _get_num_key_value_groups(queries, keys)
        keys = repeat_kv(keys, ngroups)

        keys_centered, queries_centered = self._center_KQ(
            keys, queries, sparse_meta_data, layer_idx
        )
        keys_transformed, queries_transformed = self._transform_for_mips(
            keys_centered, queries_centered
        )

        probabilities: torch.Tensor = self._compute_probabilities(
            keys_transformed, queries_transformed
        )
        matches: torch.Tensor = self._compute_lsh_matches(
            keys_transformed, queries_transformed, sparse_meta_data, layer_idx
        )

        dense_mask: torch.Tensor = matches * probabilities

        mask_shape: Tuple[int, int, int, int] = (
            batch_size,
            num_heads,
            seq_len_queries,
            seq_len_keys,
        )
        this_mask: Mask = Mask.create_mask_from_dense_mask(
            shape=mask_shape, mask=dense_mask, dtype=previous_mask.dtype
        )

        return previous_mask.merge_mask(this_mask, inplace=False)

    def _validate_inputs(
        self,
        sparse_meta_data: Dict[str, Dict[int, Optional[torch.Tensor]]],
        kwargs: Dict[str, Any],
    ) -> int:
        """Validate required inputs for hash attention computation and return layer_idx."""
        if sparse_meta_data is None:
            raise ValueError("sparse_meta_data cannot be None")

        layer_idx: Optional[int] = kwargs.get("layer_idx")
        if layer_idx is None:
            raise ValueError("layer_idx must be provided in kwargs")

        return layer_idx

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "MagicPig":
        """Create MagicPig instance from configuration.

        This factory method creates a MagicPig instance from a
        configuration object. It validates that the config is of the correct
        type before creating the masker.

        Args:
            config: Configuration object for the masker. Must be an instance
                of MagicPigConfig.

        Returns:
            A new MagicPig instance configured with the provided config.

        Raises:
            ValueError: If the config is not an instance of MagicPigConfig.
                This ensures type safety and prevents configuration mismatches.

        Example:
            >>> config = MagicPigConfig(lsh_l=5, lsh_k=4)
            >>> masker = MagicPig.create_from_config(config)
        """
        if not isinstance(config, MagicPigConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
