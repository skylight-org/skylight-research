"""Hash attention top-K masker implementation."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
from sparse_attention_hub.sparse_attention.utils.hashattention_utils import (
    load_hat_weights,
)
from sparse_attention_hub.sparse_attention.utils.kv_utils import (
    _get_num_key_value_groups,
    repeat_kv,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask

from ..base import TopKMasker, TopKMaskerConfig


@dataclass
class HashAttentionTopKMaskerConfig(TopKMaskerConfig):
    """Configuration for HashAttentionTopKMasker."""

    hat_bits: int
    hat_mlp_layers: int
    hat_mlp_hidden_size: int
    hat_mlp_activation: str
    hat_weights: Optional[Dict[int, Dict[str, List[torch.Tensor]]]] = None
    hat_weight_file: Optional[str] = None
    search_space: Dict[str, Any] = field(
        default_factory=lambda: {
            "heavy_size": tune.grid_search(
                [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
            )
        }
    )


@MaskerRegistry.register(HashAttentionTopKMaskerConfig)
class HashAttentionTopKMasker(TopKMasker):
    """Hash attention top-K masker."""

    heavy_size: Union[float, int]
    hat_bits: int
    hat_mlp_layers: int
    hat_mlp_hidden_size: int
    hat_mlp_activation: str
    hat_weights: Dict[int, Dict[str, List[torch.Tensor]]]

    def __init__(self, config: HashAttentionTopKMaskerConfig) -> None:
        """Initialize hash attention top-K masker with configuration."""
        super().__init__(config)
        self.heavy_size = config.heavy_size
        self.hat_bits = config.hat_bits
        self.hat_mlp_layers = config.hat_mlp_layers
        self.hat_mlp_hidden_size = config.hat_mlp_hidden_size
        self.hat_mlp_activation = config.hat_mlp_activation

        # Validate that only one of hat_weights or hat_weight_file is provided
        if config.hat_weights is not None and config.hat_weight_file is not None:
            raise ValueError(
                "Only one of hat_weights or hat_weight_file should be provided"
            )
        if config.hat_weights is None and config.hat_weight_file is None:
            raise ValueError("Either hat_weights or hat_weight_file must be provided")

        # Load weights from file if hat_weight_file is provided
        if config.hat_weight_file is not None:
            self.hat_weights = load_hat_weights(config.hat_weight_file)
        else:
            self.hat_weights = config.hat_weights

    def add_mask(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
        scaling: float,
        dropout: float,
        sparse_meta_data: Dict[str, Dict[int, Optional[torch.Tensor]]],
        previous_mask: Mask,
        **kwargs: Dict[str, Any],
    ) -> Mask:
        """Add hash attention top-K mask to enable hash-based attention selection."""
        layer_idx: int = self._validate_inputs(sparse_meta_data, kwargs)

        # Ensure hat weights are on the same device as the keys
        # This will move the weights to the GPU if they are on the CPU on the first call
        self._ensure_hat_weights_on_device(str(keys.device))

        if previous_mask.is_full_mask():
            return previous_mask

        tensor_dims: AttentionTensorDimensions = self._extract_tensor_dimensions(
            keys, queries
        )
        effective_heavy_size: int = self._calculate_effective_heavy_size(
            tensor_dims.seq_len_keys
        )

        if self._should_use_full_attention(tensor_dims, effective_heavy_size):
            return self._create_full_mask(
                tensor_dims, previous_mask.dtype, previous_mask.device
            )

        remaining_kwargs: Dict[str, Any] = {
            k: v for k, v in kwargs.items() if k != "layer_idx"
        }
        hash_mask: Mask = self._create_hash_topk_mask(
            tensor_dims,
            effective_heavy_size,
            keys,
            queries,
            attention_mask,
            sparse_meta_data,
            previous_mask,
            layer_idx,
            **remaining_kwargs,
        )
        return previous_mask.merge_mask(hash_mask, inplace=False)

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

    def _extract_weights_for_tensor_type(
        self, layer_idx: int, tensor_type: str
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Extract weight matrices and bias vectors for a specific tensor type (key or query)."""
        weights: Dict[str, List[torch.Tensor]] = self.hat_weights[layer_idx]
        matrix_list: List[torch.Tensor] = weights[f"{tensor_type}_matrix"]
        bias_list: List[torch.Tensor] = weights[f"{tensor_type}_bias"]
        return matrix_list, bias_list

    def _compute_signatures_for_tensor_type(
        self, tensor: torch.Tensor, layer_idx: int, tensor_type: str
    ) -> torch.Tensor:
        """Compute signatures for a specific tensor type (key or query)."""
        matrix_list: List[torch.Tensor]
        bias_list: List[torch.Tensor]
        matrix_list, bias_list = self._extract_weights_for_tensor_type(
            layer_idx, tensor_type
        )
        return self._get_signatures(tensor, matrix_list, bias_list)

    def _calculate_effective_heavy_size(self, seq_len_keys: int) -> int:
        """Calculate the effective heavy size based on configuration."""
        return self._calculate_effective_size(self.heavy_size, seq_len_keys)

    def _should_use_full_attention(
        self, dims: AttentionTensorDimensions, heavy_size: int
    ) -> bool:
        """Determine if full attention should be used instead of hash attention."""
        return dims.seq_len_keys <= heavy_size

    def _create_hash_topk_mask(
        self,
        dims: AttentionTensorDimensions,
        heavy_size: int,
        keys: torch.Tensor,
        queries: torch.Tensor,
        attention_mask: torch.Tensor,
        sparse_meta_data: Dict[str, Dict[int, Optional[torch.Tensor]]],
        previous_mask: Mask,
        layer_idx: int,
        **kwargs: Dict[str, Any],
    ) -> Mask:
        """Create hash attention top-K mask using hash-based scoring."""
        scores: torch.Tensor = self._compute_hashattention_score(
            queries,
            keys,
            attention_mask,
            previous_mask.get_dense_mask(),
            sparse_meta_data,
            layer_idx,
            **kwargs,
        )
        top_k_indices: torch.Tensor = self._get_topk_indices_from_inactive_positions(
            scores, previous_mask, heavy_size
        )
        return self._create_mask_from_rowise_indices(
            dims, top_k_indices, keys.device, previous_mask.dtype
        )

    def _get_signatures(
        self,
        input_tensor: torch.Tensor,
        matrix_list: List[torch.Tensor],
        bias_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute signatures using MLP layers with activation and sign function."""
        activation_map: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
            "relu": torch.nn.functional.relu,
            "silu": torch.nn.functional.silu,
            "gelu": torch.nn.functional.gelu,
            "tanh": torch.nn.functional.tanh,
        }

        if self.hat_mlp_activation not in activation_map:
            raise ValueError(
                f"Unsupported activation function: {self.hat_mlp_activation}"
            )

        activation_fn: Callable[[torch.Tensor], torch.Tensor] = activation_map[
            self.hat_mlp_activation
        ]
        signatures: torch.Tensor = input_tensor

        for i in range(len(matrix_list) - 1):
            signatures = self._apply_single_layer(
                signatures, matrix_list[i], bias_list[i]
            )
            signatures = activation_fn(signatures)

        if len(matrix_list) > 0:
            signatures = self._apply_single_layer(
                signatures, matrix_list[-1], bias_list[-1]
            )

        return torch.sign(signatures)

    def _apply_single_layer(
        self,
        input_tensor: torch.Tensor,
        weight_matrix: torch.Tensor,
        bias_vector: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a single linear layer (matrix multiplication + bias addition)."""
        # (B,H,s,d) x (H,d,d_out) -> (B,H,s,d_out)
        output: torch.Tensor = torch.einsum(
            "bhsd,hde->bhse", input_tensor, weight_matrix
        )
        # (B,H,s,d_out) + (H,d_out) -> (B,H,s,d_out)
        output = output + bias_vector.unsqueeze(0).unsqueeze(2)
        return output

    def _update_key_signatures(
        self,
        keys: torch.Tensor,
        sparse_meta_data: Dict[str, Dict[int, Optional[torch.Tensor]]],
        layer_idx: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Update key signatures in sparse_meta_data and return concatenated signatures."""
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

        new_signatures: torch.Tensor = self._compute_signatures_for_tensor_type(
            new_keys, layer_idx, "key"
        )
        return self._update_and_return_key_signatures(
            cached_signatures, new_signatures, sparse_meta_data, layer_idx
        )

    def _initialize_key_signature_cache(
        self,
        sparse_meta_data: Dict[str, Dict[int, Optional[torch.Tensor]]],
        layer_idx: int,
    ) -> None:
        """Initialize the key signature cache structure in sparse_meta_data."""
        if "key" not in sparse_meta_data:
            sparse_meta_data["key"] = {}
        if layer_idx not in sparse_meta_data["key"]:
            sparse_meta_data["key"][layer_idx] = None

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
        cached_signatures: Optional[torch.Tensor] = sparse_meta_data["key"][layer_idx]

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

    def _update_and_return_key_signatures(
        self,
        cached_signatures: Optional[torch.Tensor],
        new_signatures: torch.Tensor,
        sparse_meta_data: Dict[str, Dict[int, Optional[torch.Tensor]]],
        layer_idx: int,
    ) -> torch.Tensor:
        """Update the cache with new signatures and return the complete signature tensor."""
        if cached_signatures is None:
            sparse_meta_data["key"][layer_idx] = new_signatures
            return new_signatures
        else:
            concatenated_signatures: torch.Tensor = torch.cat(
                [cached_signatures, new_signatures], dim=2
            )
            sparse_meta_data["key"][layer_idx] = concatenated_signatures
            return concatenated_signatures

    def _compute_hashattention_score(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        attention_mask: torch.Tensor,
        previous_dense_mask: torch.Tensor,
        sparse_meta_data: Dict[str, Dict[int, Optional[torch.Tensor]]],
        layer_idx: int,
        **kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """Compute hash attention scores using query and key signatures."""
        num_key_value_groups: int = _get_num_key_value_groups(queries, keys)
        processed_keys: torch.Tensor = repeat_kv(keys, num_key_value_groups)

        key_signatures: torch.Tensor = self._update_key_signatures(
            processed_keys, sparse_meta_data, layer_idx, **kwargs
        )
        query_signatures: torch.Tensor = self._compute_signatures_for_tensor_type(
            queries, layer_idx, "query"
        )

        # (B, H, #queries, hat_bits) x (B, H, hat_bits, #keys) -> (B, H, #queries, #keys)
        scores: torch.Tensor = torch.matmul(
            query_signatures, key_signatures.transpose(-2, -1)
        )
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, :, : keys.shape[-2]]
        scores[previous_dense_mask != 0] = torch.finfo(scores.dtype).min
        return scores

    def _ensure_hat_weights_on_device(self, device: str) -> None:
        """Move hat weights to the specified device."""
        for layer_idx, layer_weights in self.hat_weights.items():
            for key, value in layer_weights.items():
                self.hat_weights[layer_idx][key] = [
                    tensor.to(device) for tensor in value
                ]

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "HashAttentionTopKMasker":
        """Create HashAttentionTopKMasker instance from configuration."""
        if not isinstance(config, HashAttentionTopKMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
