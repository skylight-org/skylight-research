"""Base classes for research maskers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Type, TypeVar, Union

import torch

from ...utils.mask import Mask

# Type variables for the decorator
T = TypeVar("T", bound="ResearchMasker")
ConfigType = TypeVar("ConfigType", bound="MaskerConfig")


@dataclass
class MaskerConfig:
    """Base configuration class for all maskers."""


@dataclass
class AttentionTensorDimensions:
    """Container for tensor dimensions used in attention mask computation."""

    batch_size: int
    num_heads: int
    seq_len_queries: int
    seq_len_keys: int


class MaskerRegistry:
    """Registry for masker classes using decorator-based auto-registration."""

    _registry: Dict[Type[MaskerConfig], Type["ResearchMasker"]] = {}

    @classmethod
    def register(cls, config_type: Type[ConfigType]) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a masker class with its config type.

        Args:
            config_type: The configuration class type that this masker handles.

        Returns:
            Decorator function that registers the masker class.

        Example:
            @MaskerRegistry.register(LocalMaskerConfig)
            class LocalMasker(FixedMasker):
                ...
        """

        def decorator(masker_class: Type[T]) -> Type[T]:
            cls._registry[config_type] = masker_class
            return masker_class

        return decorator

    @classmethod
    def get_masker_class(
        cls, config_type: Type[MaskerConfig]
    ) -> Type["ResearchMasker"]:
        """Get the masker class for a given config type.

        Args:
            config_type: The configuration class type.

        Returns:
            The masker class that handles this config type.

        Raises:
            ValueError: If no masker class is registered for the config type.
        """
        masker_class: Optional[Type["ResearchMasker"]] = cls._registry.get(
            config_type
        )
        if masker_class is None:
            raise ValueError(
                f"No masker class registered for config type: {config_type}"
            )
        return masker_class

    @classmethod
    def get_all_registered_types(
        cls,
    ) -> Dict[Type[MaskerConfig], Type["ResearchMasker"]]:
        """Get all registered config types and their corresponding masker classes."""
        return cls._registry.copy()


class ResearchMasker(ABC):
    """Abstract base class for research maskers."""

    config: MaskerConfig

    def __init__(self, config: MaskerConfig) -> None:
        """Initialize masker with configuration."""
        self.config = config

    def _extract_tensor_dimensions(
        self, keys: torch.Tensor, queries: torch.Tensor
    ) -> AttentionTensorDimensions:
        """Extract relevant tensor dimensions for mask computation.

        Args:
            keys: Key tensor with shape (batch_size, num_heads, seq_len_keys, head_dim)
            queries: Query tensor with shape (batch_size, num_heads, seq_len_queries, head_dim)

        Returns:
            AttentionTensorDimensions containing extracted dimensions
        """
        return AttentionTensorDimensions(
            batch_size=queries.shape[0],
            num_heads=queries.shape[1],
            seq_len_queries=queries.shape[2],
            seq_len_keys=keys.shape[2],
        )

    def _create_full_mask(
        self, dims: AttentionTensorDimensions, dtype: torch.dtype, device: torch.device
    ) -> Mask:
        """Create a full attention mask."""
        mask_shape: Tuple[int, int, int, int] = (
            dims.batch_size,
            dims.num_heads,
            dims.seq_len_queries,
            dims.seq_len_keys,
        )
        return Mask.create_full_mask(mask_shape, dtype=dtype, device=device)

    def _create_mask_from_rowise_indices(
        self,
        dims: AttentionTensorDimensions,
        indices: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Mask:
        """Create mask from row-wise indices."""
        mask_shape: Tuple[int, int, int, int] = (
            dims.batch_size,
            dims.num_heads,
            dims.seq_len_queries,
            dims.seq_len_keys,
        )
        data: torch.Tensor = torch.ones_like(indices, dtype=dtype, device=device)

        return Mask.create_from_row_wise_idx(
            shape=mask_shape,
            row_wise_idx=indices,
            data=data,
            mask_type="dense",
            dtype=dtype,
        )

    def _calculate_effective_size(
        self, size_param: Union[float, int], seq_len_keys: int
    ) -> int:
        """Calculate effective size based on configuration (supports both int and float)."""
        if isinstance(size_param, float):
            return int(size_param * seq_len_keys)
        return int(size_param)

    @abstractmethod
    def add_mask(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
        scaling: float,
        dropout: float,
        sparse_meta_data: Dict[Any, Any],  # want to keep it general here.
        previous_mask: Mask,
        **kwargs: Dict[str, Any],
    ) -> Mask:
        """Add mask to attention computation."""
        pass

    @classmethod
    @abstractmethod
    def create_from_config(cls, config: MaskerConfig) -> "ResearchMasker":
        """Create masker instance from configuration.

        Args:
            config: Configuration for the masker.

        Returns:
            Instance of the masker.
        """
        pass

    @classmethod
    def create_masker_from_config(cls, config: MaskerConfig) -> "ResearchMasker":
        """Create masker instance from configuration using the registry.

        Args:
            config: Configuration for the masker.

        Returns:
            Instance of the concrete masker class.

        Raises:
            ValueError: If no masker class is found for the config type.
                Make sure to import the masker module before calling this method.
        """
        # Get the masker class from the registry
        masker_class: Type["ResearchMasker"] = MaskerRegistry.get_masker_class(
            type(config)
        )

        return masker_class.create_from_config(config)
