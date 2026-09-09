"""Adaptive sampling masker implementation.

This module provides an adaptive sampling masker that determines sampling budgets
based on statistical error bounds. It combines base sampling with adaptive budget
allocation to achieve optimal sparsity while maintaining statistical guarantees.

The AdaptiveSamplingMasker is useful for:
- Achieving optimal sparsity with statistical guarantees
- Adaptively adjusting sampling based on attention patterns
- Maintaining error bounds while reducing computational complexity
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Union

import torch
try:
    from ray import tune
except ImportError:
    class _TuneStub:
        @staticmethod
        def grid_search(values):
            return values

    tune = _TuneStub()
from scipy.stats import norm

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask
from sparse_attention_hub.sparse_attention.utils.mask_attention_utils import (
    _get_num_key_value_groups,
    apply_inv_mask_sum,
    create_sampling_mask_with_per_head_budget,
    repeat_kv,
)

from ..base import SamplingMasker, SamplingMaskerConfig


@dataclass
class AdaptiveSamplingMaskerConfig(SamplingMaskerConfig):
    """Configuration for AdaptiveSamplingMasker.

    This configuration class inherits from SamplingMaskerConfig and adds
    validation to ensure all parameters are within valid ranges.

    Attributes:
        base_rate_sampling: Union[int, float] representing the base sampling rate.
            If float, must be in [0,1); if int, must be non-negative.
            When set to 0, the masker returns the previous mask without modification.
        epsilon: Float in range (0,1) representing the error bound.
        delta: Float in range (0,1) representing the confidence bound.
        init_offset: Union[int, float] representing the start index for sampling.
            If int, must be non-negative; if float, must be in [0,1] and will be
            multiplied by the number of keys to get the actual offset.
        local_offset: Union[int, float] representing the end offset for sampling.
            If int, must be non-negative; if float, must be in [0,1] and will be
            multiplied by the number of keys to get the actual offset.
    """

    base_rate_sampling: Union[int, float]  # Base rate (0,1) if float
    epsilon: float  # Error bound (0,1)
    delta: float  # Confidence bound (0,1)
    init_offset: Union[int, float]  # Start index
    local_offset: Union[int, float]  # End offset
    search_space: Dict[str, Any] = field(
        default_factory=lambda: {
            "base_rate_sampling": tune.grid_search([0.01, 0.02, 0.03]),
            "epsilon": tune.grid_search([0.05, 0.1, 0.2, 0.3]),
            "delta": tune.grid_search([0.05, 0.1, 0.2, 0.3]),
        }
    )

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if isinstance(self.base_rate_sampling, float):
            if not (0.0 <= self.base_rate_sampling < 1.0):
                raise ValueError(
                    f"base_rate_sampling must be in [0, 1) if float, got {self.base_rate_sampling}"
                )
        elif isinstance(self.base_rate_sampling, int):
            if self.base_rate_sampling < 0:
                raise ValueError(
                    f"base_rate_sampling must be non-negative if int, got {self.base_rate_sampling}"
                )
        else:
            raise ValueError(
                f"base_rate_sampling must be int or float, got {type(self.base_rate_sampling)}"
            )

        if not (0.0 < self.epsilon < 1.0):
            raise ValueError(f"epsilon must be in (0, 1), got {self.epsilon}")

        if not (0.0 < self.delta < 1.0):
            raise ValueError(f"delta must be in (0, 1), got {self.delta}")

        if isinstance(self.init_offset, float):
            if not (0.0 <= self.init_offset <= 1.0):
                raise ValueError(
                    f"init_offset must be in [0, 1] if float, got {self.init_offset}"
                )
        elif isinstance(self.init_offset, int):
            if self.init_offset < 0:
                raise ValueError(
                    f"init_offset must be non-negative if int, got {self.init_offset}"
                )
        else:
            raise ValueError(
                f"init_offset must be int or float, got {type(self.init_offset)}"
            )

        if isinstance(self.local_offset, float):
            if not (0.0 <= self.local_offset <= 1.0):
                raise ValueError(
                    f"local_offset must be in [0, 1] if float, got {self.local_offset}"
                )
        elif isinstance(self.local_offset, int):
            if self.local_offset < 0:
                raise ValueError(
                    f"local_offset must be non-negative if int, got {self.local_offset}"
                )
        else:
            raise ValueError(
                f"local_offset must be int or float, got {type(self.local_offset)}"
            )


@MaskerRegistry.register(AdaptiveSamplingMaskerConfig)
class AdaptiveSamplingMasker(SamplingMasker):
    """Adaptive sampling masker for sparse attention computation.

    This masker implements adaptive sampling of attention positions by combining
    base sampling with adaptive budget allocation based on statistical error bounds.
    The masker uses a two-phase approach:
    1. Base Sampling Phase: Randomly samples a base fraction of positions
    2. Adaptive Budget Phase: Computes optimal sampling budgets per row based on
       statistical error bounds (epsilon, delta)

    Attributes:
        base_rate_sampling: The base sampling rate (int or float).
        epsilon: The error bound for statistical guarantees.
        delta: The confidence bound for statistical guarantees.
        init_offset: Starting index for sampling range (int or float).
            If float, represents fraction of sequence length.
        local_offset: Ending offset for sampling range (int or float).
            If float, represents fraction of sequence length.
        delta_ppf: Pre-computed percentile point function for efficiency.

    Important Notes:
        - If base_rate_sampling is set to 0, the masker returns the previous mask
          without any modification.
        - The sampling is performed with replacement for efficiency.
        - The masker ignores the previous mask for base sampling to avoid complex
          index manipulation.
        - Merge operation adds the data in masks and clamps to 1.0.
        - Statistical guarantees are maintained through proper error bound computation.

    Example:
        >>> config = AdaptiveSamplingMaskerConfig(
        ...     base_rate_sampling=0.1, epsilon=0.1, delta=0.05,
        ...     init_offset=0.1, local_offset=0.2  # Use 10% from start, 20% from end
        ... )
        >>> masker = AdaptiveSamplingMasker(config)
        >>> # Use masker.add_mask() to apply adaptive sampling to attention masks
    """

    def __init__(self, config: AdaptiveSamplingMaskerConfig) -> None:
        """Initialize adaptive sampling masker with configuration.

        Args:
            config: Configuration object containing the sampling parameters and
                statistical bounds for the adaptive sampling masker.

        Raises:
            ValueError: If any parameter in config is not in the valid range.
                This validation is performed in the config's __post_init__ method.
        """
        super().__init__(config)
        self.base_rate_sampling = config.base_rate_sampling
        self.epsilon = config.epsilon
        self.delta = config.delta
        self.init_offset = config.init_offset
        self.local_offset = config.local_offset

        # Pre-compute delta_ppf for efficiency
        self.delta_ppf = float(norm.ppf(1 - self.delta))

    def _compute_exp_attention_scores(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        scaling: float,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute exponential attention scores with numerical stability."""
        ngroups = _get_num_key_value_groups(queries, keys)
        keys = repeat_kv(keys, ngroups)
        raw_scores = torch.matmul(queries, keys.transpose(-2, -1)) * scaling
        if attention_mask is not None:
            raw_scores = raw_scores + attention_mask[:, :, :, : keys.shape[-2]]
        max_scores = torch.max(raw_scores, dim=-1, keepdim=True)[0]
        return torch.exp(raw_scores - max_scores)

    def should_return_full_mask(self, sampling_range: int) -> bool:
        """Check if the masker should return a full mask."""
        return sampling_range < 2

    def _get_sampling_range(self, seq_len_keys: int) -> tuple[int, int, int]:
        """Get sampling range and validate it.

        Args:
            seq_len_keys: Number of keys in the sequence.

        Returns:
            Tuple of (start_idx, end_idx, sampling_range).

        Raises:
            ValueError: If the computed sampling range is invalid.
        """
        # Compute start index
        if isinstance(self.init_offset, float):
            start_idx: int = int(self.init_offset * seq_len_keys)
        else:
            start_idx = self.init_offset

        # Compute end index
        if isinstance(self.local_offset, float):
            end_idx: int = seq_len_keys - int(self.local_offset * seq_len_keys)
        else:
            end_idx = seq_len_keys - self.local_offset

        sampling_range = end_idx - start_idx
        return start_idx, end_idx, sampling_range

    def _get_base_sample_count(self, sampling_range: int) -> int:
        """Get number of base samples based on configuration."""
        # Ensure at least 2 samples since it is used for std estimation
        if isinstance(self.base_rate_sampling, int):
            return max(2, self.base_rate_sampling)
        return max(2, int(self.base_rate_sampling * sampling_range))

    def _get_std_estimate_using_base_sample(
        self,
        expwts: torch.Tensor,
        batch_size: int,
        num_heads: int,
        seq_len_queries: int,
        seq_len_keys: int,
        start_idx: int,
        end_idx: int,
        num_base_samples: int,
        dtype: torch.dtype,
    ) -> tuple[Mask, torch.Tensor]:
        """Get standard deviation estimate using base sampling and create base mask."""
        # Create base sampling indices
        base_row_wise_idx = torch.randint(
            low=start_idx,
            high=end_idx,
            size=(batch_size, num_heads, seq_len_queries, num_base_samples),
            device=expwts.device,
        )

        # Extract values and compute std
        sampled_values = torch.gather(expwts, dim=-1, index=base_row_wise_idx)
        total_rows = batch_size * num_heads * seq_len_queries
        row_sampled_values = sampled_values.view(total_rows, num_base_samples)
        std_estimate = torch.std(row_sampled_values, dim=-1, keepdim=True)
        std_estimate = torch.clamp(std_estimate, min=1e-8)
        std_estimate = std_estimate.view(batch_size, num_heads, seq_len_queries, 1)

        # Create base sampling mask
        sampling_range = end_idx - start_idx
        base_data = torch.full_like(
            base_row_wise_idx, num_base_samples / sampling_range, dtype=expwts.dtype
        )

        base_mask = Mask.create_from_row_wise_idx(
            shape=(batch_size, num_heads, seq_len_queries, seq_len_keys),
            row_wise_idx=base_row_wise_idx,
            data=base_data,
            mask_type="dense",
            dtype=dtype,
        )

        return base_mask, std_estimate

    def _compute_adaptive_budget(
        self,
        std_estimate: torch.Tensor,
        estimated_denominator: torch.Tensor,
        sampling_range: int,
    ) -> torch.Tensor:
        """Compute adaptive budget based on statistical bounds."""
        epsilon_allowable_error = self.epsilon * estimated_denominator
        epsilon_allowable_error = torch.clamp(epsilon_allowable_error, min=1e-8)

        budget_numerator = self.delta_ppf * std_estimate * sampling_range
        budget_squared = (budget_numerator / epsilon_allowable_error) ** 2

        # Ensure budget is positive and within bounds
        budget = torch.clamp(
            budget_squared,
            min=1.0,  # Minimum 1 sample
            max=float(sampling_range),  # Maximum sampling_range samples
        ).long()

        return budget

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
        """Add adaptive sampling mask to attention computation.

        This method implements the core adaptive sampling logic. It combines base
        sampling with adaptive budget allocation based on statistical error bounds.
        If base_rate_sampling is set to 0, this method returns the previous mask
        without modification.

        Args:
            keys: Key tensor with shape (batch_size, num_heads, seq_len_keys, head_dim).
            queries: Query tensor with shape (batch_size, num_heads, seq_len_queries, head_dim).
            values: Value tensor with shape (batch_size, num_heads, seq_len_keys, head_dim).
            attention_mask: Attention mask tensor indicating which positions are valid.
            sparse_meta_data: Dictionary containing sparse attention metadata.
            previous_mask: Previous attention mask to merge with the new adaptive sampling mask.
            **kwargs: Additional keyword arguments.

        Returns:
            A new Mask object representing the attention pattern after applying
            adaptive sampling.

        Raises:
            ValueError: If the sampling range is invalid.
        """
        if previous_mask.is_full_mask():
            return previous_mask

        # If base_rate_sampling is 0, return the previous mask without modification
        if self.base_rate_sampling == 0:
            return previous_mask

        # Extract dimensions and compute attention scores
        dims = self._extract_tensor_dimensions(keys, queries)
        batch_size, num_heads, seq_len_queries, seq_len_keys = (
            dims.batch_size,
            dims.num_heads,
            dims.seq_len_queries,
            dims.seq_len_keys,
        )

        # Get sampling range
        start_idx, end_idx, sampling_range = self._get_sampling_range(seq_len_keys)

        # If sequence length is too small, return full mask
        if self.should_return_full_mask(sampling_range):
            return Mask.create_full_mask(
                shape=(batch_size, num_heads, seq_len_queries, seq_len_keys),
                dtype=previous_mask.dtype,
                device=previous_mask.device,
            )

        # Compute attention scores after removing attention_mask
        expwts = self._compute_exp_attention_scores(
            queries, keys, scaling, attention_mask
        )
        static_denominator = apply_inv_mask_sum(expwts, previous_mask)

        # Get sampling parameters

        num_base_samples = self._get_base_sample_count(sampling_range)

        # Create base sampling mask and estimate std
        base_sampling_mask, std_estimate = self._get_std_estimate_using_base_sample(
            expwts,
            batch_size,
            num_heads,
            seq_len_queries,
            seq_len_keys,
            start_idx,
            end_idx,
            num_base_samples,
            previous_mask.dtype,
        )
        # Compute denominators and budget
        sampled_denominator = apply_inv_mask_sum(expwts, base_sampling_mask)
        estimated_denominator = static_denominator + sampled_denominator
        budget = self._compute_adaptive_budget(
            std_estimate, estimated_denominator, sampling_range
        )
        budget = torch.clamp(budget, min=num_base_samples, max=sampling_range)

        # Create adaptive sampling mask
        sampling_probabilities = (budget / sampling_range).to(previous_mask.dtype)
        adaptive_mask = create_sampling_mask_with_per_head_budget(
            budgets=budget,
            sampling_probability=sampling_probabilities,
            seq_len_keys=seq_len_keys,
            start_idx=start_idx,
            end_idx=end_idx,
            dtype=previous_mask.dtype,
        )
        # Merge masks
        return previous_mask.merge_mask(adaptive_mask, inplace=False)

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "AdaptiveSamplingMasker":
        """Create AdaptiveSamplingMasker instance from configuration.

        Args:
            config: Configuration for the masker.

        Returns:
            Instance of the AdaptiveSamplingMasker.

        Raises:
            ValueError: If the config is not of type AdaptiveSamplingMaskerConfig.
        """
        if not isinstance(config, AdaptiveSamplingMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
