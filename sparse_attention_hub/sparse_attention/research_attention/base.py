"""Base classes for research attention mechanisms."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from sparse_attention_hub.metric_logging.logger import MicroMetricLogger

from ..base import SparseAttention, SparseAttentionConfig
from ..utils.mask import Mask
from ..utils.mask_attention_utils import (
    get_masked_attention_output,
    get_true_attention_output,
)
from .maskers.base import MaskerConfig, ResearchMasker
from .maskers.sampling.base import SamplingMasker

MicroMetricLogger.register_metric("research_attention_density", float)
MicroMetricLogger.register_metric("research_attention_output_error", float)


@dataclass
class ResearchAttentionConfig(SparseAttentionConfig):
    """Configuration class for research attention mechanisms.

    Attributes:
        masker_configs: Maskers applied sequentially, each on the previous one's mask.
        apply_to_layer_types: Restricts sparse attention to layers whose entry in
            ``model.config.layer_types`` is in this tuple. ``None`` (the default) applies
            sparse attention to every layer, which is the legacy behaviour and the only
            behaviour for models without a ``layer_types`` config (e.g. Llama, Qwen2.5).
            For hybrid sliding/full models such as Olmo-3 or Gemma-3, passing
            ``("full_attention",)`` leaves the sliding-window layers running exact dense
            attention and suppresses their density/error micro-metrics.
    """

    masker_configs: List[MaskerConfig]
    apply_to_layer_types: Optional[Tuple[str, ...]] = None


class ResearchAttention(SparseAttention):
    """Base class for research attention mechanisms with maskers."""

    maskers: List[ResearchMasker]

    def __init__(
        self,
        sparse_attention_config: SparseAttentionConfig,
        maskers: List[ResearchMasker],
    ) -> None:
        """Initialize research attention mechanism.

        Args:
            sparse_attention_config: Configuration for the sparse attention mechanism.
            maskers: List of research maskers to apply.

        Raises:
            ValueError: If more than one sampling masker is provided.
        """
        super().__init__(sparse_attention_config)

        # Validate that there's at most one sampling masker
        sampling_masker_count: int = sum(
            1 for masker in maskers if isinstance(masker, SamplingMasker)
        )
        if sampling_masker_count > 1:
            raise ValueError(
                "Only one sampling masker supported for efficiency; consider implementing all sampling logic in one masker"
            )

        self.maskers = maskers

        # getattr keeps subclassed / older pickled configs working.
        self.apply_to_layer_types: Optional[Tuple[str, ...]] = getattr(
            sparse_attention_config, "apply_to_layer_types", None
        )

    def _applies_to_layer(self, module: nn.Module, kwargs: Dict[str, Any]) -> bool:
        """Whether sparse attention should be applied to the layer owning ``module``.

        Gates on ``module.config.layer_types[layer_idx]`` rather than an ``attention_type``
        attribute: Olmo-3 sets ``attention_type`` on the attention module, but Qwen3 and
        GPT-OSS set it on the decoder layer, and ``module`` here is the attention module.

        Returns True (sparse) whenever the model exposes no hybrid structure, so
        non-hybrid models are unaffected.
        """
        if self.apply_to_layer_types is None:
            return True

        layer_types: Any = getattr(getattr(module, "config", None), "layer_types", None)
        if not layer_types:
            return True

        layer_idx: Any = kwargs.get("layer_idx", getattr(module, "layer_idx", None))
        if not isinstance(layer_idx, int) or layer_idx >= len(layer_types):
            return True

        return bool(layer_types[layer_idx] in self.apply_to_layer_types)

    def custom_attention(
        self,
        module: nn.Module,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float,
        **kwargs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute research attention mechanism with masking.

        Args:
            module: The attention module
            queries: Query tensor of shape (b, h, sk, d)
            keys: Key tensor of shape (b, h, sq, d)
            values: Value tensor of shape (b, h, sq, d)
            attention_mask: Optional attention mask of shape (b, h, sq, sk)
            scaling: Scaling factor for attention weights
            dropout: Dropout probability
            **kwargs: Additional keyword arguments, must include sparse_meta_data

        Returns:
            Tuple of attention output and optional attention weights.
        """
        # Extract sparse_meta_data from kwargs
        if "sparse_meta_data" not in kwargs:
            raise ValueError(
                "sparse_meta_data must be provided in kwargs while calling custom_attention()"
            )
        sparse_meta_data: Dict[Any, Any] = kwargs.pop("sparse_meta_data")

        # Layer-type gating: run exact dense attention on excluded layers and emit no
        # micro-metrics for them. The mask HuggingFace passed in is already the correct
        # per-layer mask (e.g. causal AND sliding-window), so this reproduces the model's
        # native dense attention for that layer.
        if not self._applies_to_layer(module, kwargs):
            if kwargs.get("s_aux") is not None:
                raise NotImplementedError(
                    "apply_to_layer_types is not supported for attention-sink models: "
                    "the dense fallback does not apply the s_aux sink bias."
                )
            kwargs.pop("s_aux", None)
            return get_true_attention_output(
                module,
                queries,
                keys,
                values,
                attention_mask,
                scaling,
                dropout,
                **kwargs,
            )

        # Create an empty Mask object
        mask_shape: Tuple[int, int, int, int] = (
            queries.shape[0],
            queries.shape[1],
            queries.shape[2],
            keys.shape[2],
        )
        sparse_attention_mask: Mask = Mask.create_empty_mask(
            mask_shape, dtype=queries.dtype, device=queries.device
        )

        # Apply all maskers sequentially, each one on the output of the previous one
        for masker in self.maskers:
            sparse_attention_mask = masker.add_mask(
                keys=keys,
                queries=queries,
                values=values,
                attention_mask=attention_mask,
                scaling=scaling,
                dropout=dropout,
                sparse_meta_data=sparse_meta_data,
                previous_mask=sparse_attention_mask,
                **kwargs,
            )

        if MicroMetricLogger().is_metric_enabled("research_attention_density"):
            MicroMetricLogger().log(
                "research_attention_density",
                sparse_attention_mask.get_density(),
                metadata={
                    "layer_idx": kwargs["layer_idx"],
                    "q_len": queries.shape[2],
                    "k_len": keys.shape[2],
                },
            )

        s_aux_raw: Any = kwargs.pop("s_aux", None)
        s_aux: Optional[torch.Tensor] = (
            s_aux_raw if isinstance(s_aux_raw, torch.Tensor) else None
        )
        # Call compute_masked_attention_output on the result of the last mask
        # Always request attention weights to match the expected return signature
        attention_output: torch.Tensor
        attention_weights: torch.Tensor
        attention_output, attention_weights = get_masked_attention_output(
            module=module,
            queries=queries,
            keys=keys,
            values=values,
            attention_mask=attention_mask,
            sinks=s_aux,
            scaling=scaling,
            dropout=dropout,
            sparse_attention_mask=sparse_attention_mask,
            return_attention_weights=True,
            **kwargs,
        )

        if MicroMetricLogger().is_metric_enabled("research_attention_output_error"):
            true_attention_output, _ = get_true_attention_output(
                module,
                queries,
                keys,
                values,
                attention_mask,
                scaling,
                dropout,
                **kwargs,
            )
            error = torch.norm(true_attention_output - attention_output) / torch.norm(
                true_attention_output
            )
            MicroMetricLogger().log(
                "research_attention_output_error",
                float(error.item()),
                metadata={
                    "layer_idx": kwargs["layer_idx"],
                    "q_len": queries.shape[2],
                    "k_len": keys.shape[2],
                },
            )

        return attention_output, attention_weights

    @classmethod
    def create_from_config(cls, config: SparseAttentionConfig) -> "ResearchAttention":
        """Create research attention instance from configuration.

        Args:
            config: Configuration for the research attention mechanism.

        Returns:
            Instance of the research attention mechanism.

        Raises:
            TypeError: If config is not a ResearchAttentionConfig.
        """
        if not isinstance(config, ResearchAttentionConfig):
            raise TypeError(f"Expected ResearchAttentionConfig, got {type(config)}")

        # Create ResearchMasker objects from the configs using the factory method
        maskers: List[ResearchMasker] = []
        for masker_config in config.masker_configs:
            masker: ResearchMasker = ResearchMasker.create_masker_from_config(
                masker_config
            )
            maskers.append(masker)

        return cls(config, maskers)
