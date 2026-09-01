"""Configuration builder for ScaNN attention."""

from typing import Dict, List, Optional, Tuple

from ray import tune

from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttentionConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig,
    ScaNNTopKMaskerConfig,
    SinkMaskerConfig,
)

from .base import BaseConfigBuilder
from .factory import register_builder
from .utility import get_masker_list_name


@register_builder("scann_topk")
class ScaNNTopKConfigBuilder(BaseConfigBuilder):
    """Builder for ScaNN TopK sparse attention configurations."""

    def build_configs(
        self,
        model_config: Dict[str, str],
        sparsity_objectives: List[int],
        memory_objectives: List[int],
        **kwargs,
    ) -> Tuple[
        List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]],
        List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]],
    ]:
        """Get all ScaNN attention configurations.

        Uses:
            sparsity_objectives: List[int] - List of sparsity objectives to build the configurations.
        Ignores:
            memory_objectives: List[int] - List of memory objectives
            model_config: Dict[str, str] - Model configuration

        Returns:
            Tuple of (optimal_configs, to_optimize_configs)
        """
        optimal_configs: List[
            Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]
        ] = []
        to_optimize_configs: List[
            Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]
        ] = []

        for sparsity_objective in sparsity_objectives:
            heavy_size: float = float(sparsity_objective) / 100.0
            classes = [SinkMaskerConfig, LocalMaskerConfig, ScaNNTopKMaskerConfig]
            name: str = get_masker_list_name(
                classes,
                other_params={
                    "builder": "scann_topk",
                    "sparsity_obj": sparsity_objective,
                },
            )

            config = ResearchAttentionConfig(
                masker_configs=[
                    SinkMaskerConfig(sink_size=128),
                    LocalMaskerConfig(window_size=128),
                    ScaNNTopKMaskerConfig(
                        heavy_size=heavy_size - (256.0 / 32768),
                        num_leaves=256,  # ~sqrt(32768), ScaNN's rule of thumb
                        num_leaves_to_search=32,
                        dimensions_per_block=2,  # ScaNN's documented default
                        num_centers=16,  # LUT16
                        anisotropic_threshold=0.2,  # canonical value from the paper
                        init_offset=128,  # matches sink_size
                    ),
                ]
            )

            config.masker_configs[2].search_space = {
                "dimensions_per_block": tune.grid_search([2, 4, 8]),
                "num_leaves_to_search": tune.grid_search([16, 32, 64]),
                "anisotropic_threshold": tune.grid_search([None, 0.1, 0.2, 0.3]),
            }

            # Set validity to default (doesn't depend on memory objectives)
            config.validity_constraint = lambda config: True
            # Set objective function
            config.objective = sparsity_objective

            to_optimize_configs.append((name, config, classes))

        return optimal_configs, to_optimize_configs
