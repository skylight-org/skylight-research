from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.utils.kv_utils import (
    _get_num_key_value_groups,
    repeat_kv,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask

from ..base import TopKMasker, TopKMaskerConfig
from .utils.socket_utils import (
    attention_mask_to_allowed_prob,
    get_hyper_planes,
    get_protos_T,
    hard_hash,
    soft_hash,
    topk_soft_collision_scores_blockwise,
)


@dataclass
class SocketMaskerConfig(TopKMaskerConfig):
    """
    Deterministic soft-count SocketMasker (NO top-t, NO sampling, NO candidate set)

      • K:      # hyperplanes per table (buckets = 2**K)
      • L:      # hash tables
      • heavy_size (inherited): output budget M
      • tau:    temperature parameter for softmax in soft hashing
    """

    K: int = 4
    L: int = 1
    tau: float = 0.3

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.K <= 0:
            raise ValueError("K must be positive")
        if self.L <= 0:
            raise ValueError("L must be positive")
        if not isinstance(self.tau, (float, int)) or self.tau <= 0:
            raise ValueError(f"tau must be positive, got {self.tau}")


@MaskerRegistry.register(SocketMaskerConfig)
class SocketMasker(TopKMasker):
    """
    Deterministic soft-count masker:
      1) Hard-hash keys with L sets of K planes -> bucket id per table.
      2) Soft-hash queries per table -> probs over R=2^K buckets.
      3) For each key j, compute C_j(q) = sum_l p_l(bucket_l(j)).
      4) Score_j = C_j(q) * ||v_j||_2 (gate by attention_mask).
      5) Pick top-M by score and return mask.

    Note: _planes_cache is instance-level and keyed by dtype, device, etc. If a single masker instance
    is called with tensors of different dtype across calls (e.g., bf16 in one layer, fp32 in another),
    it will allocate separate plane tensors with the same seed. This is correct but may surprise users
    who assume all tables use exactly the same planes regardless of dtype.
    """

    def __init__(self, config: SocketMaskerConfig) -> None:
        super().__init__(config)

        self.P: int = int(config.K)
        self.L: int = int(config.L)
        self.heavy_size = config.heavy_size
        self.tau: float = float(config.tau)

        self._planes_cache: Dict[
            Tuple[int, torch.device, torch.dtype, int, int], torch.Tensor
        ] = {}
        self._protos_cache: Dict[
            Tuple[int, torch.device, torch.dtype], torch.Tensor
        ] = {}

        self._seed = 123456789
        self._rng_cache: Dict[torch.device, torch.Generator] = {}

    def _rng(self, device: torch.device) -> Optional[torch.Generator]:
        if self._seed is None:
            return None
        g = self._rng_cache.get(device)
        if g is None:
            g = torch.Generator(device=device)
            g.manual_seed(self._seed + 7777)
            self._rng_cache[device] = g
        return g

    def add_mask(
        self,
        keys: torch.Tensor,  # [B, H_k or G, N, D]
        queries: torch.Tensor,  # [B, H, Q, D]
        values: torch.Tensor,  # [B, H_k or G, N, Dv]
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float,
        sparse_meta_data: Dict[Any, Any],
        previous_mask: Mask,
        **kwargs: Dict[str, Any],
    ) -> Mask:
        if previous_mask.is_full_mask():
            return previous_mask

        # 1) Align KV to MHA heads if grouped (GQA/MQA)
        ngroups = _get_num_key_value_groups(queries, keys)
        keys_rep = repeat_kv(keys, ngroups)  # [B, H, N, D]
        B, H, N, D = keys_rep.shape
        _, _, Q, _ = queries.shape

        # 2) SRP planes & corners
        planes = get_hyper_planes(
            cache=self._planes_cache,
            D=D,
            L=self.L,
            P=self.P,
            device=keys_rep.device,
            dtype=keys_rep.dtype,
            rng=self._rng(keys_rep.device),
        )  # [L, P, D]

        protosT = get_protos_T(
            cache=self._protos_cache,
            P=self.P,
            device=keys_rep.device,
            dtype=keys_rep.dtype,
        )  # [P, R]

        # 3) Hard-hash keys per table -> [B, H, L, N]
        key_buckets = hard_hash(keys_rep, planes)  # int16 codes

        # 4) Soft-hash queries -> probs [B, H, Q, L, R]
        q_probs = soft_hash(queries, planes, protosT, tau=self.tau)

        # 5) Allowed mask -> allowed_bool [B,H,Q,N]
        allowed_prob = None
        if attention_mask is not None:
            allowed_prob = attention_mask_to_allowed_prob(
                attention_mask, N
            )  # [B,1,*,N]
            allowed_bool = (allowed_prob > 0).expand(B, H, Q, N)
        else:
            allowed_bool = torch.ones(
                (B, H, Q, N), device=keys_rep.device, dtype=torch.bool
            )

        # 6) Budget M
        M = max(0, min(int(self._calculate_effective_size(self.heavy_size, N)), N))
        if M == 0:
            return previous_mask
        Km = min(M, N)

        # 7) Value magnitudes
        v_rep = repeat_kv(
            values, _get_num_key_value_groups(queries, values)
        )  # [B,H,N,Dv]
        v_mag = torch.linalg.vector_norm(v_rep.float(), ord=2, dim=-1).to(
            q_probs.dtype
        )  # [B,H,N]

        # 8) Deterministic top-M by score_j = (sum_l p_l(bucket_l(j))) * ||v_j||
        top_idx, top_scores = topk_soft_collision_scores_blockwise(
            q_probs=q_probs,
            key_buckets=key_buckets,
            v_mag=v_mag,
            allowed_bool=allowed_bool,
            M=Km,
            block=4096,
        )  # [B,H,Q,Km], [B,H,Q,Km]

        # If some rows have <Km allowed tokens, top_scores may be -inf there.
        keep = torch.isfinite(top_scores)  # [B,H,Q,Km]
        if not keep.any():
            return previous_mask

        # 9) Build final mask from top_idx (scatter only kept)
        acc = torch.zeros((B, H, Q, N), device=keys_rep.device, dtype=torch.int16)
        acc.scatter_add_(dim=-1, index=top_idx, src=keep.to(acc.dtype))
        final_mask = acc > 0  # [B,H,Q,N]

        # 10) Merge with previous mask + gate by allowed_prob
        dense_prev = previous_mask.get_dense_mask()
        if not dense_prev.dtype.is_floating_point:
            dense_prev = dense_prev.to(q_probs.dtype)
        dense_prev = dense_prev.clamp_(0.0, 1.0)

        dense_bucket = final_mask.to(dense_prev.dtype)
        dense_mask = torch.maximum(dense_prev, dense_bucket)

        if allowed_prob is not None:
            ap = allowed_prob.to(dense_mask.dtype)  # [B,1,*,N]
            dense_mask = dense_mask * ap.expand_as(dense_mask)

        mask_shape = (B, H, Q, N)
        return Mask.create_mask_from_dense_mask(
            mask_shape, dense_mask, dtype=previous_mask.dtype
        )

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "SocketMasker":
        if not isinstance(config, SocketMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
