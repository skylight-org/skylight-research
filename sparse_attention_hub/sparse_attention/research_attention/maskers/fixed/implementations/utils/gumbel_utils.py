"""Categorical-with-replacement sampling and Horvitz-Thompson inclusion.

Used by AdaptiveSampling (vAttention sampling slot) and PQImportance.

PQ scores form a proposal ``p = softmax(s / T)``. ``m`` independent draws
with replacement yield a unique set ``S``. Inclusion probabilities are the
exact with-replacement formula ``pi_i = 1 - (1 - p_i)^m``.

Mask values in this hub are inclusion probabilities: ``apply_inv_mask``
turns them into Horvitz-Thompson weights ``1 / pi_i`` on exact ``q k`` logits.
"""

from typing import Tuple, Union

import torch

# floor on pi_i so 1/pi_i cannot blow up on a near-zero probability
_MIN_INCLUSION_PROBABILITY: float = 1e-4


def _as_budget_tensor(
    k: Union[int, torch.Tensor], scores: torch.Tensor
) -> torch.Tensor:
    if isinstance(k, int):
        return torch.full(
            (*scores.shape[:-1], 1),
            k,
            device=scores.device,
            dtype=torch.long,
        )
    k_tensor = k.to(device=scores.device, dtype=torch.long)
    if k_tensor.shape[-1] != 1:
        k_tensor = k_tensor.unsqueeze(-1)
    return k_tensor


def _horvitz_thompson_inclusion(
    probabilities: torch.Tensor, num_draws: torch.Tensor
) -> torch.Tensor:
    """Exact with-replacement inclusion: ``1 - (1 - p_i)^m``."""
    probabilities_f32: torch.Tensor = probabilities.to(torch.float32).clamp(0.0, 1.0)
    draws_f32: torch.Tensor = num_draws.to(torch.float32)
    # 1 - (1-p)^m = -expm1(m * log1p(-p)); p=1 -> pi=1, p=0 -> pi=0
    inclusion: torch.Tensor = -torch.expm1(
        draws_f32 * torch.log1p((-probabilities_f32).clamp(min=-1.0, max=0.0))
    )
    return inclusion.clamp(min=0.0, max=1.0)


def sample_categorical_inclusion(
    scores: torch.Tensor,
    k: Union[int, torch.Tensor],
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample ``m`` keys with replacement from ``softmax(scores / T)``.

    ``scores`` may contain ``-inf`` for positions that must not be sampled
    (already taken by sink / local / top-k). Those get ``p_i = 0``.

    Args:
        scores: ``[..., n]`` proposal logits (PQ scores or attention logits).
        k: scalar or ``[..., 1]`` number of with-replacement draws ``m``.
        temperature: Softmax temperature. ``0`` is deterministic top-k with
            ``pi_i = 1``.

    Returns:
        indices: ``[..., k_max]`` drawn positions (with replacement).
        inclusion: ``[..., k_max]`` ``pi`` of each draw.
        valid: ``[..., k_max]`` True for real draws (False is padding).
        dense_inclusion: ``[..., n]`` ``pi_i`` on the unique set ``S``, else 0.
    """
    num_scored: int = scores.shape[-1]
    k_tensor: torch.Tensor = _as_budget_tensor(k, scores)
    live: torch.Tensor = torch.isfinite(scores.to(torch.float32))
    n_live: torch.Tensor = live.sum(dim=-1, keepdim=True)
    effective_m: torch.Tensor = torch.where(
        n_live > 0, k_tensor.clamp(min=0), torch.zeros_like(k_tensor)
    )
    k_max: int = int(effective_m.max().item()) if effective_m.numel() else 0

    empty_idx: torch.Tensor = torch.zeros(
        *scores.shape[:-1], max(k_max, 1), device=scores.device, dtype=torch.long
    )
    empty_pi: torch.Tensor = torch.zeros(
        *scores.shape[:-1], max(k_max, 1), device=scores.device, dtype=scores.dtype
    )
    empty_valid: torch.Tensor = torch.zeros(
        *scores.shape[:-1], max(k_max, 1), device=scores.device, dtype=torch.bool
    )
    empty_dense: torch.Tensor = torch.zeros_like(scores)
    if k_max <= 0:
        return empty_idx, empty_pi, empty_valid, empty_dense

    col: torch.Tensor = torch.arange(k_max, device=scores.device).view(
        *([1] * (scores.ndim - 1)), k_max
    )
    valid: torch.Tensor = col < effective_m

    if temperature == 0.0:
        top_k: int = min(k_max, num_scored)
        top_indices = torch.topk(
            scores.to(torch.float32), k=top_k, dim=-1, largest=True
        ).indices
        if top_indices.shape[-1] < k_max:
            pad: int = k_max - top_indices.shape[-1]
            top_indices = torch.cat(
                [
                    top_indices,
                    top_indices[..., :1].expand(*top_indices.shape[:-1], pad),
                ],
                dim=-1,
            )
        valid = (
            valid
            & (col < n_live)
            & live.gather(dim=-1, index=top_indices)
        )
        inclusion = torch.where(
            valid,
            torch.ones_like(top_indices, dtype=scores.dtype),
            torch.zeros_like(top_indices, dtype=scores.dtype),
        )
        hit: torch.Tensor = torch.zeros_like(scores)
        hit.scatter_add_(
            dim=-1,
            index=torch.where(valid, top_indices, torch.zeros_like(top_indices)),
            src=valid.to(scores.dtype),
        )
        dense_inclusion = torch.where(
            hit > 0, torch.ones_like(scores), torch.zeros_like(scores)
        )
        return top_indices, inclusion, valid, dense_inclusion

    logits: torch.Tensor = scores.to(torch.float32) / temperature
    logits = torch.where(live, logits, torch.full_like(logits, float("-inf")))
    probabilities: torch.Tensor = torch.softmax(logits, dim=-1)
    probabilities = torch.where(
        torch.isfinite(probabilities), probabilities, torch.zeros_like(probabilities)
    )
    row_mass: torch.Tensor = probabilities.sum(dim=-1, keepdim=True)
    valid = valid & (row_mass > 0)

    uniform: torch.Tensor = torch.full_like(probabilities, 1.0 / max(num_scored, 1))
    safe_probabilities: torch.Tensor = torch.where(row_mass > 0, probabilities, uniform)
    flat_p: torch.Tensor = safe_probabilities.reshape(-1, num_scored)
    draws: torch.Tensor = torch.multinomial(flat_p, k_max, replacement=True).reshape(
        *scores.shape[:-1], k_max
    )

    pi_all: torch.Tensor = _horvitz_thompson_inclusion(probabilities, effective_m)
    pi_all = pi_all.clamp(min=_MIN_INCLUSION_PROBABILITY, max=1.0)
    sampled_pi: torch.Tensor = torch.gather(pi_all, dim=-1, index=draws).to(scores.dtype)
    sampled_pi = torch.where(
        valid, sampled_pi, torch.zeros_like(sampled_pi, dtype=scores.dtype)
    )

    hit: torch.Tensor = torch.zeros_like(probabilities)
    hit.scatter_add_(
        dim=-1,
        index=torch.where(valid, draws, torch.zeros_like(draws)),
        src=valid.to(probabilities.dtype),
    )
    dense_inclusion: torch.Tensor = torch.where(
        hit > 0, pi_all.to(scores.dtype), torch.zeros_like(scores)
    )
    return draws, sampled_pi, valid, dense_inclusion
