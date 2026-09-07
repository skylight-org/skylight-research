"""Gumbel-top-k sampling with per-row inclusion probabilities.

Used by AdaptiveSampling (vAttention sampling slot) and PQImportance.
Mask values in this hub are inclusion probabilities: ``apply_inv_mask``
turns them into Horvitz-Thompson weights ``1 / pi_i``.
"""

from typing import Tuple, Union

import torch

# keeps -log(-log(u)) finite at both ends of the uniform sample
_UNIFORM_EPS: float = 1e-6
# floor on pi_i so 1/pi_i cannot blow up on a near-zero probability
_MIN_INCLUSION_PROBABILITY: float = 1e-4


def sample_gumbel_noise(reference: torch.Tensor) -> torch.Tensor:
    """Draw standard Gumbel(0, 1) noise shaped like ``reference``, in float32."""
    uniform: torch.Tensor = torch.rand(
        reference.shape, device=reference.device, dtype=torch.float32
    ).clamp_(min=_UNIFORM_EPS, max=1.0 - _UNIFORM_EPS)
    return -torch.log(-torch.log(uniform))


def gumbel_topk_with_inclusion(
    scores: torch.Tensor,
    k: Union[int, torch.Tensor],
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample without replacement via Gumbel-top-k and return inclusion probs.

    ``scores`` may contain ``-inf`` for positions that must not be sampled
    (already taken by sink / local / top-k). Those are never selected.

    Args:
        scores: ``[..., n]`` logits (PQ scores or attention logits).
        k: scalar or ``[..., 1]`` per-row sample count.
        temperature: Gumbel noise scale. ``0`` is deterministic top-k with
            ``pi_i = 1``.

    Returns:
        indices: ``[..., k_max]`` positions in the last dim of ``scores``.
        inclusion: ``[..., k_max]`` estimated ``P(i selected)``, in ``(0, 1]``.
        valid: ``[..., k_max]`` True for real draws (False is padding).
    """
    num_scored: int = scores.shape[-1]
    if isinstance(k, int):
        k_tensor: torch.Tensor = torch.full(
            (*scores.shape[:-1], 1),
            k,
            device=scores.device,
            dtype=torch.long,
        )
    else:
        k_tensor = k.to(device=scores.device, dtype=torch.long)
        if k_tensor.shape[-1] != 1:
            k_tensor = k_tensor.unsqueeze(-1)

    live: torch.Tensor = torch.isfinite(scores.to(torch.float32))
    n_live: torch.Tensor = live.sum(dim=-1, keepdim=True)
    effective_k: torch.Tensor = torch.minimum(k_tensor, n_live)
    k_max: int = int(effective_k.max().item()) if effective_k.numel() else 0

    empty_idx: torch.Tensor = torch.zeros(
        *scores.shape[:-1], max(k_max, 1), device=scores.device, dtype=torch.long
    )
    empty_pi: torch.Tensor = torch.ones(
        *scores.shape[:-1], max(k_max, 1), device=scores.device, dtype=scores.dtype
    )
    empty_valid: torch.Tensor = torch.zeros(
        *scores.shape[:-1], max(k_max, 1), device=scores.device, dtype=torch.bool
    )
    if k_max <= 0:
        return empty_idx, empty_pi, empty_valid

    take_all: torch.Tensor = effective_k >= n_live

    logits: torch.Tensor = scores.to(torch.float32)
    if temperature == 0.0:
        top_indices = torch.topk(
            logits, k=min(k_max, num_scored), dim=-1, largest=True
        ).indices
        top_values = torch.gather(logits, dim=-1, index=top_indices)
        # pad if min(k_max, n) < k_max (should not happen: k_max <= n_live <= n)
        if top_indices.shape[-1] < k_max:
            pad = k_max - top_indices.shape[-1]
            top_indices = torch.cat(
                [top_indices, top_indices[..., :1].expand(*top_indices.shape[:-1], pad)],
                dim=-1,
            )
            top_values = torch.cat(
                [top_values, top_values[..., :1].expand(*top_values.shape[:-1], pad)],
                dim=-1,
            )
        inclusion = torch.ones_like(top_indices, dtype=scores.dtype)
    else:
        perturbed: torch.Tensor = logits + temperature * sample_gumbel_noise(logits)
        # (k+1)-st perturbed value is the Gumbel-top-k inclusion threshold.
        topk_k: int = min(k_max + 1, num_scored)
        top_values, top_indices = torch.topk(
            perturbed, k=topk_k, dim=-1, sorted=True
        )
        selected_indices: torch.Tensor = top_indices[..., :k_max]
        if topk_k > k_max:
            # per-row threshold = (k_i + 1)-th order statistic
            threshold_index: torch.Tensor = effective_k.clamp(max=topk_k - 1)
            threshold: torch.Tensor = torch.gather(top_values, dim=-1, index=threshold_index)
        else:
            # no (k+1)th sample exists globally; treat as fully selected
            threshold = torch.full(
                (*scores.shape[:-1], 1),
                float("-inf"),
                device=scores.device,
                dtype=torch.float32,
            )

        sampled_logits: torch.Tensor = torch.gather(
            logits, dim=-1, index=selected_indices
        )
        exponent: torch.Tensor = (sampled_logits - threshold) / temperature
        inclusion_f32: torch.Tensor = -torch.expm1(-torch.exp(exponent))
        inclusion_f32 = torch.where(
            torch.isfinite(exponent),
            inclusion_f32,
            torch.ones_like(inclusion_f32),
        ).clamp(min=_MIN_INCLUSION_PROBABILITY, max=1.0)
        inclusion = inclusion_f32.to(scores.dtype)
        top_indices = selected_indices
        top_values = sampled_logits

    col: torch.Tensor = torch.arange(k_max, device=scores.device).view(
        *([1] * (scores.ndim - 1)), k_max
    )
    valid: torch.Tensor = (col < effective_k) & torch.isfinite(top_values.to(torch.float32))
    inclusion = torch.where(take_all, torch.ones_like(inclusion), inclusion)
    inclusion = torch.where(valid, inclusion, torch.zeros_like(inclusion))
    return top_indices, inclusion, valid
