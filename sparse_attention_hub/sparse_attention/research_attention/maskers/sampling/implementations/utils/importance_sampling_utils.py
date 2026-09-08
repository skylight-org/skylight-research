"""Importance-sampling draws with per-row inclusion probabilities.

Two ways to spend a sampling budget on a set of proposal logits, both returning
``P(i in S)`` so the caller can build a Horvitz-Thompson estimator:

* :func:`multinomial_with_inclusion` -- exact. ``m`` independent draws with
  replacement from ``p = softmax(s / T)`` (``torch.multinomial``); the unique
  set ``S`` then has the exact closed-form inclusion probability
  ``pi_i = 1 - (1 - p_i) ** m``. Duplicates are not a problem -- they collapse
  into one mask position, which that formula already accounts for.
* :func:`gumbel_topk_with_inclusion` -- the Gumbel-top-k (priority sampling)
  approximation. One ``topk`` over ``s + T * Gumbel(0, 1)``, with
  ``pi_i = 1 - exp(-exp((s_i - kappa) / T))`` for the ``(k+1)``-th perturbed
  order statistic ``kappa``. One kernel cheaper, and the ``k`` keys are
  distinct so the whole budget buys distinct keys; the price is that ``pi_i``
  is exact only conditional on the observed ``kappa``.

Both expect proposal logits on the SAME AXIS as the attention logits, i.e.
``scaling * q . k``. A masker reusing a heavy masker's raw scores (PQCache
publishes raw ``q . k_hat``) must rescale first, or ``temperature`` reads about
``sqrt(head_dim)`` times colder than it looks and the draw degenerates into the
deterministic top-k.

Mask values in this hub are inclusion probabilities, never their reciprocal:
:meth:`Mask.apply_inv_mask` divides by the stored value, and that division is
what produces the Horvitz-Thompson weight ``1 / pi_i``.
"""

from typing import Tuple, Union

import torch

SAMPLING_MODES: Tuple[str, ...] = ("uniform", "gumbel", "multinomial")

# keeps -log(-log(u)) finite at both ends of the uniform sample
_UNIFORM_EPS: float = 1e-6
# floor on pi_i so 1 / pi_i cannot blow up on a near-zero probability
_MIN_INCLUSION_PROBABILITY: float = 1e-4


def _as_budget_tensor(
    k: Union[int, torch.Tensor], scores: torch.Tensor
) -> torch.Tensor:
    """Normalise a scalar or ``[..., 1]`` budget to a ``[..., 1]`` int64 tensor."""
    if isinstance(k, int):
        return torch.full(
            (*scores.shape[:-1], 1), k, device=scores.device, dtype=torch.long
        )
    k_tensor: torch.Tensor = k.to(device=scores.device, dtype=torch.long)
    if k_tensor.shape[-1] != 1:
        k_tensor = k_tensor.unsqueeze(-1)
    return k_tensor


def _empty_result(
    scores: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The no-budget result: nothing drawn, nothing included."""
    return (
        torch.zeros(*scores.shape[:-1], 1, device=scores.device, dtype=torch.long),
        torch.zeros(*scores.shape[:-1], 1, device=scores.device, dtype=scores.dtype),
        torch.zeros(*scores.shape[:-1], 1, device=scores.device, dtype=torch.bool),
        torch.zeros_like(scores),
    )


def sample_gumbel_noise(reference: torch.Tensor) -> torch.Tensor:
    """Draw standard Gumbel(0, 1) noise shaped like ``reference``, in float32."""
    uniform: torch.Tensor = torch.rand(
        reference.shape, device=reference.device, dtype=torch.float32
    ).clamp_(min=_UNIFORM_EPS, max=1.0 - _UNIFORM_EPS)
    return -torch.log(-torch.log(uniform))


def multinomial_with_inclusion(
    scores: torch.Tensor,
    k: Union[int, torch.Tensor],
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact importance sampling: ``m`` i.i.d. draws from ``softmax(s / T)``.

    Args:
        scores: ``[..., n]`` proposal logits. ``-inf`` marks positions that must
            not be sampled (already taken by sink / local / the heavy top-k, or
            killed by the attention mask); they get ``p_i = 0``.
        k: scalar or ``[..., 1]`` per-row number of draws ``m``.
        temperature: softmax temperature. ``0`` degenerates to a deterministic
            top-k with ``pi_i = 1``, i.e. plain truncation with no reweighting.

    Returns:
        indices: ``[..., k_max]`` drawn positions (may repeat).
        inclusion: ``[..., k_max]`` ``pi`` of each draw.
        valid: ``[..., k_max]`` True for real draws (False is padding).
        dense_inclusion: ``[..., n]`` ``pi_i`` on the unique drawn set, else 0.
    """
    num_scored: int = scores.shape[-1]
    logits: torch.Tensor = scores.to(torch.float32)
    live: torch.Tensor = torch.isfinite(logits)
    n_live: torch.Tensor = live.sum(dim=-1, keepdim=True)
    budget: torch.Tensor = _as_budget_tensor(k, scores).clamp(min=0)
    # With replacement there is no reason to cap m at n_live, but a row with
    # nothing live has no distribution to draw from.
    effective_m: torch.Tensor = torch.where(
        n_live > 0, budget, torch.zeros_like(budget)
    )
    k_max: int = int(effective_m.max().item()) if effective_m.numel() else 0
    if k_max <= 0:
        return _empty_result(scores)

    col: torch.Tensor = torch.arange(k_max, device=scores.device).view(
        *([1] * (scores.ndim - 1)), k_max
    )

    if temperature == 0.0:
        indices: torch.Tensor = torch.topk(
            logits, k=min(k_max, num_scored), dim=-1, largest=True
        ).indices
        valid: torch.Tensor = (col < effective_m) & live.gather(dim=-1, index=indices)
        inclusion: torch.Tensor = torch.where(
            valid,
            torch.ones_like(indices, dtype=scores.dtype),
            torch.zeros_like(indices, dtype=scores.dtype),
        )
        return (
            indices,
            inclusion,
            valid,
            _scatter_unique(scores, indices, inclusion, valid),
        )

    masked_logits: torch.Tensor = torch.where(
        live, logits / temperature, torch.full_like(logits, float("-inf"))
    )
    probabilities: torch.Tensor = torch.softmax(masked_logits, dim=-1)
    probabilities = torch.where(
        torch.isfinite(probabilities), probabilities, torch.zeros_like(probabilities)
    )
    row_mass: torch.Tensor = probabilities.sum(dim=-1, keepdim=True)
    # A dead row would make torch.multinomial raise; give it a dummy
    # distribution and drop its draws through `valid`.
    uniform: torch.Tensor = torch.full_like(probabilities, 1.0 / max(num_scored, 1))
    safe: torch.Tensor = torch.where(row_mass > 0, probabilities, uniform)
    indices = torch.multinomial(
        safe.reshape(-1, num_scored), k_max, replacement=True
    ).reshape(*scores.shape[:-1], k_max)
    valid = (col < effective_m) & (row_mass > 0)

    # pi_i = 1 - (1 - p_i)^m, as -expm1(m * log1p(-p)) so a tiny p does not
    # underflow to exactly 0 and blow the 1/pi weight up to infinity.
    pi_all: torch.Tensor = -torch.expm1(
        effective_m.to(torch.float32)
        * torch.log1p((-probabilities.clamp(0.0, 1.0)).clamp(min=-1.0, max=0.0))
    )
    pi_all = pi_all.clamp(min=_MIN_INCLUSION_PROBABILITY, max=1.0)
    inclusion = torch.gather(pi_all, dim=-1, index=indices).to(scores.dtype)
    inclusion = torch.where(valid, inclusion, torch.zeros_like(inclusion))
    return indices, inclusion, valid, _scatter_unique(scores, indices, inclusion, valid)


def gumbel_topk_with_inclusion(
    scores: torch.Tensor,
    k: Union[int, torch.Tensor],
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gumbel-top-k: sample ``k`` DISTINCT keys and return inclusion probs.

    Same signature and return shape as :func:`multinomial_with_inclusion`.

    Args:
        scores: ``[..., n]`` proposal logits; ``-inf`` is never selected.
        k: scalar or ``[..., 1]`` per-row sample count.
        temperature: Gumbel noise scale. ``0`` is a deterministic top-k with
            ``pi_i = 1``.
    """
    num_scored: int = scores.shape[-1]
    logits: torch.Tensor = scores.to(torch.float32)
    live: torch.Tensor = torch.isfinite(logits)
    n_live: torch.Tensor = live.sum(dim=-1, keepdim=True)
    # Without replacement a row can never take more keys than it has live.
    effective_k: torch.Tensor = torch.minimum(
        _as_budget_tensor(k, scores).clamp(min=0), n_live
    )
    k_max: int = int(effective_k.max().item()) if effective_k.numel() else 0
    if k_max <= 0:
        return _empty_result(scores)

    col: torch.Tensor = torch.arange(k_max, device=scores.device).view(
        *([1] * (scores.ndim - 1)), k_max
    )

    if temperature == 0.0:
        indices: torch.Tensor = torch.topk(
            logits, k=min(k_max, num_scored), dim=-1, largest=True
        ).indices
        valid: torch.Tensor = (col < effective_k) & live.gather(dim=-1, index=indices)
        inclusion: torch.Tensor = torch.where(
            valid,
            torch.ones_like(indices, dtype=scores.dtype),
            torch.zeros_like(indices, dtype=scores.dtype),
        )
        return (
            indices,
            inclusion,
            valid,
            _scatter_unique(scores, indices, inclusion, valid),
        )

    perturbed: torch.Tensor = logits + temperature * sample_gumbel_noise(logits)
    topk_k: int = min(k_max + 1, num_scored)
    top_values, top_indices = torch.topk(perturbed, k=topk_k, dim=-1, sorted=True)
    indices = top_indices[..., :k_max]
    # Per-row threshold: the (k_i + 1)-th perturbed order statistic. A row whose
    # k_i saturates its live set has no (k_i + 1)-th draw, so `take_all` below
    # forces pi = 1 and the clamp only keeps the gather in bounds. This must NOT
    # be widened to a whole-tensor -inf fallback -- one saturating row would
    # then set pi = 1 for every other row and silently delete the reweighting.
    threshold: torch.Tensor = torch.gather(
        top_values, dim=-1, index=effective_k.clamp(max=topk_k - 1)
    )
    sampled_logits: torch.Tensor = torch.gather(logits, dim=-1, index=indices)
    exponent: torch.Tensor = (sampled_logits - threshold) / temperature
    inclusion_f32: torch.Tensor = -torch.expm1(-torch.exp(exponent))
    inclusion_f32 = torch.where(
        torch.isfinite(exponent), inclusion_f32, torch.ones_like(inclusion_f32)
    ).clamp(min=_MIN_INCLUSION_PROBABILITY, max=1.0)
    inclusion_f32 = torch.where(
        effective_k >= n_live, torch.ones_like(inclusion_f32), inclusion_f32
    )
    valid = (col < effective_k) & torch.isfinite(sampled_logits)
    inclusion = torch.where(valid, inclusion_f32, torch.zeros_like(inclusion_f32)).to(
        scores.dtype
    )
    return indices, inclusion, valid, _scatter_unique(scores, indices, inclusion, valid)


def _scatter_unique(
    scores: torch.Tensor,
    indices: torch.Tensor,
    inclusion: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Place ``pi_i`` on the unique drawn set, zero elsewhere.

    Uses ``scatter_`` (overwrite), not ``scatter_add_``: the multinomial draw is
    WITH replacement, and a repeated index must contribute its ``pi_i`` once,
    not once per draw. Padded slots are routed to a scratch column so they
    cannot collide with a genuinely selected position 0.
    """
    num_scored: int = scores.shape[-1]
    scratch: torch.Tensor = torch.zeros(
        *scores.shape[:-1], num_scored + 1, device=scores.device, dtype=scores.dtype
    )
    safe_index: torch.Tensor = torch.where(
        valid, indices, torch.full_like(indices, num_scored)
    )
    scratch.scatter_(
        dim=-1,
        index=safe_index,
        src=torch.where(valid, inclusion, torch.zeros_like(inclusion)),
    )
    return scratch[..., :num_scored]
