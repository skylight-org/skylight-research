"""Primitives for the ScaNN masker.

Batched k-means, the anisotropic parallel-cost multiplier and the noise-shaped code
assignment of Guo et al., "Accelerating Large-Scale Inference with Anisotropic Vector
Quantization" (ICML 2020), as implemented in google-research/scann.
"""

from typing import List, Optional, Tuple

import torch

_EPS: float = 1e-12


def anisotropic_eta(threshold: float, dim: int) -> float:
    """Parallel-cost multiplier ``eta = (d - 1) * T^2 / (1 - T^2)``.

    Mirrors ``ComputeParallelCostMultiplier`` in ScaNN's ``noise_shaping_utils.h``.
    ``threshold`` is relative to the datapoint norm, i.e. it is the paper's ``T`` for the
    unit-normalized data ScaNN is configured with.
    """
    squared: float = threshold * threshold
    return (dim - 1) * squared / (1.0 - squared)


def squared_distances(points: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Squared L2 between ``[G, N, D]`` points and ``[G, K, D]`` centers -> ``[G, N, K]``."""
    return torch.baddbmm(
        centers.pow(2).sum(-1).unsqueeze(1),
        points,
        centers.transpose(1, 2),
        alpha=-2.0,
    ).add_(points.pow(2).sum(-1, keepdim=True))


def kmeans_l2(
    points: torch.Tensor, num_clusters: int, num_iters: int, seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched Lloyd k-means under squared L2 with seeded random initialization.

    Args:
        points: ``[G, N, D]``. ``N`` must be at least ``num_clusters``.

    Returns:
        ``(centers [G, K, D], labels [G, N])``. Empty clusters keep their previous center.
    """
    num_groups, num_points, dim = points.shape
    generator: torch.Generator = torch.Generator(device=points.device)
    generator.manual_seed(seed)
    draw: torch.Tensor = torch.rand(
        num_groups, num_points, generator=generator, device=points.device
    )
    init: torch.Tensor = draw.topk(num_clusters, dim=1).indices
    centers: torch.Tensor = points.gather(1, init.unsqueeze(-1).expand(-1, -1, dim))

    for _ in range(num_iters):
        labels: torch.Tensor = squared_distances(points, centers).argmin(-1)
        counts: torch.Tensor = torch.zeros(
            num_groups, num_clusters, dtype=points.dtype, device=points.device
        ).scatter_add_(1, labels, torch.ones_like(points[..., 0]))
        sums: torch.Tensor = torch.zeros_like(centers).scatter_add_(
            1, labels.unsqueeze(-1).expand(-1, -1, dim), points
        )
        means: torch.Tensor = sums / counts.clamp(min=1.0).unsqueeze(-1)
        centers = torch.where(counts.unsqueeze(-1) > 0, means, centers)

    return centers, squared_distances(points, centers).argmin(-1)


def gather_rows(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Row gather along dim 1: ``[G, K, D]`` by ``[G, N]`` -> ``[G, N, D]``."""
    return source.gather(1, index.long().unsqueeze(-1).expand(-1, -1, source.shape[-1]))


def residual_stats(
    residuals: torch.Tensor, directions: torch.Tensor, codebooks: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-block statistics of every candidate codeword.

    Args:
        residuals: ``[G, N, M, S]`` vectors being quantized.
        directions: ``[G, N, M, S]`` unit datapoint directions, chunked identically.
        codebooks: ``[G, M, C, S]``.

    Returns:
        ``(norms, parallel)``, both ``[G, N, M, C]``: the block's contribution to
        ``||z - z_tilde||^2`` and to ``<z - z_tilde, u>``.
    """
    norms: torch.Tensor = (
        residuals.pow(2).sum(-1, keepdim=True)
        - 2.0 * torch.einsum("gnms,gmcs->gnmc", residuals, codebooks)
        + codebooks.pow(2).sum(-1).unsqueeze(1)
    )
    parallel: torch.Tensor = (residuals * directions).sum(
        -1, keepdim=True
    ) - torch.einsum("gnms,gmcs->gnmc", directions, codebooks)
    return norms, parallel


def block_order(norms: torch.Tensor) -> List[int]:
    """Blocks sorted by descending mean minimum residual norm."""
    return norms.min(-1).values.mean(dim=(0, 1)).argsort(descending=True).tolist()


def noise_shaped_codes(
    norms: torch.Tensor,
    parallel: torch.Tensor,
    eta: Optional[float],
    num_rounds: int,
    order: List[int],
) -> torch.Tensor:
    """Assign PQ codes minimizing ``eta * ||r_par||^2 + ||r_perp||^2``.

    With ``P = sum_m parallel[m, A_m]`` and ``R = sum_m norms[m, A_m]`` the identities
    ``||r_par||^2 = P^2`` and ``||r_perp||^2 = R - P^2`` hold exactly, so the objective is
    ``(eta - 1) P^2 + R``. Starting from the plain minimum-residual assignment, blocks are
    swept in ``order``; a move is taken only when it lowers the total cost and does not
    increase ``P^2`` -- the gate in ScaNN's ``OptimizeSingleSubspace``.

    ``order`` fixes the sweep, so a key's code never depends on which other keys shared
    its batch. The incumbent's delta is pinned to exactly zero -- recomputing it in
    floating point leaves a residue that would otherwise register as an improvement and
    stop the sweep from ever terminating early. ``eta is None`` reproduces ScaNN's
    ``noise_shaping_threshold = NaN``.
    """
    codes: torch.Tensor = norms.argmin(-1)
    if eta is None or num_rounds <= 0:
        return codes

    picked_norm: torch.Tensor = norms.gather(-1, codes.unsqueeze(-1)).squeeze(-1)
    picked_par: torch.Tensor = parallel.gather(-1, codes.unsqueeze(-1)).squeeze(-1)
    total_par: torch.Tensor = picked_par.sum(-1)

    for _ in range(num_rounds):
        changed: torch.Tensor = torch.zeros((), dtype=torch.bool, device=codes.device)
        for block in order:
            rest: torch.Tensor = total_par - picked_par[..., block]
            delta_par: torch.Tensor = (rest.unsqueeze(-1) + parallel[:, :, block]).pow(
                2
            ) - total_par.pow(2).unsqueeze(-1)
            delta_norm: torch.Tensor = norms[:, :, block] - picked_norm[
                ..., block
            ].unsqueeze(-1)
            delta_cost: torch.Tensor = torch.where(
                delta_par > 0.0,
                torch.inf,
                eta * delta_par + delta_norm - delta_par,
            ).scatter_(-1, codes[..., block].unsqueeze(-1), 0.0)
            best: torch.Tensor = delta_cost.argmin(-1)
            improved: torch.Tensor = (
                delta_cost.gather(-1, best.unsqueeze(-1)).squeeze(-1) < 0.0
            )
            new_par: torch.Tensor = (
                parallel[:, :, block].gather(-1, best.unsqueeze(-1)).squeeze(-1)
            )
            new_norm: torch.Tensor = (
                norms[:, :, block].gather(-1, best.unsqueeze(-1)).squeeze(-1)
            )
            codes[..., block] = torch.where(improved, best, codes[..., block])
            picked_par[..., block] = torch.where(
                improved, new_par, picked_par[..., block]
            )
            picked_norm[..., block] = torch.where(
                improved, new_norm, picked_norm[..., block]
            )
            total_par = torch.where(improved, rest + new_par, total_par)
            changed = changed | improved.any()
        if not bool(changed):
            break

    return codes


def chunk_blocks(vectors: torch.Tensor, num_blocks: int) -> torch.Tensor:
    """Reshape ``[G, N, D]`` into ``[G, N, M, D / M]``."""
    return vectors.view(*vectors.shape[:-1], num_blocks, -1)


def unit_directions(vectors: torch.Tensor) -> torch.Tensor:
    """``x / ||x||`` with a zero-norm guard, same shape as the input."""
    return vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(_EPS)


def reconstruct(
    centroids: torch.Tensor,
    leaves: torch.Tensor,
    codebooks: torch.Tensor,
    codes: torch.Tensor,
) -> torch.Tensor:
    """Rebuild ``c_pi(x) + concat_m codebook[m, code_m]`` -> ``[G, N, D]``.

    Scoring against this reconstruction is algebraically identical to ScaNN's per-block
    lookup tables, and avoids materializing a ``[G, M, Q, N]`` table gather.
    """
    num_groups, num_points, num_blocks = codes.shape
    block_dim: int = codebooks.shape[-1]
    picked: torch.Tensor = codebooks.gather(
        2, codes.long().transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, block_dim)
    )
    residual: torch.Tensor = picked.transpose(1, 2).reshape(
        num_groups, num_points, num_blocks * block_dim
    )
    return residual + gather_rows(centroids, leaves)
