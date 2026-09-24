"""Tests for the ScaNN masker primitives."""

import itertools

import pytest
import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.scann_utils import (
    anisotropic_eta,
    block_order,
    chunk_blocks,
    gather_rows,
    kmeans_l2,
    noise_shaped_codes,
    reconstruct,
    residual_stats,
    squared_distances,
    unit_directions,
)


def _fixture(
    num_groups=2, num_points=200, dim=16, num_leaves=8, num_centers=4, block=2
):
    """Build a full index and its per-block statistics on reproducible random data."""
    torch.manual_seed(0)
    num_blocks = dim // block
    points = torch.randn(num_groups, num_points, dim, dtype=torch.float64)
    centroids, leaves = kmeans_l2(points, num_leaves, 5, 11)
    blocks = chunk_blocks(points - gather_rows(centroids, leaves), num_blocks)
    codebooks, _ = kmeans_l2(
        blocks.transpose(1, 2).reshape(num_groups * num_blocks, num_points, block),
        num_centers,
        5,
        12,
    )
    codebooks = codebooks.view(num_groups, num_blocks, num_centers, block)
    directions = chunk_blocks(unit_directions(points), num_blocks)
    norms, parallel = residual_stats(blocks, directions, codebooks)
    return points, centroids, leaves, codebooks, blocks, directions, norms, parallel


def _totals(norms, parallel, codes):
    """``(P, R)`` for a code assignment."""
    picked = codes.unsqueeze(-1)
    return (
        parallel.gather(-1, picked).squeeze(-1).sum(-1),
        norms.gather(-1, picked).squeeze(-1).sum(-1),
    )


def _cost(norms, parallel, codes, eta):
    total_par, total_norm = _totals(norms, parallel, codes)
    return eta * total_par.pow(2) + (total_norm - total_par.pow(2))


@pytest.mark.unit
class TestAnisotropicEta:
    """The parallel-cost multiplier of ScaNN's noise_shaping_utils.h."""

    def test_matches_published_value(self) -> None:
        """T=0.2 at d=100 is the eta=4.125 quoted in the ScaNN paper."""
        assert anisotropic_eta(0.2, 100) == pytest.approx(4.125)

    def test_closed_form(self) -> None:
        assert anisotropic_eta(0.5, 5) == pytest.approx(4 * 0.25 / 0.75)

    def test_increases_with_threshold_and_dimension(self) -> None:
        etas = [anisotropic_eta(t, 128) for t in (0.1, 0.2, 0.3, 0.4)]
        assert etas == sorted(etas)
        assert anisotropic_eta(0.2, 256) > anisotropic_eta(0.2, 128)


@pytest.mark.unit
class TestSquaredDistances:
    def test_matches_cdist(self) -> None:
        torch.manual_seed(1)
        points = torch.randn(3, 40, 8, dtype=torch.float64)
        centers = torch.randn(3, 5, 8, dtype=torch.float64)
        expected = torch.cdist(points, centers).pow(2)
        assert torch.allclose(squared_distances(points, centers), expected, atol=1e-8)


@pytest.mark.unit
class TestKmeansL2:
    def test_deterministic_for_a_fixed_seed(self) -> None:
        torch.manual_seed(2)
        points = torch.randn(2, 120, 6)
        first = kmeans_l2(points, 7, 5, 99)
        second = kmeans_l2(points, 7, 5, 99)
        assert torch.equal(first[0], second[0])
        assert torch.equal(first[1], second[1])

    def test_different_seeds_can_differ(self) -> None:
        torch.manual_seed(2)
        points = torch.randn(2, 120, 6)
        assert not torch.equal(
            kmeans_l2(points, 7, 1, 1)[0], kmeans_l2(points, 7, 1, 2)[0]
        )

    def test_labels_are_the_nearest_center(self) -> None:
        torch.manual_seed(3)
        points = torch.randn(2, 150, 5, dtype=torch.float64)
        centers, labels = kmeans_l2(points, 6, 8, 5)
        assert torch.equal(labels, squared_distances(points, centers).argmin(-1))

    def test_inertia_is_non_increasing(self) -> None:
        torch.manual_seed(4)
        points = torch.randn(2, 200, 4, dtype=torch.float64)
        inertias = [
            squared_distances(points, kmeans_l2(points, 8, n, 7)[0])
            .min(-1)
            .values.sum()
            for n in range(1, 7)
        ]
        assert all(b <= a + 1e-9 for a, b in zip(inertias, inertias[1:]))

    def test_initialization_samples_without_replacement(self) -> None:
        torch.manual_seed(5)
        points = torch.randn(1, 40, 3, dtype=torch.float64)
        centers, _ = kmeans_l2(points, 40, 0, 3)
        nearest = squared_distances(points, centers).argmin(-1)
        assert torch.equal(nearest.sort(dim=1).values, torch.arange(40).unsqueeze(0))

    def test_empty_clusters_keep_their_center(self) -> None:
        points = torch.zeros(1, 10, 2)
        points[0, :5] = torch.tensor([5.0, 5.0])
        centers, labels = kmeans_l2(points, 4, 5, 1)
        assert torch.isfinite(centers).all()
        assert labels.max().item() < 4


@pytest.mark.unit
class TestResidualStats:
    def test_matches_direct_computation(self) -> None:
        _, _, _, codebooks, blocks, directions, norms, parallel = _fixture()
        diff = blocks.unsqueeze(-2) - codebooks.unsqueeze(1)
        assert torch.allclose(norms, diff.pow(2).sum(-1), atol=1e-8)
        assert torch.allclose(
            parallel, (diff * directions.unsqueeze(-2)).sum(-1), atol=1e-8
        )

    def test_parallel_and_orthogonal_identities(self) -> None:
        """``||r_par||^2 == P^2`` and ``||r_perp||^2 == R - P^2`` for any assignment."""
        points, centroids, leaves, codebooks, _, _, norms, parallel = _fixture()
        torch.manual_seed(7)
        codes = torch.randint(codebooks.shape[2], norms.shape[:-1])
        residual = points - reconstruct(centroids, leaves, codebooks, codes)
        directions = unit_directions(points)
        total_par, total_norm = _totals(norms, parallel, codes)

        projected = (residual * directions).sum(-1, keepdim=True) * directions
        assert torch.allclose(total_par, (residual * directions).sum(-1), atol=1e-9)
        assert torch.allclose(total_norm, residual.pow(2).sum(-1), atol=1e-8)
        assert torch.allclose(projected.pow(2).sum(-1), total_par.pow(2), atol=1e-9)
        assert torch.allclose(
            (residual - projected).pow(2).sum(-1),
            total_norm - total_par.pow(2),
            atol=1e-8,
        )


@pytest.mark.unit
class TestNoiseShapedCodes:
    def test_disabled_returns_the_minimum_residual_assignment(self) -> None:
        *_, norms, parallel = _fixture()
        assert torch.equal(
            noise_shaped_codes(norms, parallel, None, 10, block_order(norms)),
            norms.argmin(-1),
        )
        assert torch.equal(
            noise_shaped_codes(norms, parallel, 5.0, 0, block_order(norms)),
            norms.argmin(-1),
        )

    def test_never_increases_the_anisotropic_cost(self) -> None:
        *_, norms, parallel = _fixture(dim=128, num_centers=16)
        eta = anisotropic_eta(0.2, 128)
        initial = norms.argmin(-1)
        shaped = noise_shaped_codes(
            norms.clone(), parallel.clone(), eta, 10, block_order(norms)
        )
        assert torch.all(
            _cost(norms, parallel, shaped, eta)
            <= _cost(norms, parallel, initial, eta) + 1e-9
        )

    def test_lowers_the_cost_and_the_parallel_error(self) -> None:
        *_, norms, parallel = _fixture(dim=128, num_centers=16)
        eta = anisotropic_eta(0.2, 128)
        initial = norms.argmin(-1)
        shaped = noise_shaped_codes(
            norms.clone(), parallel.clone(), eta, 10, block_order(norms)
        )
        assert not torch.equal(shaped, initial)
        assert (
            _cost(norms, parallel, shaped, eta).sum()
            < _cost(norms, parallel, initial, eta).sum()
        )
        assert (
            _totals(norms, parallel, shaped)[0].pow(2).sum()
            < _totals(norms, parallel, initial)[0].pow(2).sum()
        )

    def test_gate_forbids_growing_the_parallel_error(self) -> None:
        """ScaNN's ``parallel_norm_delta > 0 -> skip`` gate, checked per datapoint."""
        *_, norms, parallel = _fixture(dim=128, num_centers=16)
        eta = anisotropic_eta(0.3, 128)
        initial = norms.argmin(-1)
        shaped = noise_shaped_codes(
            norms.clone(), parallel.clone(), eta, 10, block_order(norms)
        )
        assert torch.all(
            _totals(norms, parallel, shaped)[0].pow(2)
            <= _totals(norms, parallel, initial)[0].pow(2) + 1e-9
        )

    def test_converges_and_stops_early(self) -> None:
        """Coordinate descent must reach a fixed point, not spin for every round."""
        *_, norms, parallel = _fixture(dim=128, num_centers=16)
        eta = anisotropic_eta(0.2, 128)
        order = block_order(norms)
        settled = noise_shaped_codes(norms.clone(), parallel.clone(), eta, 4, order)
        assert torch.equal(
            settled, noise_shaped_codes(norms.clone(), parallel.clone(), eta, 40, order)
        )
        assert not torch.equal(
            settled, noise_shaped_codes(norms.clone(), parallel.clone(), eta, 1, order)
        )

    def test_is_bounded_by_the_brute_force_optimum(self) -> None:
        """Greedy descent must land between the exhaustive optimum and the starting point."""
        *_, norms, parallel = _fixture(dim=6, num_centers=3, block=2)
        eta = 4.0
        num_blocks, num_centers = norms.shape[2], norms.shape[3]
        candidates = torch.tensor(
            list(itertools.product(range(num_centers), repeat=num_blocks))
        )
        expanded = candidates.view(1, 1, -1, num_blocks, 1).expand(
            norms.shape[0], norms.shape[1], -1, -1, -1
        )
        every_norm = norms.unsqueeze(2).expand(-1, -1, len(candidates), -1, -1)
        every_par = parallel.unsqueeze(2).expand(-1, -1, len(candidates), -1, -1)
        total_par = every_par.gather(-1, expanded).squeeze(-1).sum(-1)
        total_norm = every_norm.gather(-1, expanded).squeeze(-1).sum(-1)
        costs = eta * total_par.pow(2) + (total_norm - total_par.pow(2))
        shaped = _cost(
            norms,
            parallel,
            noise_shaped_codes(norms, parallel, eta, 10, block_order(norms)),
            eta,
        )
        assert torch.all(shaped >= costs.min(-1).values - 1e-9)
        assert torch.all(shaped <= _cost(norms, parallel, norms.argmin(-1), eta) + 1e-9)


@pytest.mark.unit
class TestReconstruct:
    def test_matches_an_explicit_per_block_lookup(self) -> None:
        points, centroids, leaves, codebooks, _, _, norms, _ = _fixture()
        codes = norms.argmin(-1)
        num_groups, num_points, num_blocks = codes.shape
        block = codebooks.shape[-1]
        expected = gather_rows(centroids, leaves).clone()
        for group in range(num_groups):
            for blk in range(num_blocks):
                span = slice(blk * block, (blk + 1) * block)
                expected[group, :, span] += codebooks[group, blk][codes[group, :, blk]]
        assert torch.allclose(
            reconstruct(centroids, leaves, codebooks, codes), expected
        )

    def test_scoring_equals_the_lookup_table_sum(self) -> None:
        """Reconstruct-and-matmul is the LUT16 score ScaNN computes block by block."""
        _, centroids, leaves, codebooks, _, _, norms, _ = _fixture()
        codes = norms.argmin(-1)
        torch.manual_seed(9)
        queries = torch.randn(
            centroids.shape[0], 5, centroids.shape[-1], dtype=torch.float64
        )
        approx = torch.bmm(
            queries, reconstruct(centroids, leaves, codebooks, codes).transpose(1, 2)
        )

        blocked = chunk_blocks(queries, codes.shape[-1])
        tables = torch.einsum("gqms,gmcs->gmqc", blocked, codebooks)
        lut = torch.gather(
            tables,
            3,
            codes.permute(0, 2, 1).unsqueeze(2).expand(-1, -1, queries.shape[1], -1),
        ).sum(1)
        residual_bias = torch.bmm(
            queries, gather_rows(centroids, leaves).transpose(1, 2)
        )
        assert torch.allclose(approx, lut + residual_bias, atol=1e-8)


@pytest.mark.unit
class TestChunking:
    def test_chunk_blocks_round_trips(self) -> None:
        vectors = torch.randn(2, 10, 12)
        assert torch.equal(chunk_blocks(vectors, 6).flatten(-2), vectors)

    def test_unit_directions_are_normalized_and_zero_safe(self) -> None:
        vectors = torch.cat([torch.randn(1, 4, 3), torch.zeros(1, 1, 3)], dim=1)
        directions = unit_directions(vectors)
        assert torch.allclose(directions[0, :4].norm(dim=-1), torch.ones(4), atol=1e-6)
        assert torch.all(directions[0, 4] == 0)


@pytest.mark.unit
class TestBlockOrder:
    def test_is_descending_mean_minimum_residual_norm(self) -> None:
        *_, norms, _ = _fixture()
        means = norms.min(-1).values.mean(dim=(0, 1))
        assert means[block_order(norms)].tolist() == sorted(
            means.tolist(), reverse=True
        )

    def test_a_fixed_order_makes_encoding_batch_independent(self) -> None:
        """Codes for a key must not depend on which other keys shared its batch."""
        *_, norms, parallel = _fixture(dim=128, num_centers=16)
        eta = anisotropic_eta(0.2, 128)
        order = block_order(norms)
        whole = noise_shaped_codes(norms.clone(), parallel.clone(), eta, 10, order)
        tail = noise_shaped_codes(
            norms[:, 100:].clone(), parallel[:, 100:].clone(), eta, 10, order
        )
        assert torch.equal(whole[:, 100:], tail)
