"""Sphere codebooks checked with analytic centres and independent Beta moments."""

import math

import numpy as np
import pytest
import torch

from sparse_attention_hub.kv_quantization.quantizers.implementations.turboquant import (
    sphere_lloyd_max,
    turboquant_codebook,
)
from tests.reference.turboquant import conditional_sphere_mean

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_dimension_three_has_exact_uniform_distribution_centres(bits):
    expected = -1 + (2 * torch.arange(2**bits, dtype=torch.float64) + 1) / 2**bits
    torch.testing.assert_close(
        sphere_lloyd_max(3, bits), expected, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("dimension", [2, 3, 5, 32, 128, 256])
def test_one_bit_matches_analytic_gamma_function_centre(dimension):
    centre = math.exp(
        math.lgamma(dimension / 2) - math.lgamma((dimension + 1) / 2)
    ) / math.sqrt(math.pi)
    expected = torch.tensor([-centre, centre], dtype=torch.float64)
    torch.testing.assert_close(
        sphere_lloyd_max(dimension, 1), expected, rtol=2e-11, atol=2e-12
    )


@pytest.mark.parametrize("dimension", [2, 5, 32, 128])
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_lloyd_fixed_point_matches_independent_beta_moments(dimension, bits):
    centres = sphere_lloyd_max(dimension, bits)
    edges = np.r_[-1.0, ((centres[:-1] + centres[1:]) / 2).numpy(), 1.0]
    expected = [
        conditional_sphere_mean(a, b, dimension) for a, b in zip(edges[:-1], edges[1:])
    ]
    torch.testing.assert_close(centres, torch.tensor(expected), rtol=2e-8, atol=2e-10)
    higher_order = sphere_lloyd_max(dimension, bits, quadrature_order=192)
    torch.testing.assert_close(centres, higher_order, rtol=2e-8, atol=2e-10)
    assert torch.equal(centres, -centres.flip(0))
    assert (centres[1:] > centres[:-1]).all()


def test_cached_codebook_inspection_cannot_mutate_future_tables():
    expected = turboquant_codebook(128, 3)
    changed = turboquant_codebook(128, 3)
    changed.zero_()
    assert torch.equal(turboquant_codebook(128, 3), expected)
    assert expected.dtype == torch.float32
    gaussian = turboquant_codebook(128, 3, "gaussian_limit")
    assert not torch.equal(expected, gaussian)
    torch.testing.assert_close(
        expected * math.sqrt(128), gaussian * math.sqrt(128), rtol=0.03, atol=0.02
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"dimension": 1, "bits": 2},
        {"dimension": True, "bits": 2},
        {"dimension": 32, "bits": 0},
        {"dimension": 32, "bits": True},
        {"dimension": 32, "bits": 5},
        {"dimension": 32, "bits": 2, "tolerance": float("nan")},
        {"dimension": 32, "bits": 2, "quadrature_order": 0},
        {"dimension": 32, "bits": 2, "max_iterations": 0},
    ],
)
def test_invalid_solver_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        sphere_lloyd_max(**settings)


def test_nonconvergence_is_not_silently_replaced_with_a_gaussian_table():
    with pytest.raises(RuntimeError, match="converge"):
        sphere_lloyd_max(128, 4, max_iterations=1)
    with pytest.raises(ValueError):
        turboquant_codebook(128, 3, "unknown")
