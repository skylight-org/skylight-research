"""Tests for the torch NVFP4 fake-quantization kernel.

All CPU, no modelopt: this is the part that can be checked properly anywhere,
so it carries the bulk of the numerics coverage. The cases are the ones that
catch real mistakes — the element grid, E4M3 saturation, the tiny-block
underflow that the 2^-9 block-scale floor exists to prevent, and wide dynamic range.
"""

from typing import List

import pytest
import torch

from sparse_attention_hub.kv_quantization.constants import (
    E2M1_MAX,
    E4M3_MAX,
    E4M3_MIN_SUBNORMAL,
)
from sparse_attention_hub.kv_quantization.nvfp4 import (
    bits_per_element,
    fake_quantize_nvfp4,
    q_e2m1,
    q_e4m3,
    quantize_nvfp4,
)

E2M1_GRID: List[float] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _relative_error(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Relative Frobenius error between a tensor and its round trip."""
    difference: torch.Tensor = reconstructed.float() - original.float()
    return float(
        (
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(original.float())
        ).item()
    )


@pytest.mark.unit
class TestElementFormat:
    """q_e2m1: the 4-bit element type."""

    def test_grid_is_exactly_the_eight_values(self) -> None:
        """Nothing outside {0, .5, 1, 1.5, 2, 3, 4, 6} may come out."""
        x: torch.Tensor = torch.linspace(-6.0, 6.0, 401)
        produced: List[float] = sorted({abs(v) for v in q_e2m1(x).tolist()})
        assert produced == pytest.approx(E2M1_GRID)

    @pytest.mark.parametrize(
        "value,expected",
        [
            (0.0, 0.0),
            (0.24, 0.0),
            (0.26, 0.5),
            (0.9, 1.0),
            (1.2, 1.0),
            (1.3, 1.5),
            (2.4, 2.0),
            (2.6, 3.0),
            (3.9, 4.0),
        ],
    )
    def test_rounds_to_nearest_grid_point(self, value: float, expected: float) -> None:
        """Rounding picks the nearest representable magnitude."""
        assert q_e2m1(torch.tensor([value])).item() == pytest.approx(expected)

    @pytest.mark.parametrize("value,expected", [(5.0, 4.0), (0.25, 0.0), (2.5, 2.0)])
    def test_breaks_ties_to_even(self, value: float, expected: float) -> None:
        """Exact midpoints round to the even code, as the hardware does.

        5.0 sits exactly between 4 and 6, so "nearest" is ambiguous and the tie
        rule decides. Getting this wrong is invisible on random data and shows
        up as a systematic bias on data that lands on midpoints.
        """
        assert q_e2m1(torch.tensor([value])).item() == pytest.approx(expected)

    def test_saturates_instead_of_overflowing(self) -> None:
        """Out-of-range magnitudes clamp to 6, they do not go to infinity."""
        assert q_e2m1(torch.tensor([9.0, -9.0, 1e9])).abs().max().item() == E2M1_MAX

    def test_sign_is_preserved(self) -> None:
        """Negative inputs stay negative."""
        assert q_e2m1(torch.tensor([-2.4])).item() == pytest.approx(-2.0)


@pytest.mark.unit
class TestScaleFormat:
    """q_e4m3: the 8-bit per-block scale."""

    def test_tops_out_at_448_not_480(self) -> None:
        """The all-ones exponent with mantissa 111 is reserved for NaN."""
        assert q_e4m3(torch.tensor([1000.0])).item() == E4M3_MAX

    def test_representable_values_survive(self) -> None:
        """Powers of two and exact mantissas must round trip unchanged."""
        exact: torch.Tensor = torch.tensor([0.25, 1.0, 1.125, 2.0, 28.0, 448.0])
        assert torch.equal(q_e4m3(exact), exact)

    def test_smallest_subnormal_survives(self) -> None:
        """2^-9 is the floor on block scales, so it must be exact."""
        assert q_e4m3(torch.tensor([E4M3_MIN_SUBNORMAL])).item() == E4M3_MIN_SUBNORMAL

    def test_subnormals_are_multiples_of_2_to_the_minus_9(self) -> None:
        """Below 2^-6 the spacing is a fixed 2^-9 (subnormal E4M3)."""
        subnormals: torch.Tensor = torch.arange(1, 8) * E4M3_MIN_SUBNORMAL
        assert torch.equal(q_e4m3(subnormals), subnormals)


@pytest.mark.unit
class TestRoundTrip:
    """fake_quantize_nvfp4 end to end."""

    def test_shape_and_dtype_are_preserved(self) -> None:
        """Fake quantization is in-place in format terms: same shape, same dtype."""
        x: torch.Tensor = torch.randn(2, 4, 32, 64, dtype=torch.bfloat16)
        out: torch.Tensor = fake_quantize_nvfp4(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_all_zeros_stay_all_zeros(self) -> None:
        """A tensor with no magnitude has no scale, and must not produce NaN."""
        out: torch.Tensor = fake_quantize_nvfp4(torch.zeros(4, 32))
        assert torch.all(out == 0)

    def test_never_produces_nan_or_inf(self) -> None:
        """The 2^-9 floor on block scales stops a tiny block underflowing to 0/0.

        A block whose amax is negligible next to the tensor amax is exactly the
        case that used to divide by zero, so it is the case worth asserting.
        """
        x: torch.Tensor = torch.cat(
            [torch.full((1, 16), 1e4), torch.full((1, 16), 1e-30)], dim=1
        )
        out: torch.Tensor = fake_quantize_nvfp4(x)
        assert torch.isfinite(out).all()

    def test_tiny_block_gets_the_floor_scale(self) -> None:
        """A block tiny next to the tensor max gets block scale 2^-9 (modelopt's rule)."""
        x: torch.Tensor = torch.cat(
            [torch.full((1, 16), 1e4), torch.full((1, 16), 1e-6)], dim=1
        )
        _, block_scale, _ = quantize_nvfp4(x)
        assert block_scale.flatten().tolist() == [448.0, E4M3_MIN_SUBNORMAL]

    def test_all_zero_block_gets_scale_one_and_zero_codes(self) -> None:
        """An all-zero block next to a non-zero one: scale 1, every code 0."""
        x: torch.Tensor = torch.cat([torch.randn(1, 16), torch.zeros(1, 16)], dim=1)
        codes, block_scale, _ = quantize_nvfp4(x)
        assert block_scale.flatten()[1].item() == 1.0
        assert torch.all(codes[0, 1] == 0)

    def test_values_on_the_grid_survive_scaling(self) -> None:
        """A block of a single power of two is exactly representable."""
        x: torch.Tensor = torch.full((1, 32), 2.0)
        assert torch.allclose(fake_quantize_nvfp4(x), x)

    def test_error_is_small_for_gaussian_data(self) -> None:
        """A loose bound that a broken scale or axis lands well outside."""
        x: torch.Tensor = torch.randn(
            8, 128, generator=torch.Generator().manual_seed(0)
        )
        assert _relative_error(x, fake_quantize_nvfp4(x)) < 0.15

    def test_survives_wide_dynamic_range(self) -> None:
        """Per-block scales are the whole point: each block keeps its own range.

        Plain gaussian data passes even when the scale math is wrong, so the
        magnitudes-spanning-many-decades case is the one that tells you
        something.
        """
        generator: torch.Generator = torch.Generator().manual_seed(1)
        exponents: torch.Tensor = torch.randint(-6, 6, (16, 64), generator=generator)
        x: torch.Tensor = torch.randn(16, 64, generator=generator) * (
            10.0 ** exponents.float()
        )
        out: torch.Tensor = fake_quantize_nvfp4(x)
        assert torch.isfinite(out).all()
        assert _relative_error(x, out) < 0.25

    def test_pads_a_last_axis_that_does_not_divide(self) -> None:
        """The token axis is arbitrary once transposed, so padding must work."""
        x: torch.Tensor = torch.randn(2, 40)
        out: torch.Tensor = fake_quantize_nvfp4(x, block_size=16)
        assert out.shape == x.shape
        assert _relative_error(x, out) < 0.2

    def test_smaller_blocks_are_more_accurate(self) -> None:
        """Fewer elements per shared scale means less range to compromise on."""
        generator: torch.Generator = torch.Generator().manual_seed(2)
        x: torch.Tensor = torch.randn(8, 256, generator=generator)
        x[:, ::32] *= 50.0  # something for the scale to have to accommodate
        assert _relative_error(x, fake_quantize_nvfp4(x, 8)) < _relative_error(
            x, fake_quantize_nvfp4(x, 64)
        )

    def test_requantizing_costs_almost_nothing(self) -> None:
        """A second pass barely moves the tensor.

        Not exactly idempotent: the per-tensor scale is derived from the amax,
        which shifts slightly once the tensor is on the grid, and every block
        scale is relative to it. But a second pass finding real error would mean
        the codes are not landing where the scales say they are.
        """
        x: torch.Tensor = torch.randn(4, 64, generator=torch.Generator().manual_seed(3))
        once: torch.Tensor = fake_quantize_nvfp4(x)
        assert _relative_error(once, fake_quantize_nvfp4(once)) < 0.01


@pytest.mark.unit
class TestBitsPerElement:
    """The memory the real backend would save, reported by the fake one."""

    def test_counts_codes_scales_and_the_global_scale(self) -> None:
        """4 bits per element, one 8-bit scale per block, one 32-bit per tensor."""
        assert bits_per_element(numel=1024, block_size=16) == pytest.approx(
            4.0 + 0.5 + 32.0 / 1024
        )


@pytest.mark.unit
class TestExactExponentHelpers:
    """The rounding reads exponents from float32 bits instead of calling
    log2/exp2, so the rounding step is exact on every device (CPU, NVIDIA, AMD, Apple)."""

    def test_floor_log2_is_exact_at_and_around_powers_of_two(self) -> None:
        from sparse_attention_hub.kv_quantization.nvfp4 import _floor_log2

        exponents: torch.Tensor = torch.arange(-20, 20)
        powers: torch.Tensor = torch.pow(2.0, exponents.float())
        assert torch.equal(_floor_log2(powers), exponents.to(torch.int32))
        just_below: torch.Tensor = torch.nextafter(powers, torch.zeros_like(powers))
        assert torch.equal(_floor_log2(just_below), (exponents - 1).to(torch.int32))

    def test_pow2_is_exact(self) -> None:
        from sparse_attention_hub.kv_quantization.nvfp4 import _pow2

        exponents: torch.Tensor = torch.arange(-12, 12, dtype=torch.int32)
        assert torch.equal(_pow2(exponents), torch.pow(2.0, exponents.float()))
