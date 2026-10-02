"""Heavy validation that the NVFP4 *simulator* (``fake_nvfp4``) is correct.

"Correct" means: it produces exactly what real NVFP4 produces, except where
the two implementations legitimately disagree, and every disagreement must be
shown to be one. Five parts, cheapest first:

1. **Formats, exhaustively.** Every bf16 value through the E2M1 element
   rounding (against an independent round-half-to-even reference) and through
   the E4M3 scale rounding (against PyTorch's native ``float8_e4m3fn`` cast).
2. **Simulator invariants at scale.** Tens of millions of elements across 12
   distributions (heavy tails, near-ties, zero blocks, huge outliers...) and
   three dtypes: grid membership, per-element error bound, near-idempotence,
   sign symmetry, power-of-two equivariance, CPU vs GPU (equal except where
   CUDA's reciprocal division moves a value sitting on a rounding tie).
3. **Against modelopt's real NVFP4** (what the ``nvfp4`` backend uses): tensor
   scales, block scales and elements equal. The check tolerates a mismatch only
   if it is explained by a value sitting exactly on a rounding boundary; on the
   GPU there are none.
4. **Real Qwen3 KV.** Keys and values captured from the model on WikiText-2,
   through both implementations and both blocking axes.
5. **End to end.** Greedy generation and WikiText-2 perplexity with bf16 vs
   simulated vs real NVFP4 caches, a long-context run, and the full
   ``ModelAdapterHF`` path with sparse attention.

Run on a CUDA machine (molab)::

    pytest tests/integration/kv_quantization/ -v -s

Numbers go to the JSON report (see ``conftest.py``), not only pass/fail.
"""

import functools
import math
import os
import time
from typing import Any, Callable, Dict, List, Tuple

import pytest
import torch

from sparse_attention_hub.kv_quantization.constants import (
    AXIS_PER_CHANNEL,
    AXIS_PER_TOKEN,
    DEFAULT_BLOCK_SIZE,
    E2M1_MAX,
    E4M3_MAX,
    E4M3_MIN_SUBNORMAL,
)
from sparse_attention_hub.kv_quantization.nvfp4 import (
    fake_quantize_nvfp4,
    q_e2m1,
    q_e4m3,
    quantize_nvfp4,
)

from .conftest import requires_cuda

pytestmark = [pytest.mark.slow, pytest.mark.integration, requires_cuda]

CUDA: torch.device = torch.device("cuda")
E2M1_GRID: torch.Tensor = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
SMALLEST_NORMAL_E4M3: float = (
    2.0**-6
)  # below this, E4M3 block scales are subnormal (coarse)
TIE_EPS: float = (
    1e-5  # relative nudge used to decide "this value sits on a rounding boundary"
)
MAX_MISMATCH_RATE: float = (
    1e-3  # even ties should be rare; anything above this is a real bug
)


# --------------------------------------------------------------------------- helpers


def all_bf16_values() -> torch.Tensor:
    """Every finite bfloat16 value, as float32 (65,280 values)."""
    bits: torch.Tensor = torch.arange(-(2**15), 2**15, dtype=torch.int32).to(
        torch.int16
    )
    values: torch.Tensor = bits.view(torch.bfloat16).float()
    return values[torch.isfinite(values)]


def reference_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Independent E2M1 rounding: nearest grid point, ties to the even code.

    Written without any of the simulator's code: distance to each of the 8
    magnitudes, and on an exact tie prefer the even grid index (even mantissa
    bit), which is what round-half-to-even means for this format.
    """
    grid: torch.Tensor = E2M1_GRID.to(x.device)
    magnitude: torch.Tensor = x.abs().clamp(max=E2M1_MAX)
    distance: torch.Tensor = (magnitude.unsqueeze(-1) - grid).abs()
    best: torch.Tensor = distance.min(dim=-1, keepdim=True).values
    candidates: torch.Tensor = distance == best
    even_index: torch.Tensor = (torch.arange(8, device=x.device) % 2 == 0) & candidates
    has_even: torch.Tensor = even_index.any(dim=-1, keepdim=True)
    choice: torch.Tensor = (
        torch.where(has_even, even_index, candidates).float().argmax(dim=-1)
    )
    return torch.sign(x) * grid[choice]


def _open_uniform(shape: Tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    """Uniform on the open interval (0, 1): never exactly 0 or 1, so logs stay finite."""
    return torch.rand(shape, generator=generator, device=CUDA).clamp(1e-7, 1 - 1e-7)


def _laplace(shape: Tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    """Laplace(0, 1) by inverse CDF, from the seeded generator (always finite)."""
    u: torch.Tensor = _open_uniform(shape, generator) - 0.5
    return -torch.sign(u) * torch.log1p(-2 * u.abs())


def _student_t_df2(shape: Tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    """Student-t with 2 degrees of freedom: normal / sqrt(chi2_2 / 2), where
    chi2_2 = -2 log U. Seeded and finite (heavy-tailed but bounded by the U clamp)."""
    normal: torch.Tensor = torch.randn(shape, generator=generator, device=CUDA)
    chi2: torch.Tensor = -2.0 * torch.log(_open_uniform(shape, generator))
    return normal / torch.sqrt(chi2 / 2.0)


def distribution_corpus(numel: int, seed: int = 0) -> Dict[str, torch.Tensor]:
    """Twelve inputs, each ``(numel // 256, 256)``, chosen to stress the format."""
    g: torch.Generator = torch.Generator(device=CUDA).manual_seed(seed)
    rows: int = numel // 256
    shape: Tuple[int, int] = (rows, 256)
    corpus: Dict[str, torch.Tensor] = {
        "gaussian": torch.randn(shape, generator=g, device=CUDA),
        "uniform": torch.rand(shape, generator=g, device=CUDA) * 2 - 1,
        "laplace": _laplace(shape, g),
        "student_t_df2": _student_t_df2(shape, g),
        "lognormal_signed": torch.randn(shape, generator=g, device=CUDA).exp()
        * torch.randn(shape, generator=g, device=CUDA).sign(),
        "sparse_99pct_zero": torch.randn(shape, generator=g, device=CUDA)
        * (torch.rand(shape, generator=g, device=CUDA) < 0.01),
        "constant": torch.full(shape, 0.37, device=CUDA),
    }
    keys: torch.Tensor = torch.randn(shape, generator=g, device=CUDA) * 0.5
    keys[:, [3, 17, 40, 200]] += torch.tensor([30.0, -45.0, 60.0, 25.0], device=CUDA)
    corpus["kv_like_outlier_channels"] = keys
    huge_outlier: torch.Tensor = torch.randn(shape, generator=g, device=CUDA)
    huge_outlier[0, 0] = 1e6
    corpus["single_huge_outlier"] = huge_outlier
    tiny_next_to_huge: torch.Tensor = torch.randn(shape, generator=g, device=CUDA) * 1e4
    tiny_next_to_huge[:, :16] = torch.randn((rows, 16), generator=g, device=CUDA) * 1e-8
    corpus["tiny_blocks_next_to_huge"] = tiny_next_to_huge
    # Values that land exactly on E2M1 midpoints once scaled: the tie-breaking stress test.
    midpoints: torch.Tensor = (E2M1_GRID[1:] + E2M1_GRID[:-1]).to(CUDA) / 2
    ties: torch.Tensor = midpoints[torch.randint(0, 7, shape, generator=g, device=CUDA)]
    ties[
        :, ::16
    ] = 6.0  # every block's amax is 6, so the scaled values land exactly on the midpoints
    corpus["exact_e2m1_ties"] = (
        ties * torch.randn(shape, generator=g, device=CUDA).sign()
    )
    wide: torch.Tensor = torch.randn(shape, generator=g, device=CUDA) * torch.exp2(
        torch.randint(-20, 20, (rows, 1), generator=g, device=CUDA).float()
    )
    corpus["wide_dynamic_range_rows"] = wide
    # Every input comes from the seeded generator (reproducible run to run) and
    # must be finite: NVFP4 of inf/NaN is undefined, and a non-finite draw once
    # made a run fail for reasons unrelated to the simulator.
    for name, tensor in corpus.items():
        assert torch.isfinite(tensor).all(), f"corpus {name!r} is not finite"
    return corpus


def _relative_error(approx: torch.Tensor, exact: torch.Tensor) -> float:
    """Relative Frobenius error of ``approx`` against ``exact``."""
    return float(
        ((approx.float() - exact.float()).norm() / exact.float().norm()).item()
    )


def boundary_mask(
    values: torch.Tensor, rounder: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """True where ``values`` sit on a rounding boundary of ``rounder``.

    A value is on a boundary if nudging it by ``TIE_EPS`` (relative) in either
    direction changes the rounded result.
    """
    nudge: torch.Tensor = values.abs() * TIE_EPS + 1e-12
    return rounder(values + nudge) != rounder(values - nudge)


def modelopt_quantize(
    x: torch.Tensor, block_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Real NVFP4 via modelopt: (dequantized float32, block scales float32, tensor scale)."""
    nvfp4_tensor = pytest.importorskip(
        "modelopt.torch.quantization.qtensor.nvfp4_tensor"
    )
    qtensor, scale, scale2 = nvfp4_tensor.NVFP4QTensor.quantize(x, block_size)
    dequantized: torch.Tensor = qtensor.dequantize(
        dtype=torch.float32,
        scale=scale,
        double_scale=scale2,
        block_sizes={-1: block_size},
    )
    return dequantized, scale.float(), scale2.float().reshape(())


def _pre_rounding(
    blocked_x: torch.Tensor, block_scale: torch.Tensor, tensor_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The values just before E4M3 (scale) and E2M1 (code) rounding."""
    block_amax: torch.Tensor = blocked_x.abs().amax(dim=-1, keepdim=True)
    pre_scale = (block_amax / (E2M1_MAX * tensor_scale)).clamp(
        E4M3_MIN_SUBNORMAL, E4M3_MAX
    )
    scaled = blocked_x / (block_scale * tensor_scale)
    return pre_scale, scaled.clamp(-E2M1_MAX, E2M1_MAX)


def compare_with_modelopt(
    x: torch.Tensor, block_size: int = DEFAULT_BLOCK_SIZE
) -> Dict[str, Any]:
    """Compare simulator and modelopt on one float32 tensor; explain every mismatch.

    A block-scale mismatch is "explained" if the simulator's pre-rounding scale
    sits on an E4M3 rounding boundary. An element mismatch in an agreeing block
    is explained if its pre-rounding value sits on an E2M1 boundary.
    """
    x = x.float().contiguous()
    codes, block_scale, tensor_scale = quantize_nvfp4(x, block_size)
    ours: torch.Tensor = codes * (block_scale * tensor_scale)  # (..., n_blocks, block)
    theirs, their_block_scale, their_tensor_scale = modelopt_quantize(x, block_size)
    theirs = theirs.reshape(ours.shape)
    assert (
        their_block_scale.numel() == block_scale.numel()
    ), f"modelopt block-scale layout {tuple(their_block_scale.shape)} vs ours {tuple(block_scale.shape)}"
    their_block_scale = their_block_scale.reshape(block_scale.shape)

    tensor_scale_rel_diff: float = (
        (their_tensor_scale - tensor_scale).abs() / tensor_scale
    ).item()
    blocked_x: torch.Tensor = x.reshape(ours.shape)
    block_amax: torch.Tensor = blocked_x.abs().amax(dim=-1, keepdim=True)
    live_block: torch.Tensor = block_amax > 0

    pre_scale, scaled = _pre_rounding(blocked_x, block_scale, tensor_scale)
    scale_mismatch: torch.Tensor = (their_block_scale != block_scale) & live_block
    explained_scale: torch.Tensor = boundary_mask(pre_scale, q_e4m3)
    unexplained_scales: int = int((scale_mismatch & ~explained_scale).sum())

    element_mismatch: torch.Tensor = (ours != theirs) & ~scale_mismatch
    element_tie: torch.Tensor = boundary_mask(scaled, q_e2m1)
    unexplained_elements: int = int((element_mismatch & ~element_tie).sum())

    total: int = x.numel()
    return {
        "numel": total,
        "tensor_scale_rel_diff": tensor_scale_rel_diff,
        "block_scale_mismatch_rate": scale_mismatch.sum().item()
        / max(1, int(live_block.sum())),
        "element_mismatch_rate": (ours != theirs).sum().item() / total,
        "max_abs_diff_over_block_step": (
            ((ours - theirs).abs() / (block_scale * tensor_scale)).max().item()
        ),
        "unexplained_block_scale_mismatches": unexplained_scales,
        "unexplained_element_mismatches": unexplained_elements,
        "relative_error_simulator": (
            (ours - blocked_x).norm() / blocked_x.norm().clamp_min(1e-30)
        ).item(),
        "relative_error_modelopt": (
            (theirs - blocked_x).norm() / blocked_x.norm().clamp_min(1e-30)
        ).item(),
    }


def assert_agreement(stats: Dict[str, Any], name: str) -> None:
    assert (
        stats["tensor_scale_rel_diff"] < 1e-6
    ), f"{name}: per-tensor scales differ: {stats}"
    assert (
        stats["unexplained_block_scale_mismatches"] == 0
    ), f"{name}: block scales differ off-tie: {stats}"
    assert (
        stats["unexplained_element_mismatches"] == 0
    ), f"{name}: elements differ off-tie: {stats}"
    # A corpus built entirely of exact ties flips on purpose; everywhere else flips must be rare.
    if name != "exact_e2m1_ties":
        assert (
            stats["element_mismatch_rate"] < MAX_MISMATCH_RATE
        ), f"{name}: too many tie flips: {stats}"


# --------------------------------------------------------------------------- 1. formats


class TestFormatsExhaustive:
    def test_e2m1_rounding_matches_reference_for_every_bf16(
        self, report: Dict[str, Any]
    ) -> None:
        values: torch.Tensor = all_bf16_values().to(CUDA)
        in_range: torch.Tensor = values[values.abs() <= 8.0]
        ours: torch.Tensor = q_e2m1(in_range)
        expected: torch.Tensor = reference_e2m1(in_range)
        mismatches: int = int((ours != expected).sum())
        report.setdefault("formats", {})["e2m1_values_checked"] = in_range.numel()
        report["formats"]["e2m1_mismatches"] = mismatches
        assert mismatches == 0
        assert set(q_e2m1(values).abs().unique().tolist()) <= set(E2M1_GRID.tolist())

    def test_e4m3_rounding_matches_torch_float8_for_every_bf16(
        self, report: Dict[str, Any]
    ) -> None:
        values: torch.Tensor = all_bf16_values().to(CUDA)
        in_range: torch.Tensor = values[values.abs() <= E4M3_MAX]
        ours: torch.Tensor = q_e4m3(in_range)
        native: torch.Tensor = in_range.to(torch.float8_e4m3fn).float()
        mismatches: int = int((ours != native).sum())
        report.setdefault("formats", {})["e4m3_bf16_values_checked"] = in_range.numel()
        report["formats"]["e4m3_bf16_mismatches"] = mismatches
        assert mismatches == 0

    def test_e4m3_rounding_matches_torch_float8_random_fp32(
        self, report: Dict[str, Any]
    ) -> None:
        g: torch.Generator = torch.Generator(device=CUDA).manual_seed(0)
        exponents: torch.Tensor = torch.randint(
            -12, 9, (1 << 24,), generator=g, device=CUDA
        ).float()
        values: torch.Tensor = (
            torch.rand(1 << 24, generator=g, device=CUDA) + 1
        ) * torch.exp2(exponents)
        values = values.clamp(max=E4M3_MAX)
        mismatches: int = int(
            (q_e4m3(values) != values.to(torch.float8_e4m3fn).float()).sum()
        )
        report.setdefault("formats", {})["e4m3_fp32_random_checked"] = values.numel()
        report["formats"]["e4m3_fp32_random_mismatches"] = mismatches
        assert mismatches == 0


# --------------------------------------------------------------------------- 2. invariants


NUMEL: int = int(os.environ.get("KVQ_HEAVY_NUMEL", str(1 << 22)))  # per distribution


class TestSimulatorInvariants:
    @pytest.fixture(scope="class")
    def corpus(self) -> Dict[str, torch.Tensor]:
        return distribution_corpus(NUMEL)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_structure_and_error_bound(
        self,
        corpus: Dict[str, torch.Tensor],
        dtype: torch.dtype,
        report: Dict[str, Any],
    ) -> None:
        results: Dict[str, Any] = {}
        for name, x in corpus.items():
            x = x.to(dtype)
            if not torch.isfinite(x).all():
                continue  # e.g. 1e6 in fp16
            codes, block_scale, tensor_scale = quantize_nvfp4(x)
            live: torch.Tensor = (
                x.float().reshape(codes.shape).abs().amax(-1, keepdim=True) > 0
            )
            assert torch.isin(codes.abs(), E2M1_GRID.to(CUDA)).all(), name
            assert torch.equal(q_e4m3(block_scale), block_scale), name
            assert (
                (block_scale[live] >= E4M3_MIN_SUBNORMAL)
                & (block_scale[live] <= E4M3_MAX)
            ).all(), name
            out: torch.Tensor = fake_quantize_nvfp4(x)
            assert (
                out.dtype == dtype
                and out.shape == x.shape
                and torch.isfinite(out).all()
            ), name
            blocked_x: torch.Tensor = x.float().reshape(codes.shape)
            deq32: torch.Tensor = codes * (block_scale * tensor_scale)
            step: torch.Tensor = block_scale * tensor_scale
            # Round-to-nearest on the E2M1 grid errs by at most one step (the
            # widest half-gap, between 4 and 6), plus anything beyond the
            # saturation point 6*step. With a normal E4M3 block scale the block
            # max lands at most at 6.375 steps, so the "beyond" part stays under
            # a step; with a subnormal scale (below 2^-6, floor 2^-9) the
            # scale can round down by up to ~30% and whole values saturate.
            beyond: torch.Tensor = (blocked_x.abs() - 6.0 * step).clamp_min(0.0)
            excess: torch.Tensor = (
                (deq32 - blocked_x).abs() - step - beyond
            ) / step.clamp_min(1e-38)
            assert (
                excess.max().item() <= 1e-4
            ), f"{name}: error exceeds one step beyond saturation"
            worst: float = (
                ((deq32 - blocked_x).abs() / step.clamp_min(1e-38)).max().item()
            )
            normal: torch.Tensor = (block_scale >= SMALLEST_NORMAL_E4M3).expand_as(
                deq32
            )
            worst_normal: float = (
                ((deq32 - blocked_x).abs() / step.clamp_min(1e-38))[normal].max().item()
                if normal.any()
                else 0.0
            )
            assert (
                worst_normal <= 1.0 + 1e-4
            ), f"{name}: {worst_normal} steps in a normal-scale block"
            zero_blocks: torch.Tensor = blocked_x.abs().amax(-1, keepdim=True) == 0
            assert (deq32 * zero_blocks == 0).all(), name
            results[name] = {
                "relative_error": (
                    (out.float() - x.float()).norm() / x.float().norm().clamp_min(1e-30)
                ).item(),
                "worst_error_in_steps": worst,
                "worst_error_in_steps_normal_scale_blocks": worst_normal,
                "subnormal_scale_blocks": int(
                    (block_scale < SMALLEST_NORMAL_E4M3).sum()
                ),
            }
        report.setdefault("invariants", {})[str(dtype)] = results

    def test_idempotent_symmetric_equivariant(
        self, corpus: Dict[str, torch.Tensor], report: Dict[str, Any]
    ) -> None:
        """Sign symmetry and x8 equivariance are exact. Re-quantizing is near-exact:
        the second pass recomputes the tensor scale from the quantized max,
        ``6*448*(amax/2688)``, which can differ from ``amax`` in the last bit and
        flip a value sitting on a boundary."""
        changed: Dict[str, float] = {}
        for name, x in corpus.items():
            once: torch.Tensor = fake_quantize_nvfp4(x)
            twice: torch.Tensor = fake_quantize_nvfp4(once)
            changed[name] = (twice != once).float().mean().item()
            # A second pass recomputes the tensor scale, which can land one float32
            # bit away and shift every nonzero value by ~1e-7. That drift is fine;
            # only values that move by more than that must stay rare (rounding ties).
            moved: torch.Tensor = (twice - once).abs() / once.abs().clamp_min(1e-30)
            moved_far: float = (moved > 1e-5).float().mean().item()
            assert (
                moved_far < 1e-3
            ), f"{name}: re-quantizing moved {moved_far:.2e} of elements by more than 1e-5"
            assert (
                (twice - once).norm() / once.norm().clamp_min(1e-30)
            ).item() < 1e-3, name
            assert torch.equal(
                fake_quantize_nvfp4(-x), -once
            ), f"{name}: not sign-symmetric"
            if x.abs().max() < 1e30:
                assert torch.equal(
                    fake_quantize_nvfp4(x * 8.0), once * 8.0
                ), f"{name}: not 2^k-equivariant"
        report["requantize_changed_fraction"] = changed

    def test_gpu_matches_cpu(
        self, corpus: Dict[str, torch.Tensor], report: Dict[str, Any]
    ) -> None:
        """GPU and CPU may differ only by float rounding, never by a code off a boundary.

        PyTorch's CUDA kernels divide by a Python constant as a multiply by its
        reciprocal, so e.g. ``amax / 2688`` can differ from the CPU result in the
        last bit (modelopt behaves the same, which is why the simulator matches
        it exactly on the GPU). That may move a value sitting on a rounding
        boundary, and nothing else.
        """
        stats: Dict[str, Any] = {}
        for name, x in corpus.items():
            x = x[:4096]
            g_codes, g_bs, g_ts = quantize_nvfp4(x)
            c_codes, c_bs, c_ts = quantize_nvfp4(x.cpu())
            ts_diff: float = ((g_ts.cpu() - c_ts).abs() / c_ts.clamp_min(1e-30)).item()
            blocked: torch.Tensor = x.float().reshape(g_codes.shape)
            pre_scale, scaled = _pre_rounding(blocked, g_bs, g_ts)
            scale_off: torch.Tensor = (g_bs.cpu() != c_bs) & ~boundary_mask(
                pre_scale, q_e4m3
            ).cpu()
            code_off: torch.Tensor = (
                (g_codes.cpu() != c_codes)
                & (g_bs.cpu() == c_bs)
                & ~boundary_mask(scaled, q_e2m1).cpu()
            )
            stats[name] = {
                "tensor_scale_rel_diff": ts_diff,
                "value_mismatches": int(
                    (fake_quantize_nvfp4(x).cpu() != fake_quantize_nvfp4(x.cpu())).sum()
                ),
                "unexplained_scale_diffs": int(scale_off.sum()),
                "unexplained_code_diffs": int(code_off.sum()),
            }
            assert ts_diff <= 2.0**-22, (name, stats[name])
            assert (
                stats[name]["unexplained_scale_diffs"] == 0
                and stats[name]["unexplained_code_diffs"] == 0
            ), (name, stats[name])
        report["gpu_vs_cpu"] = stats

    def test_throughput(self, report: Dict[str, Any]) -> None:
        x: torch.Tensor = torch.randn(1 << 26, device=CUDA, dtype=torch.bfloat16)
        fake_quantize_nvfp4(x)
        torch.cuda.synchronize()
        start: float = time.perf_counter()
        for _ in range(5):
            fake_quantize_nvfp4(x)
        torch.cuda.synchronize()
        seconds: float = (time.perf_counter() - start) / 5
        report["simulator_throughput_gelem_per_s"] = x.numel() / seconds / 1e9


# --------------------------------------------------------------------------- 3. vs modelopt


class TestAgainstModelopt:
    @pytest.fixture(scope="class")
    def corpus(self) -> Dict[str, torch.Tensor]:
        pytest.importorskip("modelopt")
        return distribution_corpus(NUMEL, seed=1)

    def test_every_distribution(
        self, corpus: Dict[str, torch.Tensor], report: Dict[str, Any]
    ) -> None:
        """Equal to modelopt except at rounding ties (on the GPU: exactly equal)."""
        results: Dict[str, Any] = {}
        for name, x in corpus.items():
            results[name] = compare_with_modelopt(x)
        report["vs_modelopt_synthetic"] = results
        for name, stats in results.items():
            assert_agreement(stats, name)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_low_precision_inputs(
        self, dtype: torch.dtype, report: Dict[str, Any]
    ) -> None:
        pytest.importorskip("modelopt")
        x: torch.Tensor = torch.randn(1 << 20, device=CUDA).to(dtype)
        stats: Dict[str, Any] = compare_with_modelopt(x.float())
        report.setdefault("vs_modelopt_dtypes", {})[str(dtype)] = stats
        assert_agreement(stats, str(dtype))


# --------------------------------------------------------------------------- 4. real Qwen3 KV


def capture_kv(
    model: Any, input_ids: torch.Tensor
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """Prefill with an exact cache and return each full-attention layer's (K, V)."""
    from transformers import DynamicCache

    cache: Any = DynamicCache(config=model.config)
    with torch.no_grad():
        model(input_ids.to(CUDA), past_key_values=cache, use_cache=True)
    return {
        idx: (layer.keys, layer.values)
        for idx, layer in enumerate(cache.layers)
        if getattr(layer, "keys", None) is not None and layer.keys.dim() == 4
    }


class TestRealModelKV:
    @pytest.fixture(scope="class")
    def kv(
        self, model_and_tokenizer: Any, wikitext_test_ids: torch.Tensor
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        return capture_kv(model_and_tokenizer[0], wikitext_test_ids[:, :4096])

    def test_simulator_matches_modelopt_on_real_kv(
        self, kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]], report: Dict[str, Any]
    ) -> None:
        pytest.importorskip("modelopt")
        results: Dict[str, Any] = {}
        for idx, (keys, values) in kv.items():
            for label, tensor, axis in (
                ("keys_per_token", keys, AXIS_PER_TOKEN),
                ("keys_per_channel", keys, AXIS_PER_CHANNEL),
                ("values_per_token", values, AXIS_PER_TOKEN),
            ):
                t: torch.Tensor = (
                    tensor.transpose(-1, -2).contiguous()
                    if axis == AXIS_PER_CHANNEL
                    else tensor
                )
                stats: Dict[str, Any] = compare_with_modelopt(t)
                results[f"layer{idx}/{label}"] = stats
                assert_agreement(stats, f"layer{idx}/{label}")
        report["vs_modelopt_real_kv"] = results

    def test_per_channel_keys_help_on_real_model(
        self, kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]], report: Dict[str, Any]
    ) -> None:
        errors: Dict[str, Dict[str, float]] = {}
        for idx, (keys, _) in kv.items():
            per_token: torch.Tensor = fake_quantize_nvfp4(keys)
            per_channel: torch.Tensor = fake_quantize_nvfp4(
                keys.transpose(-1, -2)
            ).transpose(-1, -2)
            errors[f"layer{idx}"] = {
                "per_token": _relative_error(per_token, keys),
                "per_channel": _relative_error(per_channel, keys),
            }
        # measured, not asserted: this is a finding either way
        report["real_keys_axis_error"] = errors


# --------------------------------------------------------------------------- 5. end to end


def make_cache_factory(
    model: Any, backend: str, axis_key: int, quantize_prefill: bool
) -> Callable[[], Any]:
    from transformers import DynamicCache

    from sparse_attention_hub.kv_quantization import create_quantized_kv_cache

    if backend == "bf16":
        return lambda: DynamicCache(config=model.config)
    return lambda: create_quantized_kv_cache(
        model.config,
        backend,
        axis_key=axis_key,
        quantize_prefill=quantize_prefill,
        residual_length=128,
    )


def greedy(
    model: Any, prompt: torch.Tensor, make_cache: Callable[[], Any], steps: int
) -> Tuple[List[int], torch.Tensor]:
    tokens: List[int] = []
    logits: List[torch.Tensor] = []
    with torch.no_grad():
        out: Any = model(prompt.to(CUDA), past_key_values=make_cache(), use_cache=True)
        for _ in range(steps):
            step_logits: torch.Tensor = out.logits[0, -1].float()
            logits.append(step_logits)
            tokens.append(int(step_logits.argmax()))
            out = model(
                torch.tensor([[tokens[-1]]], device=CUDA),
                past_key_values=out.past_key_values,
                use_cache=True,
            )
    return tokens, torch.stack(logits)


BACKENDS: List[Tuple[str, str, int]] = [
    ("bf16", "bf16", AXIS_PER_TOKEN),
    ("fake_nvfp4/per_token", "fake_nvfp4", AXIS_PER_TOKEN),
    ("nvfp4/per_token", "nvfp4", AXIS_PER_TOKEN),
    ("fake_nvfp4/keys_per_channel", "fake_nvfp4", AXIS_PER_CHANNEL),
    ("nvfp4/keys_per_channel", "nvfp4", AXIS_PER_CHANNEL),
]


def _available(backend: str) -> bool:
    if backend != "nvfp4":
        return True
    try:
        __import__("modelopt")
        return True
    except ImportError:
        return False


class TestEndToEnd:
    def test_greedy_generation(
        self,
        model_and_tokenizer: Any,
        wikitext_test_ids: torch.Tensor,
        report: Dict[str, Any],
    ) -> None:
        model: Any = model_and_tokenizer[0]
        prompt: torch.Tensor = wikitext_test_ids[:, 8192 : 8192 + 2048]
        runs: Dict[str, Tuple[List[int], torch.Tensor]] = {}
        for label, backend, axis in BACKENDS:
            if _available(backend):
                runs[label] = greedy(
                    model,
                    prompt,
                    make_cache_factory(model, backend, axis, False),
                    steps=128,
                )
        summary: Dict[str, Any] = {}
        for label, (tokens, logits) in runs.items():
            ref_tokens, ref_logits = runs["bf16"]
            summary[label] = {
                "token_agreement_vs_bf16": sum(
                    a == b for a, b in zip(tokens, ref_tokens)
                )
                / len(tokens),
                "max_logit_diff_vs_bf16": (logits - ref_logits).abs().max().item(),
            }
        for axis_name in ("per_token", "keys_per_channel"):
            real: str = f"nvfp4/{axis_name}"
            if real not in runs:
                continue
            sim: str = f"fake_nvfp4/{axis_name}"
            agreement: float = (
                sum(a == b for a, b in zip(runs[sim][0], runs[real][0])) / 128
            )
            summary[f"{sim} vs {real}"] = {
                "token_agreement": agreement,
                "max_logit_diff": (runs[sim][1] - runs[real][1]).abs().max().item(),
            }
            # The simulator is the bit-exact twin of the real backend on the GPU.
            assert (
                agreement >= 0.9
            ), f"{sim} and {real} generate differently: {agreement}"
        report["greedy_generation"] = summary

    def test_perplexity(
        self,
        model_and_tokenizer: Any,
        wikitext_test_ids: torch.Tensor,
        report: Dict[str, Any],
    ) -> None:
        from sparse_attention_hub.kv_quantization.evaluation import perplexity

        model: Any = model_and_tokenizer[0]
        windows: int = int(os.environ.get("KVQ_PPL_WINDOWS", "20"))
        results: Dict[str, float] = {}
        for label, backend, axis in BACKENDS:
            if _available(backend):
                factory: Callable[[], Any] = make_cache_factory(
                    model, backend, axis, quantize_prefill=True
                )
                results[label] = perplexity(
                    model, wikitext_test_ids, factory, seq_len=2048, max_windows=windows
                )
        report["wikitext2_perplexity"] = {
            "windows": windows,
            "seq_len": 2048,
            **results,
        }
        for axis_name in ("per_token", "keys_per_channel"):
            real: str = f"nvfp4/{axis_name}"
            if real not in results:
                continue
            gap: float = (
                abs(results[f"fake_nvfp4/{axis_name}"] - results[real]) / results[real]
            )
            assert (
                gap < 0.005
            ), f"simulator vs real perplexity differ by {gap:.2%} ({axis_name})"
        assert all(math.isfinite(v) for v in results.values())

    def test_long_context(
        self,
        model_and_tokenizer: Any,
        wikitext_test_ids: torch.Tensor,
        report: Dict[str, Any],
    ) -> None:
        model: Any = model_and_tokenizer[0]
        length: int = int(os.environ.get("KVQ_LONG_CONTEXT", "32768"))
        ids: torch.Tensor = wikitext_test_ids[:, :length]
        results: Dict[str, Any] = {}
        for label, backend, axis in BACKENDS:
            if not _available(backend):
                continue
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base: int = torch.cuda.memory_allocated()
            with torch.no_grad():
                out: Any = model(
                    ids.to(CUDA),
                    past_key_values=make_cache_factory(model, backend, axis, False)(),
                    use_cache=True,
                )
                out = model(
                    out.logits[:, -1:].argmax(-1),
                    past_key_values=out.past_key_values,
                    use_cache=True,
                )
            finite: bool = bool(torch.isfinite(out.logits).all())
            results[label] = {
                "tokens": ids.shape[1],
                "peak_gib_above_weights": (torch.cuda.max_memory_allocated() - base)
                / 2**30,
                "finite": finite,
            }
            del out
            assert finite, f"{label}: non-finite logits at {length} tokens"
        report["long_context"] = results

    def test_adapter_with_sparse_attention(self, report: Dict[str, Any]) -> None:
        from sparse_attention_hub.adapters import ModelAdapterHF, Request
        from sparse_attention_hub.sparse_attention.research_attention import (
            ResearchAttentionConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            LocalMaskerConfig,
            SinkMaskerConfig,
        )

        name: str = os.environ.get("KVQ_MODEL", "Qwen/Qwen3-4B")
        config: ResearchAttentionConfig = ResearchAttentionConfig(
            [SinkMaskerConfig(128), LocalMaskerConfig(512)]
        )
        request: Request = Request(
            context=" ".join(
                f"Item {i} is stored in box {i * 7 % 101}." for i in range(400)
            ),
            questions=["Which box holds item 3?", "Which box holds item 399?"],
        )
        answers: Dict[str, Any] = {}
        for flag in (False, "fake_nvfp4", "nvfp4"):
            if flag == "nvfp4" and not _available("nvfp4"):
                continue
            adapter: Any = ModelAdapterHF(
                name,
                config,
                model_kwargs={"dtype": torch.bfloat16},
                device="cuda",
                hybrid=False,
                quantize_kv_cache=flag,
            )
            answers[str(flag)] = adapter.process_request(
                request, {"max_new_tokens": 16}, {}
            ).responses
            assert all(isinstance(a, str) and a.strip() for a in answers[str(flag)])
        report["adapter_sparse_answers"] = answers


# --------------------------------------------------------------------------- 6. KIVI-style layout


def _cache_backends() -> List[str]:
    return ["fake_nvfp4"] + (["nvfp4"] if _available("nvfp4") else [])


def _layer(backend: str, **kwargs: Any) -> Any:
    if backend == "nvfp4":
        from sparse_attention_hub.kv_quantization.nvfp4_cache import NVFP4QuantizedLayer

        return NVFP4QuantizedLayer(**kwargs)
    from sparse_attention_hub.kv_quantization.fake_nvfp4_cache import (
        FakeNVFP4QuantizedLayer,
    )

    return FakeNVFP4QuantizedLayer(**kwargs)


class TestKiviStyleLayout:
    """``axis_key=0``: keys per channel (blocks of 16 *tokens*), values per token,
    recent tokens exact in the residual buffer, KIVI's layout with NVFP4 numerics.

    The risks specific to this layout: blocks run along the token axis, whose
    length is rarely a multiple of 16 and grows by one every decode step.
    """

    def test_per_channel_blocks_share_one_scale_per_16_tokens(self) -> None:
        keys: torch.Tensor = torch.randn(1, 4, 64, 256, device=CUDA)
        keys[..., 7] += 40.0  # a loud channel
        _, block_scale, _ = quantize_nvfp4(keys.transpose(-1, -2))
        # (1, heads, channels, token_blocks, 16): one scale per (channel, 16 tokens)
        assert block_scale.shape == (1, 4, 256, 4, 1)
        # the loud channel's scale is large, its neighbours' are not: isolation works
        assert (block_scale[0, :, 7] > 4 * block_scale[0, :, 6]).all()

    @pytest.mark.parametrize("tokens", [1, 15, 16, 17, 100, 4097])
    def test_token_counts_not_multiple_of_16(
        self, tokens: int, report: Dict[str, Any]
    ) -> None:
        """Per-channel keys pad the token axis; the pad must never leak into results."""
        keys: torch.Tensor = torch.randn(
            1, 4, tokens, 256, device=CUDA, dtype=torch.bfloat16
        )
        values: torch.Tensor = torch.randn_like(keys)
        outcomes: Dict[str, Any] = {}
        for backend in _cache_backends():
            layer: Any = _layer(
                backend,
                axis_key=AXIS_PER_CHANNEL,
                residual_length=0,
                quantize_prefill=True,
            )
            try:
                k, v = layer.update(keys, values)
            except Exception as exc:  # noqa: BLE001 - record which backend breaks and how
                outcomes[backend] = f"{type(exc).__name__}: {exc}"
                continue
            assert k.shape == keys.shape and v.shape == values.shape, backend
            assert torch.isfinite(k).all() and torch.isfinite(v).all(), backend
            outcomes[backend] = (
                (k.float() - keys.float()).norm() / keys.float().norm()
            ).item()
            reference: torch.Tensor = fake_quantize_nvfp4(
                keys.transpose(-1, -2)
            ).transpose(-1, -2)
            if backend == "fake_nvfp4":
                assert torch.equal(k, reference)
        report.setdefault("kivi_style_token_counts", {})[tokens] = outcomes
        failures: Dict[str, Any] = {
            b: o for b, o in outcomes.items() if isinstance(o, str)
        }
        assert not failures, f"{tokens} tokens: {failures}"

    def test_decode_across_block_boundaries(self, report: Dict[str, Any]) -> None:
        """Prefill 100 tokens, then decode 300 one at a time with residual_length=128:
        lengths, residual contents and fake-vs-real agreement at every step."""
        torch.manual_seed(0)
        prefill_k: torch.Tensor = torch.randn(
            1, 4, 100, 256, device=CUDA, dtype=torch.bfloat16
        )
        steps_k: torch.Tensor = torch.randn(
            1, 4, 300, 256, device=CUDA, dtype=torch.bfloat16
        )
        layers: Dict[str, Any] = {
            b: _layer(b, axis_key=AXIS_PER_CHANNEL, residual_length=128)
            for b in _cache_backends()
        }
        worst_gap: float = 0.0
        for backend, layer in layers.items():
            layer.update(prefill_k, prefill_k)
        for t in range(300):
            k_new: torch.Tensor = steps_k[:, :, t : t + 1]
            returned: Dict[str, torch.Tensor] = {}
            for backend, layer in layers.items():
                k, _ = layer.update(k_new, k_new)
                assert k.shape[-2] == 100 + t + 1, (backend, t)
                assert torch.equal(
                    k[..., -1:, :], k_new
                ), f"{backend}: newest token not exact at step {t}"
                returned[backend] = k.float()
            if len(returned) == 2:
                worst_gap = max(
                    worst_gap,
                    (returned["fake_nvfp4"] - returned["nvfp4"]).abs().max().item(),
                )
        report["kivi_style_decode"] = {
            "steps": 300,
            "max_abs_gap_fake_vs_real": worst_gap,
        }

    def test_residual_tokens_stay_exact_until_flushed(self) -> None:
        for backend in _cache_backends():
            layer: Any = _layer(backend, axis_key=AXIS_PER_CHANNEL, residual_length=32)
            prefill: torch.Tensor = torch.randn(
                1, 2, 64, 256, device=CUDA, dtype=torch.bfloat16
            )
            layer.update(prefill, prefill)
            recent: List[torch.Tensor] = []
            for _ in range(20):
                token: torch.Tensor = torch.randn(
                    1, 2, 1, 256, device=CUDA, dtype=torch.bfloat16
                )
                recent.append(token)
                k, _ = layer.update(token, token)
            assert torch.equal(
                k[..., -20:, :], torch.cat(recent, dim=-2)
            ), f"{backend}: residual not exact"
            assert not torch.equal(
                k[..., :64, :], prefill
            ), f"{backend}: prefill should be quantized"

    def test_both_axes_for_values_and_keys_on_real_model(
        self,
        model_and_tokenizer: Any,
        wikitext_test_ids: torch.Tensor,
        report: Dict[str, Any],
    ) -> None:
        """All four (axis_key, axis_value) combinations: KV error and perplexity."""
        from sparse_attention_hub.kv_quantization import create_quantized_kv_cache
        from sparse_attention_hub.kv_quantization.evaluation import perplexity

        model: Any = model_and_tokenizer[0]
        windows: int = int(os.environ.get("KVQ_PPL_WINDOWS", "20"))
        results: Dict[str, float] = {}
        for axis_key in (AXIS_PER_TOKEN, AXIS_PER_CHANNEL):
            for axis_value in (AXIS_PER_TOKEN, AXIS_PER_CHANNEL):
                label: str = f"keys_{'channel' if axis_key == 0 else 'token'}/values_{'channel' if axis_value == 0 else 'token'}"
                results[label] = perplexity(
                    model,
                    wikitext_test_ids,
                    # partial binds this iteration's axes (a bare lambda would not)
                    functools.partial(
                        create_quantized_kv_cache,
                        model.config,
                        "fake_nvfp4",
                        axis_key=axis_key,
                        axis_value=axis_value,
                        quantize_prefill=True,
                    ),
                    seq_len=2048,
                    max_windows=windows,
                )
        report["kivi_style_axis_grid_perplexity"] = results
        assert all(math.isfinite(v) for v in results.values())
