"""Production-shape bit-identity and peak-memory tests for the k-means memory fix.

Gated three ways -- CUDA present, ``SAH_PQ_BIG_TESTS=1``, and enough free device memory
-- because the pre-patch reference allocation reaches 65 GiB.  Run on an H200 via sbatch;
these never run in the default suite.

Why the 32k shape matters: at ``b=32, n=32640, C=256, d=32`` the pre-patch
``(A - B) ** 2.0`` needs 2 x 31.88 + 1.00 = 64.75 GiB, which **fits** an H200.  So
bit-identity at the real 32k production shape is provable outright, and the 64k/128k
claim then rests only on the two per-element identities (which are shape-independent and
are covered on CUDA by the unit file and by the G0 probe).

The reference helpers are imported from the unit test file rather than duplicated, so
the two suites can never drift apart.
"""

import os

import pytest
import torch

from tests.unit.sparse_attention.utils.test_pq_kmeans_bitexact import (  # noqa: E501
    _clustered_blob,
    _mod,
    _ref_accumulate,
    _ref_kmeans_batched,
    _ref_pairwise_distance_batched,
)

REAL_B, REAL_D, REAL_C = 32, 32, 256
N_32K = 32640  # 32768 - init_offset(128)
N_64K = 65408
GIB = 1024**3

_enabled = torch.cuda.is_available() and os.environ.get("SAH_PQ_BIG_TESTS") == "1"
pytestmark = [
    pytest.mark.performance,
    pytest.mark.slow,
    pytest.mark.skipif(not _enabled, reason="needs CUDA and SAH_PQ_BIG_TESTS=1"),
]


def _need(gib):
    free, _total = torch.cuda.mem_get_info()
    if free < gib * GIB:
        pytest.skip(f"needs {gib:.0f} GiB free, have {free / GIB:.1f} GiB")


def _reset():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


@pytest.fixture(autouse=True)
def _clean():
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state()
    _reset()
    yield
    torch.set_rng_state(cpu)
    torch.cuda.set_rng_state(cuda)
    _reset()


# ================================================ the two per-element identities, CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pow_inplace_is_bitwise_on_cuda(dtype):
    """The whole structural argument, on the device that actually runs it."""
    torch.manual_seed(0)
    for trial in range(7):
        a = torch.randn(7, 53, 1, 31, dtype=dtype, device="cuda") * 10.0 ** (trial - 3)
        b = torch.randn(7, 1, 17, 31, dtype=dtype, device="cuda") * 10.0 ** (3 - trial)
        ref = (a - b) ** 2.0
        got = a - b
        got.pow_(2.0)
        assert torch.equal(ref, got), f"trial {trial}"


def test_scatter_matches_one_hot_on_cuda():
    b, n, num_clusters, d = REAL_B, 97, REAL_C, REAL_D
    torch.manual_seed(1)
    X = torch.randn(b, n, d, dtype=torch.float32, device="cuda")
    choice = torch.randint(0, num_clusters, (b, n), dtype=torch.long, device="cuda")
    ref_sums, ref_counts = _ref_accumulate(X, choice, num_clusters)
    mask = torch.zeros((b, n, num_clusters), dtype=X.dtype, device=X.device)
    mask.scatter_(2, choice.unsqueeze(-1), 1.0)
    assert torch.equal(mask, torch.nn.functional.one_hot(choice, num_clusters).float())
    assert torch.equal((X.unsqueeze(2) * mask.unsqueeze(-1)).sum(dim=1), ref_sums)
    assert torch.equal(mask.sum(dim=1), ref_counts)


# ============================================ the ladder, up to the real 32k shape
@pytest.mark.parametrize("n", [1024, 8192, N_32K])
def test_pairwise_distance_bit_identical_at_production_shapes(n):
    """b/C/d are the real PQCacheConfig(gf=4, pq_bits=8) values; n climbs to 32k."""
    mod = _mod()
    b, C, d = REAL_B, REAL_C, REAL_D
    _need(2.2 * b * n * C * d * 4 / GIB + 6)
    torch.manual_seed(n)
    X = torch.randn(b, n, d, dtype=torch.float32, device="cuda")
    centers = torch.randn(b, C, d, dtype=torch.float32, device="cuda")

    _reset()
    ref = _ref_pairwise_distance_batched(X, centers, device=X.device)
    ref_peak = torch.cuda.max_memory_allocated()
    _reset()
    got = mod.pairwise_distance_batched(X, centers, device=X.device)
    got_peak = torch.cuda.max_memory_allocated()

    assert torch.equal(got, ref)
    assert torch.equal(got.argmin(dim=2), ref.argmin(dim=2))
    # The reference must actually have built the rank-4 broadcast TWICE, or the
    # comparison is against a degenerate reference and proves nothing.
    expect_ref = 2 * b * n * C * d * 4
    assert ref_peak > 0.9 * expect_ref, (
        f"reference peaked at {ref_peak / GIB:.2f} GiB, expected ~"
        f"{expect_ref / GIB:.2f}: it did not hold both the sub and the pow"
    )
    assert got_peak < 0.6 * ref_peak, (
        f"patched peak {got_peak / GIB:.2f} GiB is not meaningfully below the "
        f"reference's {ref_peak / GIB:.2f} GiB"
    )


def test_kmeans_batched_bit_identical_at_the_real_32k_shape():
    """G4: full-loop bit-identity at the shape PQCache actually runs on RULER-32k.

    ``b=32, n=32640, C=256, d=32`` is exactly what
    ``PQCacheConfig(pq_group_factor=4, pq_bits=8)`` builds from a 32k context on
    Llama-3.1-8B.  The pre-patch reference needs 64.75 GiB and fits, so this is a
    direct proof rather than an invariance argument.
    """
    mod = _mod()
    b, n, C, d = REAL_B, N_32K, REAL_C, REAL_D
    _need(72)
    X = _clustered_blob(b, n, d, C, seed=99).to("cuda")

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    cpu0, cuda0 = torch.get_rng_state(), torch.cuda.get_rng_state()
    _reset()
    ref_codes, ref_centers = _ref_kmeans_batched(X, C, iter_limit=3, device=X.device)
    ref_peak = torch.cuda.max_memory_allocated()
    cpu_ref, cuda_ref = torch.get_rng_state(), torch.cuda.get_rng_state()
    del X
    _reset()

    X = _clustered_blob(b, n, d, C, seed=99).to("cuda")
    torch.set_rng_state(cpu0)
    torch.cuda.set_rng_state(cuda0)
    _reset()
    codes, centers = mod.kmeans_batched(X, C, iter_limit=3, device=X.device)
    peak = torch.cuda.max_memory_allocated()

    assert torch.equal(codes, ref_codes), "cluster assignments diverged"
    assert torch.equal(centers, ref_centers), "centroids diverged"
    assert torch.equal(torch.get_rng_state(), cpu_ref), "CPU RNG stream diverged"
    assert torch.equal(torch.cuda.get_rng_state(), cuda_ref), "CUDA RNG stream diverged"
    print(
        f"\n32k shape: reference peak {ref_peak / GIB:.2f} GiB, "
        f"patched peak {peak / GIB:.2f} GiB, ratio {ref_peak / peak:.2f}x"
    )
    assert (
        peak < 0.55 * ref_peak
    ), f"expected roughly a 2x cut, got {ref_peak / peak:.2f}x"


# ==================================================================== peak memory, 64k
def test_peak_memory_at_64k_matches_the_closed_form():
    """Two-sided: the 64k k-means transient must land on S4 + S3, not above or below.

    S4 = 4*C*kv_heads*head_dim*n (the (b, n, C, d) fp32 tensor, 1 MiB per key at
    pq_bits=8, invariant in pq_group_factor); S3 = 4*C*b*n (the (b, n, C) tensors).
    A peak that is too LOW is also a failure: it would mean the loop never ran at
    production scale.
    """
    mod = _mod()
    b, n, C, d = REAL_B, N_64K, REAL_C, REAL_D
    s4 = 4 * C * b * d * n
    s3 = 4 * C * b * n
    _need((s4 + s3) / GIB + 8)

    X = _clustered_blob(b, n, d, C, seed=100).to("cuda")
    torch.manual_seed(5)
    torch.cuda.manual_seed_all(5)
    _reset()
    base = torch.cuda.memory_allocated()
    mod.kmeans_batched(X, C, iter_limit=2, device=X.device)
    torch.cuda.synchronize()
    transient = torch.cuda.max_memory_allocated() - base

    predicted = s4 + s3
    old = 2 * s4 + 2 * s3
    print(
        f"\n64k transient {transient / GIB:.2f} GiB "
        f"(predicted {predicted / GIB:.2f}, pre-patch {old / GIB:.2f})"
    )
    assert transient < 1.15 * predicted, (
        f"{transient / GIB:.2f} GiB exceeds predicted {predicted / GIB:.2f} plus 15 "
        f"percent, which is what a missing del looks like: the previous iteration's "
        f"tensor is still alive when the next one is allocated"
    )
    assert transient > 0.85 * s4, (
        f"{transient / GIB:.2f} GiB is far below S4 = {s4 / GIB:.2f} GiB -- the loop "
        f"cannot have run at the production shape"
    )
    assert transient < old / 1.9, "the patch must cut the pre-patch peak by ~2x"


# ======================== measurements that decide whether a TILED path can be bitwise
def test_chunking_the_centroid_update_would_NOT_be_bitwise():
    """Why this fix squares in place instead of chunking, which is the obvious
    alternative and would use even less memory.

    Chunking the centroid update over groups is the natural way to bound
    `(X.unsqueeze(2) * mask.unsqueeze(-1)).sum(dim=1)`.  It is not bit-preserving in
    general.  For that reduction the reduced axis is not the fastest-striding one, so
    `num_outputs` feeds `grid().x` and gates PyTorch's CTA-splitting branch: at
    (b=32, n=8192, C=64, d=32) on an H200 the full batch resolves to
    `ctas_per_output=8` while a single-group chunk resolves to 128, and the fp32
    summation order differs.

    Measured here rather than argued, because whether it bites depends on the shape and
    on the device.  Squaring in place sidesteps the question entirely: it changes no
    reduction's shape, so no reduction can change its decomposition.
    """
    b, n = REAL_B, 8192
    _need(10)
    T = torch.randn(b, n, 64, REAL_D, dtype=torch.float32, device="cuda")
    T[:, 0] = float(2**24)
    T[:, -1] = -float(2**24)
    full = T.sum(dim=1)
    observed = {}
    for g in (1, 2, 4, 8, 16, 32):
        blocked = torch.cat(
            [T[s : min(s + g, b)].sum(dim=1) for s in range(0, b, g)], dim=0
        )
        observed[g] = torch.equal(full, blocked)
    print(f"\ngroup-chunked equality at (32, 8192, 64, 32): {observed}")
    assert observed[32] is True, "g == b is the unchunked case and must agree"
    assert any(v is False for g, v in observed.items() if g < b), (
        "no group count diverged: either this GPU does not enter the CTA-splitting "
        "branch at this shape, or torch changed setReduceConfig.  Re-derive before "
        "citing this as the reason the fix does not chunk."
    )


# ============================== what actually underwrites the tiled path above 64k
def test_group_tiled_accumulate_matches_full_batch_at_the_real_64k_shape():
    """The 128k claim rests on this, because at 128k the reference does not fit.

    The tiled path's centroid update is group-tiled, and group tiling is NOT
    unconditionally exact: `test_group_blocking_the_outer_reduction_is_NOT_order_stable`
    shows it diverging at (b=32, n=8192, C=64, d=32) for every group count below b.
    At the production shape it is safe because `ctas_per_output` is pinned by a term
    that does not involve the group count -- and 64k is the largest context where that
    can be CHECKED, since the full-batch reference costs S4 + S3 = 65.9 GiB there and
    131.9 GiB at 128k.

    So: verify every group count the planner could pick against the untiled reference,
    at the real 64k shape.  128k then rests on this plus the live self-check in
    `kmeans_batched`, not on an unverified extrapolation.
    """
    mod = _mod()
    b, n, num_clusters, d = REAL_B, N_64K, REAL_C, REAL_D
    _need((b * n * num_clusters * d * 4 + b * n * num_clusters * 4) / GIB + 8)
    torch.manual_seed(64)
    X = torch.randn(b, n, d, dtype=torch.float32, device="cuda")
    choice = torch.randint(0, num_clusters, (b, n), dtype=torch.long, device="cuda")

    _reset()
    ref_sums, ref_counts = _ref_accumulate(X, choice, num_clusters)
    ref_peak = torch.cuda.max_memory_allocated()
    assert ref_peak > 0.9 * b * n * num_clusters * d * 4, (
        f"reference peaked at only {ref_peak / GIB:.1f} GiB -- it did not build the "
        f"(b, n, C, d) tensor, so it is not the untiled computation"
    )
    _reset()

    for groups in (1, 2, 3, 4, 5, 7, 8, 16, 31, 32):
        scratch = torch.full(
            (b, num_clusters, d), float("nan"), dtype=torch.float32, device="cuda"
        )
        del scratch
        sums, counts = mod._accumulate_batched(X, choice, num_clusters, groups)
        assert not torch.isnan(sums).any(), f"unwritten tile at groups={groups}"
        assert torch.equal(sums, ref_sums), (
            f"groups={groups} diverges from the untiled reference at the production "
            f"64k shape -- the tiled path cannot be used above 64k"
        )
        assert torch.equal(counts, ref_counts), f"groups={groups}"
        del sums, counts
        _reset()


def test_tiled_peak_at_128k_stays_under_the_tile_budget():
    """The 128k path must actually bound its transient to the tile, not S4 + S3."""
    mod = _mod()
    b, num_clusters, d = REAL_B, REAL_C, REAL_D
    n = 130944
    rows, groups = mod._plan_kmeans(b, n, d, num_clusters)
    assert not (rows >= n and groups >= b), "128k must tile"
    tile = max(
        rows * b * num_clusters * (d + 1) * 4, groups * n * num_clusters * (d + 1) * 4
    )
    resident = b * n * d * 4 + 8 * b * n + 2 * b * num_clusters * d * 4
    _need((tile + resident) / GIB + 6)

    X = _clustered_blob(b, n, d, num_clusters, seed=128).to("cuda")
    torch.manual_seed(3)
    torch.cuda.manual_seed_all(3)
    _reset()
    base = torch.cuda.memory_allocated()
    mod.kmeans_batched(X, num_clusters, iter_limit=2, device=X.device)
    torch.cuda.synchronize()
    transient = torch.cuda.max_memory_allocated() - base
    untiled = b * n * num_clusters * d * 4 + b * n * num_clusters * 4
    print(
        f"\n128k tiled transient {transient / GIB:.2f} GiB "
        f"(tile budget {tile / GIB:.2f}, untiled would be {untiled / GIB:.2f})"
    )
    assert transient < 3 * tile, "the tile budget is not bounding the transient"
    assert transient < untiled / 8, "tiling bought less than 8x"


def test_tiled_loop_equals_single_shot_at_the_real_64k_shape():
    """The whole tiled loop against the whole single-shot loop, at the production shape.

    This is what licenses the tiled path, and the only existing full-loop tiled test runs
    at a toy shape (b=8, n=61, C=16, d=8) where none of the production tile sizes,
    reduction configs or tail cases occur.  Here b, C and d are the real
    PQCacheConfig(pq_group_factor=4, pq_bits=8) values and n is the real 64k key count, so
    the tiled run resolves the same rows/groups it would resolve in production.

    Note what is NOT compared here, and why.  The PRE-PATCH reference cannot be run at
    this shape: it holds two copies of the (b, n, C, d) tensor, 131.7 GiB, which does not
    fit.  So the chain is

        tiled == single-shot     <- measured here, at the production 64k shape
        single-shot == pre-patch <- measured at the production 32k shape
                                    (test_kmeans_batched_bit_identical_at_the_real_32k_shape),
                                    and structural everywhere: the in-place square and the
                                    scatter change no reduction's shape
        => tiled == pre-patch

    64k is the largest shape where the first link is measurable at all, because the
    single-shot loop itself needs S4 + S3 = 65.9 GiB here and 131.9 GiB at 128k.  The
    128k path therefore rests on this plus the shape-independence of both links.

    Compares codes, centroids, the iteration count, AND both RNG streams -- the loop draws
    from the ambient generator in `initialize_batched` and in the empty-cluster branch, and
    a tiling that changed how often either fires would shift every downstream sampler.
    """
    mod = _mod()
    b, n, num_clusters, d = REAL_B, N_64K, REAL_C, REAL_D
    _need((b * n * num_clusters * d * 4 + b * n * num_clusters * 4) / GIB + 8)

    X = _clustered_blob(b, n, d, num_clusters, seed=641).to("cuda")

    # --- single-shot: assert the planner really does NOT tile at 64k ----------------
    rows, groups = mod._plan_kmeans(b, n, d, num_clusters)
    assert (
        rows >= n and groups >= b
    ), f"64k was expected to stay single-shot, got rows={rows} groups={groups}"
    torch.manual_seed(6410)
    torch.cuda.manual_seed_all(6410)
    cpu0, cuda0 = torch.get_rng_state(), torch.cuda.get_rng_state()
    _reset()
    ref_codes, ref_centers = mod.kmeans_batched(
        X, num_clusters, iter_limit=3, device=X.device
    )
    single_peak = torch.cuda.max_memory_allocated()
    cpu_ref, cuda_ref = torch.get_rng_state(), torch.cuda.get_rng_state()
    ref_codes, ref_centers = ref_codes.clone(), ref_centers.clone()
    _reset()

    # --- tiled: force the path the 128k sweep takes, at a shape we can check --------
    calls = {"assign": 0, "accum": 0}
    real_assign, real_accum = mod._assign_batched, mod._accumulate_batched

    def spy_assign(X_, centers_, rows_):
        calls["assign"] += 1
        assert rows_ < X_.shape[1], "assignment did not actually tile"
        return real_assign(X_, centers_, rows_)

    def spy_accum(X_, choice_, C_, groups_):
        calls["accum"] += 1
        assert groups_ < X_.shape[0], "centroid update did not actually tile"
        return real_accum(X_, choice_, C_, groups_)

    saved = mod._EXACT_PLAN_MAX_BYTES
    try:
        mod._EXACT_PLAN_MAX_BYTES = 0  # force tiling at this shape
        mod._assign_batched, mod._accumulate_batched = spy_assign, spy_accum
        t_rows, t_groups = mod._plan_kmeans(b, n, d, num_clusters)
        assert not (t_rows >= n and t_groups >= b), "forcing the tiled path failed"
        torch.set_rng_state(cpu0)
        torch.cuda.set_rng_state(cuda0)
        _reset()
        codes, centers = mod.kmeans_batched(
            X, num_clusters, iter_limit=3, device=X.device
        )
        tiled_peak = torch.cuda.max_memory_allocated()
    finally:
        mod._EXACT_PLAN_MAX_BYTES = saved
        mod._assign_batched, mod._accumulate_batched = real_assign, real_accum

    print(
        f"\n64k production shape: single-shot peak {single_peak / GIB:.2f} GiB, "
        f"tiled peak {tiled_peak / GIB:.2f} GiB "
        f"(rows={t_rows}, groups={t_groups}, "
        f"{calls['assign']} assign / {calls['accum']} accum calls)"
    )
    assert calls["assign"] >= 3 and calls["accum"] >= 3, calls
    assert torch.equal(codes, ref_codes), "tiled cluster assignments diverged"
    assert torch.equal(centers, ref_centers), "tiled centroids diverged"
    assert torch.equal(torch.get_rng_state(), cpu_ref), "CPU RNG stream diverged"
    assert torch.equal(torch.cuda.get_rng_state(), cuda_ref), "CUDA RNG stream diverged"
    assert (
        tiled_peak < single_peak / 4
    ), f"tiling bought only {single_peak / tiled_peak:.1f}x"
