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
        f"{transient / GIB:.2f} GiB exceeds predicted {predicted / GIB:.2f} + 15%; a "
        f"missing `del` leaves the previous iteration's tensor alive across the loop"
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
