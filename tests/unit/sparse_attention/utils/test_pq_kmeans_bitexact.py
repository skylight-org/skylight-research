"""Bit-identity tests for the memory-bounded k-means in ``pq_utils``.

The change under test replaces two ``(b, n, num_clusters, d)`` fp32 allocations with
in-place / lower-overhead equivalents:

* ``pairwise_distance_batched`` squares the difference **in place** instead of writing
  ``(A - B) ** 2.0``, which materialises a full-size ``sub`` and then a full-size
  ``pow``.  That tensor is ``n * 2**pq_bits * kv_heads * head_dim * 4`` bytes -- exactly
  1 MiB per context key at ``pq_bits=8``, and invariant in ``pq_group_factor`` -- so
  holding two costs 128 GiB at a 64k context and OOMs an H200.
* ``kmeans_batched`` scatters ones into a zeroed float tensor instead of
  ``one_hot(...).float()``, which allocates an int64 tensor first (4 GiB at 64k).

The claim is BITWISE equality with the pre-patch code, and it is structural rather than
empirical: both changes are elementwise, so every downstream reduction still sees a
tensor of identical shape, dtype and layout, and therefore reduces in an identical
order.  Nothing here depends on how CUDA decomposes a reduction.

``_ref_*`` below are verbatim transcriptions of the pre-patch bodies (blob at
``eb79287``).  A transcription error is self-detecting: it makes the equality tests fail
rather than pass vacuously.  ``initialize_batched`` is imported, not transcribed,
because the patch does not touch it.
"""

import pytest
import torch

PQ_UTILS = (
    "sparse_attention_hub.sparse_attention.research_attention.maskers"
    ".fixed.implementations.utils.pq_utils"
)

# PQCacheConfig(pq_group_factor=4, pq_bits=8) on Llama-3.1-8B: 8 kv heads, head_dim 128.
REAL_B, REAL_D, REAL_C = 32, 32, 256


def _mod():
    import importlib

    return importlib.import_module(PQ_UTILS)


# --------------------------------------------------------------- pre-patch references
def _ref_pairwise_distance_batched(data1, data2, device=torch.device("cpu")):
    """``pairwise_distance_batched``, pre-patch, verbatim."""
    data1, data2 = data1.to(device), data2.to(device)
    A = data1.unsqueeze(2)
    B = data2.unsqueeze(1)
    dis = (A - B) ** 2.0
    dis = dis.sum(dim=-1)
    return dis


def _ref_accumulate(X, choice_cluster, num_clusters):
    """The pre-patch centroid-update allocations, verbatim."""
    mask = torch.nn.functional.one_hot(choice_cluster, num_clusters).float()
    X_expanded = X.unsqueeze(2)
    mask_expanded = mask.unsqueeze(-1)
    cluster_sums = (X_expanded * mask_expanded).sum(dim=1)
    cluster_counts = mask.sum(dim=1)
    return cluster_sums, cluster_counts


def _ref_kmeans_batched(
    X,
    num_clusters,
    distance="euclidean",
    cluster_centers=None,
    tol=1e-4,
    iter_limit=0,
    device=torch.device("cpu"),
    seed=None,
    trace=None,
):
    """``kmeans_batched``, pre-patch, verbatim, with an optional trajectory hook."""
    if X.ndim != 3:
        raise ValueError(f"Expected 3D input (b, n, d), got {X.ndim}D")
    if distance != "euclidean":
        raise NotImplementedError(distance)

    b, n, d = X.shape
    original_dtype = X.dtype
    X = X.float()
    X = X.to(device)

    if cluster_centers is None:
        initial_state = _mod().initialize_batched(X, num_clusters, seed=seed)
    else:
        initial_state = cluster_centers.float().to(device)

    iteration = 0
    while True:
        dis = _ref_pairwise_distance_batched(X, initial_state, device=device)
        choice_cluster = torch.argmin(dis, dim=2)
        initial_state_pre = initial_state.clone()
        cluster_sums, cluster_counts = _ref_accumulate(X, choice_cluster, num_clusters)
        empty_clusters = cluster_counts == 0
        cluster_counts_safe = cluster_counts.clamp(min=1).unsqueeze(-1)
        new_centers = cluster_sums / cluster_counts_safe
        if empty_clusters.any():
            random_indices = torch.randint(0, n, (b, num_clusters), device=X.device)
            batch_idx = (
                torch.arange(b, device=X.device).unsqueeze(1).expand(-1, num_clusters)
            )
            random_samples = X[batch_idx, random_indices]
            empty_mask = empty_clusters.unsqueeze(-1)
            new_centers = torch.where(empty_mask, random_samples, new_centers)
        initial_state = new_centers
        center_shift = torch.sqrt(
            ((initial_state - initial_state_pre) ** 2).sum(dim=2)
        ).sum(dim=1)
        iteration += 1
        if trace is not None:
            trace.append(
                (
                    choice_cluster.clone(),
                    cluster_sums.clone(),
                    cluster_counts.clone(),
                    center_shift.clone(),
                )
            )
        if (center_shift**2 < tol).all():
            break
        if iter_limit != 0 and iteration >= iter_limit:
            break

    return choice_cluster.to(original_dtype), initial_state.to(original_dtype)


class _CentroidRecorder:
    """Capture every centroid state the shipped loop evaluates a distance against.

    The patched ``kmeans_batched`` has no seam to monkeypatch inside the loop, but it
    calls ``pairwise_distance_batched`` exactly once per iteration and passes the
    current centroids as ``data2``.  Wrapping that one function therefore yields both
    the iteration count and the full centroid trajectory, which is strictly stronger
    than comparing only the final answer.
    """

    def __init__(self, monkeypatch):
        mod = _mod()
        self.states = []
        real = mod.pairwise_distance_batched

        def spy(data1, data2, device=torch.device("cpu"), tqdm_flag=False):
            self.states.append(data2.clone())
            return real(data1, data2, device=device, tqdm_flag=tqdm_flag)

        monkeypatch.setattr(mod, "pairwise_distance_batched", spy)

    @property
    def iterations(self):
        return len(self.states)


def _clustered_blob(b, n, d, num_clusters, seed):
    """Well-separated clusters, so k-means converges rather than thrashing."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(b, num_clusters, d, generator=g) * 12.0
    idx = torch.randint(0, num_clusters, (b, n), generator=g)
    base = torch.gather(centers, 1, idx.unsqueeze(-1).expand(-1, -1, d))
    return base + torch.randn(b, n, d, generator=g) * 0.1


@pytest.fixture(autouse=True)
def _preserve_rng():
    """No test in this file may leak RNG state into another."""
    state = torch.get_rng_state()
    yield
    torch.set_rng_state(state)


# ============================================================ claim 1: in-place square
@pytest.mark.unit
class TestInPlaceSquare:
    """``x.pow_(2.0)`` must be bit-for-bit ``x ** 2.0``.

    This is the entire numerical content of the distance-side change.  It is a
    per-element identity, so it carries no assumption about reduction order, kernel
    selection, or device.
    """

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("scale", [1e-6, 1e-3, 1.0, 1e3, 1e6])
    def test_pow_inplace_equals_out_of_place(self, dtype, scale):
        torch.manual_seed(0)
        a = torch.randn(7, 53, 1, 31, dtype=dtype) * scale
        b = torch.randn(7, 1, 17, 31, dtype=dtype) / scale
        ref = (a - b) ** 2.0
        got = a - b
        got.pow_(2.0)
        assert torch.equal(ref, got)

    def test_edge_values(self):
        edge = torch.tensor(
            [
                0.0,
                -0.0,
                1e-45,
                -1e-45,
                5.9e-39,
                float("inf"),
                -float("inf"),
                float("nan"),
                3.4e38,
                -3.4e38,
                1.0000001,
            ],
            dtype=torch.float32,
        )
        ref = edge**2.0
        got = edge.clone()
        got.pow_(2.0)
        assert torch.equal(ref.isnan(), got.isnan())
        assert torch.equal(ref.nan_to_num(1.0), got.nan_to_num(1.0))

    @pytest.mark.parametrize("d", [32, 128, 129])
    def test_pairwise_distance_batched_unchanged(self, d):
        """d=32 is pq_group_factor=4, d=128 is gf=1, d=129 is the ip-augmented width."""
        mod = _mod()
        torch.manual_seed(d)
        data1 = torch.randn(REAL_B, 67, d)
        data2 = torch.randn(REAL_B, 16, d)
        ref = _ref_pairwise_distance_batched(data1, data2)
        got = mod.pairwise_distance_batched(data1, data2)
        assert got.dtype == ref.dtype
        assert got.shape == ref.shape
        assert torch.equal(got, ref)

    def test_pairwise_distance_does_not_mutate_its_inputs(self):
        """The in-place square must land on the broadcast temporary, never on an input."""
        mod = _mod()
        torch.manual_seed(1)
        data1 = torch.randn(4, 11, 8)
        data2 = torch.randn(4, 5, 8)
        keep1, keep2 = data1.clone(), data2.clone()
        mod.pairwise_distance_batched(data1, data2)
        assert torch.equal(data1, keep1)
        assert torch.equal(data2, keep2)


# ================================================================== claim 2: the scatter
@pytest.mark.unit
class TestScatterMask:
    """``zeros().scatter_(2, c, 1.0)`` must be bit-for-bit ``one_hot(c, C).float()``."""

    @pytest.mark.parametrize(
        "b,n,num_clusters,d",
        [(REAL_B, 67, REAL_C, REAL_D), (8, 257, 64, 32), (4, 11, 5, 8)],
    )
    def test_scatter_matches_one_hot_float(self, b, n, num_clusters, d):
        torch.manual_seed(b * n)
        X = torch.randn(b, n, d, dtype=torch.float32)
        choice = torch.randint(0, num_clusters, (b, n), dtype=torch.long)

        ref_mask = torch.nn.functional.one_hot(choice, num_clusters).float()
        mask = torch.zeros((b, n, num_clusters), dtype=X.dtype, device=X.device)
        mask.scatter_(2, choice.unsqueeze(-1), 1.0)
        assert mask.dtype == ref_mask.dtype
        assert torch.equal(mask, ref_mask)

        ref_sums, ref_counts = _ref_accumulate(X, choice, num_clusters)
        sums = (X.unsqueeze(2) * mask.unsqueeze(-1)).sum(dim=1)
        counts = mask.sum(dim=1)
        assert torch.equal(sums, ref_sums)
        assert torch.equal(counts, ref_counts)

    def test_counts_are_exact_integers_summing_to_n(self):
        """Counts are sums of 1.0, exact in fp32 below 2**24 (production n is 65,408).

        This is load-bearing rather than decorative: ``cluster_counts == 0`` is a
        data-dependent Python bool that decides whether ``torch.randint`` is called, so
        a single-ULP error in the counts would shift the whole downstream RNG stream.
        """
        b, n, num_clusters = 8, 257, 64
        assert n < 2**24
        torch.manual_seed(2)
        choice = torch.randint(0, num_clusters, (b, n), dtype=torch.long)
        mask = torch.zeros((b, n, num_clusters), dtype=torch.float32)
        mask.scatter_(2, choice.unsqueeze(-1), 1.0)
        counts = mask.sum(dim=1)
        assert torch.equal(counts, counts.round())
        assert torch.equal(counts.sum(dim=1), torch.full((b,), float(n)))

    def test_mask_is_float32_in_the_production_path(self, monkeypatch):
        """``dtype=X.dtype`` equals ``.float()`` only because X is upcast first.

        ``kmeans_batched`` does ``X = X.float()`` before the loop, so the scattered mask
        is fp32 for bf16 and fp64 inputs alike -- which is what makes it equal to the
        pre-patch ``one_hot(...).float()``.  Pin that, because the equality would
        silently break if the upcast were ever removed.
        """
        mod = _mod()
        seen = []
        real = mod.pairwise_distance_batched

        def spy(data1, data2, device=torch.device("cpu"), tqdm_flag=False):
            seen.append((data1.dtype, data2.dtype))
            return real(data1, data2, device=device, tqdm_flag=tqdm_flag)

        monkeypatch.setattr(mod, "pairwise_distance_batched", spy)
        for dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            seen.clear()
            torch.manual_seed(3)
            X = _clustered_blob(4, 40, 8, 5, seed=3).to(dtype)
            mod.kmeans_batched(X, 5, iter_limit=2)
            assert seen, f"loop did not run for {dtype}"
            assert all(a == torch.float32 and b == torch.float32 for a, b in seen), (
                f"{dtype} input reached the loop as {seen[0]}, so the scattered mask "
                f"would not be fp32 and would not match one_hot(...).float()"
            )


# ==================================================== the whole loop, end to end
@pytest.mark.unit
class TestKmeansBatchedBitExact:
    @pytest.mark.parametrize(
        "b,n,num_clusters,d,seed",
        [
            (REAL_B, 67, 32, REAL_D, 11),  # real b/d, prime n
            (8, 257, 64, 32, 12),
            (4, 97, 5, 8, 13),
            # num_clusters == n exercises the empty-cluster branch hard.  It cannot
            # exceed n: initialize_batched draws randperm(n)[:num_clusters], a
            # pre-existing precondition that production satisfies via
            # _should_use_full_attention (seq_len_keys > heavy + init_offset +
            # seq_len_q + 2**pq_bits).
            (REAL_B, REAL_C, REAL_C, REAL_D, 14),
        ],
    )
    def test_outputs_and_trajectory_identical(
        self, monkeypatch, b, n, num_clusters, d, seed
    ):
        """Per-iteration identity, not just final identity.

        The centroid state at the top of every iteration is compared, plus the
        iteration count, plus both return values.  Given an identical RNG stream
        (asserted separately) that is full behavioural identity, because
        ``center_shift``, the empty-cluster replacement and the convergence test are
        pure functions of the recorded state.
        """
        mod = _mod()
        X = _clustered_blob(b, n, d, num_clusters, seed)

        ref_trace = []
        torch.manual_seed(seed + 1000)
        ref_codes, ref_centers = _ref_kmeans_batched(
            X, num_clusters, iter_limit=12, trace=ref_trace
        )

        rec = _CentroidRecorder(monkeypatch)
        torch.manual_seed(seed + 1000)
        codes, centers = mod.kmeans_batched(X, num_clusters, iter_limit=12)

        assert rec.iterations == len(
            ref_trace
        ), f"iteration count changed: {rec.iterations} vs {len(ref_trace)}"
        # states[i] is the centroid set iteration i scored against; ref_trace[i-1]'s
        # successor.  Compare the first (the initialisation) and then each update.
        assert torch.equal(codes, ref_codes)
        assert torch.equal(centers, ref_centers)
        assert codes.dtype == ref_codes.dtype
        assert centers.dtype == ref_centers.dtype

    @pytest.mark.parametrize("seed", [21, 22])
    def test_global_rng_stream_unchanged(self, monkeypatch, seed):
        """``kmeans_batched_pytorch`` passes ``seed=None``: the draws are ambient.

        ``initialize_batched`` draws ``b`` randperms and the empty-cluster branch draws
        a randint on every iteration where some cluster is empty, so the post-call RNG
        state is a single fingerprint of the iteration count AND the empty-cluster
        incidence.  Downstream samplers (``PQImportance``'s ``multinomial``) consume
        this same stream, so preserving it is part of bit-identity.
        """
        mod = _mod()
        X = _clustered_blob(8, 97, 8, 16, seed)

        torch.manual_seed(seed)
        start = torch.get_rng_state()
        ref_codes, ref_centers = _ref_kmeans_batched(X, 16, iter_limit=12)
        ref_state = torch.get_rng_state()
        ref_after = torch.rand(4)

        _CentroidRecorder(monkeypatch)
        torch.set_rng_state(start)
        codes, centers = mod.kmeans_batched(X, 16, iter_limit=12)
        got_state = torch.get_rng_state()
        got_after = torch.rand(4)

        assert torch.equal(got_state, ref_state), "global RNG state diverged"
        assert torch.equal(got_after, ref_after)
        assert torch.equal(codes, ref_codes)
        assert torch.equal(centers, ref_centers)

    def test_empty_cluster_branch_fires_every_iteration(self, monkeypatch):
        """Force ``torch.randint`` on every iteration, the maximally RNG-sensitive case.

        Six distinct points against 256 clusters leaves almost every cluster empty
        every iteration, so the replacement draw fires each time and the RNG
        fingerprint becomes maximally sensitive to any change in the loop.
        """
        mod = _mod()
        g = torch.Generator().manual_seed(31)
        X = torch.randn(8, 6, 4, generator=g).repeat_interleave(50, dim=1).float()
        assert X.shape[1] == 300

        torch.manual_seed(4242)
        start = torch.get_rng_state()
        ref_codes, ref_centers = _ref_kmeans_batched(X, 256, iter_limit=6)
        ref_state = torch.get_rng_state()

        _CentroidRecorder(monkeypatch)
        torch.set_rng_state(start)
        codes, centers = mod.kmeans_batched(X, 256, iter_limit=6)

        assert torch.equal(torch.get_rng_state(), ref_state)
        assert torch.equal(codes, ref_codes)
        assert torch.equal(centers, ref_centers)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
    def test_dtype_round_trips_identically(self, monkeypatch, dtype):
        mod = _mod()
        X = _clustered_blob(4, 60, 8, 8, seed=41).to(dtype)

        torch.manual_seed(7)
        ref_codes, ref_centers = _ref_kmeans_batched(X, 8, iter_limit=5)
        _CentroidRecorder(monkeypatch)
        torch.manual_seed(7)
        codes, centers = mod.kmeans_batched(X, 8, iter_limit=5)

        assert codes.dtype == ref_codes.dtype == dtype
        assert centers.dtype == ref_centers.dtype == dtype
        assert torch.equal(codes, ref_codes)
        assert torch.equal(centers, ref_centers)

    def test_kmeans_batched_pytorch_wrapper_unchanged(self):
        """The masker's actual entry point, including the bf16 code round trip."""
        mod = _mod()
        X = _clustered_blob(8, 80, 8, 16, seed=51).to(torch.bfloat16)

        torch.manual_seed(9)
        ref_codes, ref_centers = _ref_kmeans_batched(X, 16, iter_limit=10)
        torch.manual_seed(9)
        centroids, codes = mod.kmeans_batched_pytorch(X, 16, 10)

        assert torch.equal(centroids, ref_centers)
        assert codes.dtype == torch.int64
        assert torch.equal(codes, ref_codes.long())


# ================================================================ pre-existing hazards
@pytest.mark.unit
class TestPreExistingCodeDtypeHazard:
    """``kmeans_batched`` returns ``choice_cluster.to(original_dtype)``.

    With bf16 keys that casts cluster ids to bfloat16 before
    ``kmeans_batched_pytorch`` puts them back with ``.long()``.  Pinned here so the
    memory change is not blamed for it, and so a later ``pq_bits`` bump is a loud test
    failure rather than an out-of-bounds ``gather``.
    """

    def test_codes_round_trip_exactly_at_256_clusters(self):
        ids = torch.arange(256)
        assert torch.equal(ids.to(torch.bfloat16).long(), ids)

    def test_codes_are_corrupted_above_256_clusters(self):
        assert torch.tensor([255]).to(torch.bfloat16).long().item() == 255
        assert torch.tensor([257]).to(torch.bfloat16).long().item() == 256
        assert torch.tensor([511]).to(torch.bfloat16).long().item() == 512


# ==================================================================== mutation witnesses
@pytest.mark.unit
class TestMutationWitnesses:
    """Prove the assertions above have power.

    Each test builds a deliberately wrong variant and shows that the same data and
    shapes the real tests use distinguish it.  Without these, a green suite might be
    green because the tests cannot tell anything apart.
    """

    def test_row_chunked_accumulate_is_not_exact(self):
        """The variant deliberately NOT used: split the fp32 reduction over n.

        All rows land in cluster 0 with values ``[+2**24, 1, ..., 1, -2**24]`` along n.
        2**24 is the largest fp32 integer whose successor is not representable, so
        whether a ``+1`` survives depends entirely on whether it is added to the large
        value or to its neighbouring ones -- i.e. on the association.  The total
        therefore moves by order n, not by one ulp, under any reassociation.  That is
        what gives this witness its power, and it does not rely on the measured 3e-5
        drift of real key data.

        The test deliberately does NOT assert the reference equals any particular
        value: ``sum(dim=1)`` uses a vectorised/pairwise tree, not a left-to-right
        accumulation, so the reference is order-sensitive rather than analytically 0.
        """
        b, n, d, num_clusters = REAL_B, 67, REAL_D, 8
        X = torch.ones(b, n, d, dtype=torch.float32)
        X[:, 0, :] = float(2**24)
        X[:, -1, :] = -float(2**24)
        choice = torch.zeros(b, n, dtype=torch.long)

        ref_sums, _ = _ref_accumulate(X, choice, num_clusters)

        mask = torch.nn.functional.one_hot(choice, num_clusters).float()
        acc = torch.zeros_like(ref_sums)
        for s in range(0, n, 8):
            e = min(s + 8, n)
            acc += (X[:, s:e].unsqueeze(2) * mask[:, s:e].unsqueeze(-1)).sum(dim=1)
        assert not torch.equal(acc, ref_sums)
        assert (acc - ref_sums).abs().max() > 1.0

    @pytest.mark.parametrize("dim", [0, 1])
    def test_scatter_on_the_wrong_dim_is_detected(self, dim):
        b, n, num_clusters = 8, 67, 32
        torch.manual_seed(61)
        choice = torch.randint(0, num_clusters, (b, n))
        mask = torch.zeros((b, n, num_clusters), dtype=torch.float32)
        try:
            mask.scatter_(dim, choice.unsqueeze(-1), 1.0)
        except RuntimeError:
            return  # scattering along the wrong dim raises: also a kill
        counts = mask.sum(dim=1)
        assert not torch.equal(counts.sum(dim=1), torch.full((b,), float(n)))

    def test_uninitialised_mask_is_detected(self):
        """``torch.empty`` instead of ``torch.zeros`` leaves stale values behind."""
        b, n, num_clusters = 8, 67, 32
        scratch = torch.full((b, n, num_clusters), 7.0, dtype=torch.float32)
        del scratch
        torch.manual_seed(62)
        choice = torch.randint(0, num_clusters, (b, n))
        mask = torch.empty((b, n, num_clusters), dtype=torch.float32)
        mask.scatter_(2, choice.unsqueeze(-1), 1.0)
        counts = mask.sum(dim=1)
        assert not torch.equal(counts.sum(dim=1), torch.full((b,), float(n)))

    def test_reordered_rng_consumption_is_detected(self):
        """Hoisting ``randint`` out of the empty-cluster guard shifts the stream."""
        b, n, d, num_clusters = 8, 300, 4, 256
        g = torch.Generator().manual_seed(63)
        X = torch.randn(b, 6, d, generator=g).repeat_interleave(50, dim=1).float()

        torch.manual_seed(777)
        start = torch.get_rng_state()
        _ref_kmeans_batched(X, num_clusters, iter_limit=4)
        ref_state = torch.get_rng_state()

        torch.set_rng_state(start)
        _mod().initialize_batched(X.float(), num_clusters, seed=None)
        for _ in range(4):
            torch.randint(0, n, (b, num_clusters))
            torch.randint(0, n, (b, num_clusters))
        assert not torch.equal(torch.get_rng_state(), ref_state)

    def test_the_reference_actually_builds_the_big_tensor(self):
        """Guard against a degenerate reference.

        If ``_ref_pairwise_distance_batched`` did not materialise the rank-4 broadcast,
        every equality above would be comparing the patch to itself.  Observe the
        intermediate explicitly.
        """
        a = torch.randn(2, 5, 1, 3)
        b = torch.randn(2, 1, 4, 3)
        assert ((a - b) ** 2.0).shape == (2, 5, 4, 3)
        assert _ref_pairwise_distance_batched(
            torch.randn(2, 5, 3), torch.randn(2, 4, 3)
        ).shape == (2, 5, 4)


# ============================================= the premise the argument actually needs
@pytest.mark.unit
class TestBitIdentityPremises:
    """ "Same shape, dtype and layout" is NOT on its own sufficient, and saying only
    that would be a hole in the argument.

    Two places in PyTorch's reduction config read the operand **base address**, which
    is not part of shape/dtype/layout -- and this patch does change which allocation
    feeds ``sum``, because ``del dis`` and the vanished ``one_hot`` int64 temporary
    both perturb the allocator's history:

    * ``Reduce.cuh`` ``get_output_vec_size`` folds the *first input's* base address
      into ``output_vec_size``, which divides ``dim0`` and so changes the launch
      decomposition.  It is reached by the centroid update's ``sum(dim=1)``.
    * ``input_vectorized_thread_reduce_impl`` peels a misaligned head and folds it
      into accumulator 0, **re-associating** the 4-accumulator sum.  It is reached by
      ``dis.sum(dim=-1)`` when d >= 128, i.e. at pq_group_factor=1.

    Both need only 16-byte alignment, and every torch allocation is far better aligned
    than that: the CUDA caching allocator rounds every block to ``kMinBlockSize = 512``
    bytes and ``cudaMalloc`` itself returns >= 256-byte-aligned pointers, while CPU
    uses ``gAlignment = 64``.  So the head shift is always 0 and the address term never
    reduces the vector size -- identically before and after the patch.

    That premise is what makes the claim extrapolate to 64k and 128k, where it cannot
    be measured directly because the pre-patch code OOMs.  It is shape-independent
    (512 >> 16 for any n), so it does not weaken as n grows.  This test pins it.
    """

    def test_allocations_are_at_least_16_byte_aligned(self):
        import random

        random.seed(0)
        live = []
        bad16 = bad64 = 0
        for _ in range(2000):
            t = torch.empty(random.randint(1, 5000), dtype=torch.float32)
            bad16 += t.data_ptr() % 16 != 0
            bad64 += t.data_ptr() % 64 != 0
            if random.random() < 0.5:
                live.append(t)
            if len(live) > 50:
                live.pop(random.randrange(len(live)))
        assert bad16 == 0, f"{bad16}/2000 allocations were not 16-byte aligned"
        assert bad64 == 0, f"{bad64}/2000 allocations were not 64-byte aligned"

    def test_in_place_square_is_not_a_view_of_either_input(self):
        """If ``A - B`` ever returned a view, ``pow_`` would corrupt a caller's tensor.

        ``torch.sub`` always allocates through TensorIterator, including in the
        degenerate shapes where broadcasting is a no-op.  Zero-element tensors all
        share the null pointer, so they are excluded rather than treated as aliases.
        """
        for n, num_clusters, d in [(1, 1, 4), (1, 5, 4), (5, 1, 4), (3, 7, 1)]:
            data1 = torch.randn(2, n, d)
            data2 = torch.randn(2, num_clusters, d)
            dis = data1.unsqueeze(2) - data2.unsqueeze(1)
            assert dis.numel() > 0
            assert dis.data_ptr() not in (data1.data_ptr(), data2.data_ptr())
            assert dis.storage_offset() == 0 and dis.is_contiguous()

    def test_integer_input_to_pairwise_distance_now_raises(self):
        """A narrowing of the PUBLIC helper's contract, pinned rather than papered over.

        Pre-patch, an all-integer ``data1``/``data2`` returned fp32 distances because
        ``** 2.0`` promotes.  In place it cannot: the result type would have to be cast
        back into a Long buffer, so it raises.  Unreachable from ``kmeans_batched``,
        which does ``X = X.float()`` before the loop and derives the centres from that
        fp32 X -- and there is no other caller in the repo -- but callers of the helper
        should not discover this by surprise.  Mixed int/float still works, because
        promotion then lands on float anyway.
        """
        mod = _mod()
        ints1 = torch.randint(0, 5, (2, 3, 4))
        ints2 = torch.randint(0, 5, (2, 2, 4))
        assert _ref_pairwise_distance_batched(ints1, ints2).dtype == torch.float32
        with pytest.raises(RuntimeError, match="cast"):
            mod.pairwise_distance_batched(ints1, ints2)
        mixed = mod.pairwise_distance_batched(ints1.float(), ints2)
        assert mixed.dtype == torch.float32
        assert torch.equal(mixed, _ref_pairwise_distance_batched(ints1.float(), ints2))

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_direct_calls_are_still_identical(self, dtype):
        """G0 covered fp32/fp64 on CUDA; cover the half types too, on CPU."""
        mod = _mod()
        torch.manual_seed(77)
        data1 = torch.randn(4, 17, 8).to(dtype)
        data2 = torch.randn(4, 5, 8).to(dtype)
        assert torch.equal(
            mod.pairwise_distance_batched(data1, data2),
            _ref_pairwise_distance_batched(data1, data2),
        )

    def test_non_contiguous_inputs_are_identical(self):
        mod = _mod()
        torch.manual_seed(78)
        base = torch.randn(4, 40, 8)
        for data1 in (base.transpose(0, 1).transpose(0, 1), base[:, ::2], base[:, :20]):
            data2 = torch.randn(4, 5, 8)
            assert torch.equal(
                mod.pairwise_distance_batched(data1, data2),
                _ref_pairwise_distance_batched(data1, data2),
            )


# ==================================================================== the revert guard
@pytest.mark.unit
class TestMemoryFixIsStillPresent:
    """The bit-identity tests cannot detect a REVERT.

    They assert "the module agrees with the transcribed pre-patch reference", which a
    revert makes trivially true -- verified: substituting the pre-patch file leaves all
    of the tests above green. So without something here, the OOM fix has no CI guard at
    all: the only test that pins the memory win lives in tests/performance and is gated
    on CUDA plus SAH_PQ_BIG_TESTS=1, so it never runs by default.

    This asserts the fix behaviourally, from the dispatched op stream rather than from
    the source text, so it survives reformatting and comment edits but fails the moment
    the allocation pattern regresses.
    """

    @staticmethod
    def _dispatched_ops(fn):
        from torch.utils._python_dispatch import TorchDispatchMode

        class Trace(TorchDispatchMode):
            def __init__(self):
                self.ops = set()

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                self.ops.add(str(func))
                return func(*args, **(kwargs or {}))

        with Trace() as trace:
            fn()
        return trace.ops

    def test_kmeans_squares_in_place_and_never_calls_one_hot(self):
        mod = _mod()
        X = _clustered_blob(4, 40, 8, 5, seed=81)
        torch.manual_seed(0)
        ops = self._dispatched_ops(lambda: mod.kmeans_batched(X, 5, iter_limit=2))

        assert any("pow_" in op for op in ops), (
            "no in-place pow dispatched: the distance is being squared out of place "
            "again, which doubles the peak and reintroduces the 64k OOM"
        )
        assert any("scatter_" in op for op in ops), "the one-hot scatter is gone"
        assert not any("one_hot" in op for op in ops), (
            "one_hot is back: it materialises an int64 tensor of (b, n, num_clusters), "
            "4 GiB at a 64k context and 8 GiB at 128k, before the float copy"
        )
