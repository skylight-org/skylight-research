"""Utility functions for Product Quantization (PQ) operations.

K-means implementation is extracted from:
https://github.com/subhadarship/kmeans_pytorch
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# ============================================================================
# Minimal K-means implementation extracted from kmeans_pytorch
# Source: https://github.com/subhadarship/kmeans_pytorch
# ============================================================================


def _initialize_kmeans(
    X: torch.Tensor, num_clusters: int, seed: int = None
) -> torch.Tensor:
    """Initialize cluster centers.

    Args:
        X: Input matrix of shape (n_samples, n_features)
        num_clusters: Number of clusters
        seed: Random seed for reproducibility

    Returns:
        Initial cluster centers of shape (num_clusters, n_features)
    """
    num_samples: int = len(X)
    if seed is None:
        indices: np.ndarray = np.random.choice(num_samples, num_clusters, replace=False)
    else:
        np.random.seed(seed)
        indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state: torch.Tensor = X[indices]
    return initial_state


def _pairwise_distance(
    data1: torch.Tensor,
    data2: torch.Tensor,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Compute pairwise Euclidean distances.

    Args:
        data1: First data matrix of shape (n_samples_1, n_features)
        data2: Second data matrix of shape (n_samples_2, n_features)
        device: Device for computation

    Returns:
        Distance matrix of shape (n_samples_1, n_samples_2)
    """
    # Transfer to device
    data1 = data1.to(device)
    data2 = data2.to(device)

    # N*1*M
    A: torch.Tensor = data1.unsqueeze(dim=1)

    # 1*N*M
    B: torch.Tensor = data2.unsqueeze(dim=0)

    dis: torch.Tensor = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis


def kmeans(
    X: torch.Tensor,
    num_clusters: int,
    distance: str = "euclidean",
    cluster_centers: torch.Tensor = None,
    tol: float = 1e-4,
    iter_limit: int = 0,
    device: torch.device = torch.device("cpu"),
    seed: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform k-means clustering.

    Args:
        X: Input matrix of shape (n_samples, n_features)
        num_clusters: Number of clusters
        distance: Distance metric (only 'euclidean' supported)
        cluster_centers: Optional initial cluster centers
        tol: Convergence tolerance threshold
        iter_limit: Maximum number of iterations (0 = no limit)
        device: Device for computation
        seed: Random seed for initialization

    Returns:
        cluster_ids: Cluster assignment for each sample (n_samples,)
        cluster_centers: Final cluster centers (num_clusters, n_features)
    """
    if distance != "euclidean":
        raise NotImplementedError(f"Distance '{distance}' not supported")

    original_dtype: torch.dtype = X.dtype
    # Convert to float
    X = X.float()

    # Transfer to device
    X = X.to(device)

    # Initialize
    if cluster_centers is None:
        initial_state: torch.Tensor = _initialize_kmeans(X, num_clusters, seed=seed)
    else:
        # Find data point closest to the initial cluster center
        initial_state = cluster_centers.to(device)
        dis: torch.Tensor = _pairwise_distance(X, initial_state, device=device)
        choice_points: torch.Tensor = torch.argmin(dis, dim=0)
        initial_state = X[choice_points]
        initial_state = initial_state.to(device)

    iteration: int = 0
    while True:
        dis = _pairwise_distance(X, initial_state, device=device)

        choice_cluster: torch.Tensor = torch.argmin(dis, dim=1)

        initial_state_pre: torch.Tensor = initial_state.clone()

        for index in range(num_clusters):
            selected: torch.Tensor = (
                torch.nonzero(choice_cluster == index).squeeze().to(device)
            )

            selected = torch.index_select(X, 0, selected)

            # Handle empty clusters
            if selected.shape[0] == 0:
                selected = X[torch.randint(len(X), (1,))]

            initial_state[index] = selected.mean(dim=0)

        center_shift: torch.Tensor = torch.sum(
            torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1))
        )

        # Increment iteration
        iteration = iteration + 1

        if center_shift**2 < tol:
            break
        if iter_limit != 0 and iteration >= iter_limit:
            break

    return choice_cluster, initial_state.to(original_dtype)


# ============================================================================
# End of kmeans_pytorch extraction
# ============================================================================

# ============================================================================
# Batched K-means implementation following the code in kmeans_pytorch
# Source: https://github.com/subhadarship/kmeans_pytorch
# ============================================================================


"""
Batched K-Means Implementation

This module implements a batched version of k-means clustering that operates on
tensors with shape (b, n, d) where:
- b = batch size (number of independent clustering problems)
- n = number of samples per batch
- d = dimensionality of each sample

Design choices:
- Fully independent batches (each batch has its own cluster centers)
- One-hot masking for vectorized center updates
- Fully vectorized empty cluster handling
- All-batch convergence (iterate until all batches meet tolerance)
"""


def initialize_batched(
    X: torch.Tensor, num_clusters: int, seed: Optional[int] = None
) -> torch.Tensor:
    """
    Initialize cluster centers for batched k-means

    Args:
        X: (torch.tensor) Input data of shape (b, n, d)
        num_clusters: (int) Number of clusters
        seed: (int) Random seed for reproducibility

    Returns:
        (torch.tensor) Initial cluster centers of shape (b, num_clusters, d)
    """
    b, n, d = X.shape

    if seed is not None:
        torch.manual_seed(seed)

    # Sample indices for each batch independently
    # We'll use random permutation and take first k elements
    indices = torch.stack(
        [torch.randperm(n, device=X.device)[:num_clusters] for _ in range(b)]
    )  # Shape: (b, num_clusters)

    # Gather initial centers using advanced indexing
    # Create batch indices that align with sampled indices
    batch_indices = (
        torch.arange(b, device=X.device).unsqueeze(1).expand(-1, num_clusters)
    )

    # Select initial centers
    initial_state = X[batch_indices, indices]  # Shape: (b, num_clusters, d)

    return initial_state


def pairwise_distance_batched(
    data1: torch.Tensor,
    data2: torch.Tensor,
    device: torch.device = torch.device("cpu"),
    tqdm_flag: bool = False,
) -> torch.Tensor:
    """
    Compute pairwise Euclidean distances for batched data

    Args:
        data1: (torch.tensor) Shape (b, n, d) - input samples
        data2: (torch.tensor) Shape (b, num_clusters, d) - cluster centers
        device: (torch.device) Device for computation
        tqdm_flag: (bool) Whether to print debug info

    Returns:
        (torch.tensor) Pairwise distances of shape (b, n, num_clusters)
    """
    if tqdm_flag:
        print(f"device is: {device}")

    # Transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # Expand dimensions for broadcasting
    # (b, n, 1, d)
    A = data1.unsqueeze(2)

    # (b, 1, num_clusters, d)
    B = data2.unsqueeze(1)

    # Broadcasting: (b, n, num_clusters, d).
    #
    # Squared in place: `(A - B) ** 2.0` keeps the full-size `sub` alive while `pow`
    # allocates a second tensor of the same shape, and that shape costs
    # `n_keys * 2**pq_bits * kv_heads * head_dim * 4` bytes -- 1 MiB per context key at
    # pq_bits=8, and invariant in pq_group_factor.  Holding two is 128 GiB at a 64k
    # context, which does not fit beside the model and a long KV cache.
    #
    # This is exact, not merely equivalent: `pow_.Scalar` and `pow.Tensor_Scalar`
    # resolve to the same structured implementation, the op is elementwise, and
    # `sum(dim=-1)` below still sees a tensor of the same shape and alignment.
    # One contract change: all-integer inputs now raise instead of promoting to float.
    # `kmeans_batched` upcasts X to fp32 before the loop, so it cannot reach that.
    dis = A - B
    dis.pow_(2.0)

    # Sum over feature dimension: (b, n, num_clusters)
    dis = dis.sum(dim=-1)

    return dis


# --- memory-bounded k-means internals, for contexts above ~64k -----------------------
#
# Squaring the distance in place (see `pairwise_distance_batched`) halves the peak and
# is enough through a 64k context: the transient is S4 + S3 where
#
#     S4 = 4 * 2**pq_bits * kv_heads * head_dim * n_keys   # the (b, n, C, d) fp32 temp
#     S3 = 4 * 2**pq_bits * b * n_keys                     # the (b, n, C) fp32 temps
#
# S4 is exactly 1 MiB per context key at pq_bits=8 and is invariant in
# pq_group_factor, so S4 + S3 is 32.9 GiB at 32k, 65.9 GiB at 64k and 131.9 GiB at
# 128k.  Measured end to end on an H200: 105.30 GiB of process peak at 64k against
# 139.80 GiB of device, and the same arithmetic puts 128k around 197 GiB.  So above 64k
# there is no choice but to tile.
#
# The two phases are tiled along DIFFERENT axes, for different reasons.
#
#   assignment      tiled over ROWS.  Bit-invariant by construction: the only reduction
#                   is `sum(dim=-1)` over d, which is the fastest-striding dimension,
#                   so `values_per_thread` is ceil(d / 32) = 1 -- far below the 256
#                   threshold that gates the CTA-splitting branch, the only part of
#                   PyTorch's reduction config that reads the grid size.  The
#                   decomposition is therefore the same warp-shuffle tree at every
#                   block size, on every device.
#
#   centroid update tiled over GROUPS, never rows.  `cluster_sums` reduces over n;
#                   splitting n and adding the partials re-associates those additions
#                   and moves the centroids at the 1e-5 level, which flips downstream
#                   argmins.  Slicing groups keeps each output element's reduction over
#                   the full n inside one kernel launch.
#
# Group tiling is NOT unconditionally exact, and that is measured rather than assumed:
# at (b=32, n=8192, C=64, d=32) on an H200 a group-tiled `sum(dim=1)` differs from the
# full-batch result for EVERY group count below b, because `num_outputs` feeds grid().x
# and selects a different `ctas_per_output`.
#
# It IS exact at every shape this code tiles, for a reason that can be stated exactly.
# `ctas_per_output = max(min(cpo1, cpo2), cpo3)` with
# `cpo1 = div_up(target_grid_size, grid)` and `cpo3 = div_up(values_per_thread, 256)`.
# `cpo3` depends only on n; `cpo1` is bounded by `div_up(2112, 64) = 33` at this
# geometry.  Once n is beyond roughly 34k, `cpo3 >= 33 >= cpo1` and the group count
# drops out of the answer entirely -- and tiling does not begin until n is around
# 71,500.  The counter-example above has n = 8192, far below that crossover.  (Torch
# reaches the same place a second way: this iterator cannot use 32-bit indexing, and
# `get_dim_to_split` peels the batch axis first, so the untiled reduction is already
# decomposed into single-group sub-iterators before any tiling is applied.)
#
# That derivation is checked against reality rather than trusted:
# `test_group_tiled_accumulate_matches_full_batch_at_the_real_64k_shape` compares every
# group count 1..32 against the untiled reference at the production 64k shape -- the
# largest where the reference still fits (65.9 GiB; 131.9 at 128k) -- and
# `test_group_blocking_the_outer_reduction_is_NOT_order_stable` pins the counter-example
# so that a torch change which moved the crossover would be caught.
#
# An earlier revision carried a runtime self-check here instead.  It was removed:
# simulating `setReduceConfig` showed its refusal branch is unreachable at any shape
# that tiles (by the argument above), while its comparison allocation was 12.36 GiB
# against an 8.00 GiB tile -- 55% of the 128k peak spent on a guard that cannot fire.
# The tests are the right home for this check, because they can afford the memory that
# makes it meaningful.

#: Below this working set, do not tile at all.
#:
#: Derived from a measurement rather than guessed.  On an H200 with
#: Llama-3.1-8B-Instruct, a 64k RULER context (n = 64,917) peaks at 105.30 GiB, of
#: which 65.37 GiB is this transient and 39.93 GiB is everything else live at that
#: moment.  Both terms are linear in n, so the whole process peak is ~105.30 * n/64,917
#: GiB and a 139.80 GiB device runs out at n ~= 85,500.  A 72 GiB ceiling on the
#: transient corresponds to n ~= 71,500, i.e. it keeps every context through 64k (and a
#: little above) on the unconditionally-exact single-shot path while leaving ~24 GiB of
#: headroom, and tiles anything larger.  128k (131.9 GiB) is comfortably above it.
#:
#: A CONSTANT, not a free-memory query: deriving it from machine state would make the
#: numerical output depend on what else happened to be resident on the device.  The
#: consequence is deliberate -- on a smaller card a 64k context OOMs here instead of
#: quietly switching to the tiled path and producing different bits.
#:
#: Note the worst case is NOT the longest context.  Peak is discontinuous at the
#: ceiling: n = 71,400 stays untiled and is modelled at ~114 GiB, while n = 71,500 tiles
#: and drops to ~58, and 128k lands at ~78.  So the largest peak this code can produce
#: sits just BELOW the ceiling, not at the top of the supported range.
_EXACT_PLAN_MAX_BYTES: int = 72 * 1024**3

#: Per-tile budget once tiling is on.  Genuinely free on both axes at the shapes this
#: code tiles: row tiling cannot change a bit, and the group count drops out of the
#: centroid update's reduction once n is past the crossover (see the note above), which
#: every tiling shape is.  Note it sits just under 2**31 fp32 elements per row tile --
#: raising it past ~8.25 GiB would push a tile over INT32_MAX and into torch's
#: 32-bit-indexing split path.
_TILE_CHUNK_BYTES: int = 8 * 1024**3


def _plan_kmeans(b: int, n: int, d: int, num_clusters: int) -> Tuple[int, int]:
    """Tile sizes for one k-means problem, as a pure function of its shape.

    Returns ``(rows, groups)``.  ``rows >= n`` and ``groups >= b`` mean "do not tile",
    and the callers then run the single-shot code verbatim.
    """
    element = 4  # the loop runs in fp32
    s4 = b * n * num_clusters * d * element
    s3 = b * n * num_clusters * element
    if s4 + s3 <= _EXACT_PLAN_MAX_BYTES:
        return n, b
    per_row = b * num_clusters * (d + 1) * element
    per_group = n * num_clusters * (d + 1) * element
    rows = max(1, min(n, _TILE_CHUNK_BYTES // max(1, per_row)))
    groups = max(1, min(b, _TILE_CHUNK_BYTES // max(1, per_group)))
    return rows, groups


def _assign_batched(X: torch.Tensor, centers: torch.Tensor, rows: int) -> torch.Tensor:
    """``argmin`` over clusters of the squared distance, in blocks of ``rows`` rows.

    Bit-identical to ``argmin(pairwise_distance_batched(X, centers), dim=2)`` at every
    block size: each output row depends only on its own slice of X, and the reduction
    over d cannot change its decomposition (see the note above).
    """
    b, n, _d = X.shape
    if rows >= n:
        return torch.argmin(
            pairwise_distance_batched(X, centers, device=X.device, tqdm_flag=False),
            dim=2,
        )
    out = torch.empty((b, n), dtype=torch.long, device=X.device)
    B = centers.unsqueeze(1)
    for start in range(0, n, rows):
        stop = min(start + rows, n)
        dis = X[:, start:stop, :].unsqueeze(2) - B
        dis.pow_(2.0)
        dis = dis.sum(dim=-1)
        out[:, start:stop] = torch.argmin(dis, dim=2)
        del dis  # a Python local outlives the iteration; the next tile needs the room
    return out


def _accumulate_batched(
    X: torch.Tensor, choice_cluster: torch.Tensor, num_clusters: int, groups: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-cluster sums and counts, a few groups at a time."""
    b, n, d = X.shape
    if groups >= b:
        mask = torch.zeros((b, n, num_clusters), dtype=X.dtype, device=X.device)
        mask.scatter_(2, choice_cluster.unsqueeze(-1), 1.0)
        mask_expanded = mask.unsqueeze(-1)
        sums = (X.unsqueeze(2) * mask_expanded).sum(dim=1)
        counts = mask.sum(dim=1)
        del mask, mask_expanded
        return sums, counts

    sums = torch.empty((b, num_clusters, d), dtype=X.dtype, device=X.device)
    counts = torch.empty((b, num_clusters), dtype=X.dtype, device=X.device)
    for start in range(0, b, groups):
        stop = min(start + groups, b)
        mask = torch.zeros(
            (stop - start, n, num_clusters), dtype=X.dtype, device=X.device
        )
        mask.scatter_(2, choice_cluster[start:stop].unsqueeze(-1), 1.0)
        mask_expanded = mask.unsqueeze(-1)
        # Assign through a slice rather than `out=sums[start:stop]`: the reduction's
        # output must be a freshly allocated contiguous tensor, because PyTorch's
        # reduction config reads operand addresses and strides.
        sums[start:stop] = (X[start:stop].unsqueeze(2) * mask_expanded).sum(dim=1)
        counts[start:stop] = mask.sum(dim=1)
        del mask, mask_expanded
    return sums, counts


def kmeans_batched(
    X: torch.Tensor,
    num_clusters: int,
    distance: str = "euclidean",
    cluster_centers: Optional[torch.Tensor] = None,
    tol: float = 1e-4,
    tqdm_flag: bool = False,
    iter_limit: int = 0,
    device: torch.device = torch.device("cpu"),
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform batched k-means clustering

    Args:
        X: (torch.tensor) Input data of shape (b, n, d)
            b = batch size
            n = number of samples per batch
            d = dimensionality
        num_clusters: (int) Number of clusters
        distance: (str) Distance metric [currently only 'euclidean' supported]
        cluster_centers: (torch.tensor or None) Initial centers of shape (b, num_clusters, d)
            If None, will initialize randomly
        tol: (float) Convergence tolerance threshold [default: 1e-4]
        tqdm_flag: (bool) Whether to show progress bar and logs
        iter_limit: (int) Maximum number of iterations (0 = no limit)
        device: (torch.device) Device for computation
        seed: (int) Random seed for initialization

    Returns:
        choice_cluster: (torch.tensor) Cluster assignments of shape (b, n)
        cluster_centers: (torch.tensor) Final cluster centers of shape (b, num_clusters, d)

    Raises:
        NotImplementedError: If distance metric other than 'euclidean' is specified
        ValueError: If X is not 3-dimensional
    """
    # Validate input
    if X.ndim != 3:
        raise ValueError(
            f"Expected 3D input (b, n, d), got {X.ndim}D tensor with shape {X.shape}"
        )

    if distance != "euclidean":
        raise NotImplementedError(
            f"Distance '{distance}' not yet implemented for batched k-means. Only 'euclidean' is supported."
        )

    if tqdm_flag:
        print(f"Running batched k-means on {device}...")
        print(
            f"Input shape: {X.shape} (batch_size={X.shape[0]}, n_samples={X.shape[1]}, n_features={X.shape[2]})"
        )

    b, n, d = X.shape

    # Store original dtype
    original_dtype = X.dtype

    # Convert to float for computation
    X = X.float()

    # Transfer to device
    X = X.to(device)

    # Initialize cluster centers
    if cluster_centers is None:
        initial_state = initialize_batched(X, num_clusters, seed=seed)
    else:
        if tqdm_flag:
            print("Using provided cluster centers")

        if cluster_centers.shape != (b, num_clusters, d):
            raise ValueError(
                f"cluster_centers shape mismatch. Expected ({b}, {num_clusters}, {d}), "
                f"got {cluster_centers.shape}"
            )

        initial_state = cluster_centers.float().to(device)

    iteration = 0

    if tqdm_flag:
        tqdm_meter = tqdm(desc="[running batched kmeans]")

    # Tile sizes from the SHAPE alone.  Below _EXACT_PLAN_MAX_BYTES both come back as
    # "do not tile" and the helpers run the single-shot code verbatim, so every context
    # through 64k keeps the unconditionally-exact path.
    rows, groups = _plan_kmeans(b, n, d, num_clusters)

    # Main k-means loop
    while True:
        # Assign each sample to nearest cluster: (b, n).  Row-tiled above the ceiling;
        # the tiling cannot change a bit (see _assign_batched).
        choice_cluster = _assign_batched(X, initial_state, rows)

        # Store previous state for convergence check
        initial_state_pre = initial_state.clone()

        # Per-cluster sums and counts.  Group-tiled above the ceiling, never row-tiled:
        # cluster_sums reduces over n, and splitting n re-associates those additions.
        cluster_sums, cluster_counts = _accumulate_batched(
            X, choice_cluster, num_clusters, groups
        )

        # Identify empty clusters
        empty_clusters = cluster_counts == 0  # (b, num_clusters)

        # Compute new centers (avoid division by zero)
        cluster_counts_safe = cluster_counts.clamp(min=1).unsqueeze(
            -1
        )  # (b, num_clusters, 1)
        new_centers = cluster_sums / cluster_counts_safe  # (b, num_clusters, d)

        # Handle empty clusters with fully vectorized approach
        if empty_clusters.any():
            # Sample random indices for all potential empty clusters upfront
            random_indices = torch.randint(0, n, (b, num_clusters), device=X.device)

            # Gather random samples: (b, num_clusters, d)
            batch_idx = (
                torch.arange(b, device=X.device).unsqueeze(1).expand(-1, num_clusters)
            )
            random_samples = X[batch_idx, random_indices]

            # Replace only empty clusters using torch.where
            empty_mask = empty_clusters.unsqueeze(-1)  # (b, num_clusters, 1)
            new_centers = torch.where(empty_mask, random_samples, new_centers)

        # Update state
        initial_state = new_centers

        # Compute center shift for convergence check
        # Shape: (b, num_clusters, d) -> (b, num_clusters) -> (b,)
        center_shift = torch.sqrt(
            ((initial_state - initial_state_pre) ** 2).sum(dim=2)
        ).sum(dim=1)

        # Increment iteration counter
        iteration += 1

        # Update progress bar
        if tqdm_flag:
            max_shift = center_shift.max().item()
            mean_shift = center_shift.mean().item()
            tqdm_meter.set_postfix(
                iteration=f"{iteration}",
                max_shift=f"{max_shift ** 2:.6f}",
                mean_shift=f"{mean_shift ** 2:.6f}",
                tol=f"{tol:.6f}",
            )
            tqdm_meter.update()

        # Check convergence: all batches must meet tolerance
        if (center_shift**2 < tol).all():
            if tqdm_flag:
                print(f"\nConverged after {iteration} iterations")
            break

        # Check iteration limit
        if iter_limit != 0 and iteration >= iter_limit:
            if tqdm_flag:
                print(f"\nReached iteration limit: {iter_limit}")
            break

    if tqdm_flag:
        tqdm_meter.close()

    # Return results in original dtype
    return choice_cluster.to(original_dtype), initial_state.to(original_dtype)


# Actual PQ utils


def kmeans_loop_sklearn(
    data: torch.Tensor, k: int, max_iter: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch K-means clustering using sklearn (CPU-based).

    Args:
        data: [n_groups, n_samples, d] - torch.Tensor on GPU
        k: number of clusters
        max_iter: max iterations

    Returns:
        centroids: [n_groups, k, d] - torch.Tensor on GPU
        codes: [n_groups, n_samples] - torch.Tensor on GPU (int64)
    """
    from sklearn.cluster import MiniBatchKMeans

    n_groups, n_samples, d = data.shape

    # Move to CPU and convert to numpy
    data_cpu: np.ndarray = data.detach().cpu().float().numpy()

    # Initialize output arrays
    centroids_list: list = []
    codes_list: list = []

    # Run K-means for each group independently
    for i in range(n_groups):
        group_data: np.ndarray = data_cpu[i]

        # Use MiniBatchKMeans for efficiency with large datasets
        kmeans: MiniBatchKMeans = MiniBatchKMeans(
            n_clusters=k,
            max_iter=max_iter,
            batch_size=min(1024, n_samples),
            random_state=42,
            n_init=3,
        )

        # Fit and predict
        kmeans.fit(group_data)
        group_codes: np.ndarray = kmeans.predict(group_data)
        group_centroids: np.ndarray = kmeans.cluster_centers_

        centroids_list.append(group_centroids)
        codes_list.append(group_codes)

    # Stack and convert back to torch tensors
    centroids_np: np.ndarray = np.stack(centroids_list, axis=0)
    codes_np: np.ndarray = np.stack(codes_list, axis=0)

    # Move back to GPU
    centroids: torch.Tensor = torch.from_numpy(centroids_np).to(
        data.device, dtype=data.dtype
    )
    codes: torch.Tensor = torch.from_numpy(codes_np).to(data.device, dtype=torch.int64)

    return centroids, codes


def kmeans_loop_pytorch(
    data: torch.Tensor, k: int, max_iter: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch K-means clustering using PyTorch (GPU-based) with loop.

    This function loops over each group and runs kmeans independently.
    For better performance, consider using kmeans_batched_pytorch instead.

    Args:
        data: [n_groups, n_samples, d] - torch.Tensor on GPU
        k: number of clusters
        max_iter: max iterations

    Returns:
        centroids: [n_groups, k, d] - torch.Tensor on GPU
        codes: [n_groups, n_samples] - torch.Tensor on GPU (int64)
    """
    n_groups, n_samples, d = data.shape

    # Initialize output arrays
    centroids_list: list = []
    codes_list: list = []

    # Run K-means for each group independently
    for i in range(n_groups):
        group_data: torch.Tensor = data[i]

        # Use MiniBatchKMeans for efficiency with large datasets
        cluster_ids_x, cluster_centers = kmeans(
            X=group_data,
            num_clusters=k,
            distance="euclidean",
            device=data.device,
        )

        centroids_list.append(cluster_centers)
        codes_list.append(cluster_ids_x)

    centroids: torch.Tensor = torch.stack(centroids_list, dim=0)
    codes: torch.Tensor = torch.stack(codes_list, dim=0)

    return centroids, codes


def kmeans_batched_pytorch(
    data: torch.Tensor, k: int, max_iter: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch K-means clustering using PyTorch (GPU-based) with vectorized batching.

    This function processes all groups simultaneously using the batched kmeans
    implementation, which is more efficient than looping over groups.

    Args:
        data: [n_groups, n_samples, d] - torch.Tensor on GPU
        k: number of clusters
        max_iter: max iterations

    Returns:
        centroids: [n_groups, k, d] - torch.Tensor on GPU
        codes: [n_groups, n_samples] - torch.Tensor on GPU (int64)
    """
    n_groups, n_samples, d = data.shape

    # Use batched kmeans implementation
    codes, centroids = kmeans_batched(
        X=data,
        num_clusters=k,
        distance="euclidean",
        cluster_centers=None,
        tol=1e-4,
        tqdm_flag=False,
        iter_limit=max_iter,
        device=data.device,
        seed=None,
    )

    # Ensure codes are int64 for consistency with kmeans_loop_pytorch
    codes = codes.long()

    return centroids, codes


def ip2l2_augment(xb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Augment vectors to convert inner product search to L2 search.

    This augmentation allows using L2 distance for approximate inner product search.
    After augmentation, minimizing L2 distance is equivalent to maximizing inner product.

    Reference: Johnson et al., "Billion-scale similarity search with GPUs"

    Args:
        xb: [n_groups, n_samples, d] - vectors to augment

    Returns:
        xb_aug: [n_groups, n_samples, d+1] - augmented vectors
        phi: [n_groups, 1, 1] - normalization constant
    """
    n_groups, n_samples, d = xb.shape

    # Compute max squared norm per group
    norms_sq: torch.Tensor = (xb**2).sum(dim=2, keepdim=True)
    phi: torch.Tensor = norms_sq.max(dim=1, keepdim=True)[0]

    # Compute extra column: sqrt(phi - ||x||^2)
    extracol: torch.Tensor = torch.sqrt(phi - norms_sq)

    # Concatenate
    xb_aug: torch.Tensor = torch.cat([xb, extracol], dim=2)

    return xb_aug, phi


def ip2l2_augment_queries(xq: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Augment query vectors for IP to L2 conversion.

    The IP2L2 transformation is asymmetric:
    - Keys are augmented with sqrt(phi - ||x||^2)
    - Queries are augmented with 0

    This converts inner product search into L2 distance search:
    ||x_aug - q_aug||^2 = phi + ||q||^2 - 2<x, q>

    Args:
        xq: [n_groups, n_queries, d] - query vectors
        phi: [n_groups, 1, 1] - normalization constant (not used, kept for API compatibility)

    Returns:
        xq_aug: [n_groups, n_queries, d+1] - augmented query vectors with zero column
    """
    n_groups, n_queries, d = xq.shape

    # Augment queries with zero column
    zero_col: torch.Tensor = torch.zeros(
        n_groups, n_queries, 1, dtype=xq.dtype, device=xq.device
    )
    xq_aug: torch.Tensor = torch.cat([xq, zero_col], dim=2)

    return xq_aug


def compute_reconstruction_errors(
    original_keys: torch.Tensor,
    centroids: torch.Tensor,
    codebook: torch.Tensor,
    pq_sub_dim: int,
    use_ip_metric: bool = False,
) -> Dict[str, float]:
    """Compute reconstruction errors after product quantization.

    This function reconstructs keys from the quantized codebook and centroids,
    then computes various error metrics. Useful for debugging and validating
    PQ quality.

    Args:
        original_keys: [bsz, num_heads, n_keys, head_dim] - original key vectors
        centroids: [bsz, num_heads, n_subvec, cent_cnt, subvec_d] or
                   [bsz, num_heads, n_subvec, cent_cnt, subvec_d+1] for IP metric
        codebook: [bsz, n_keys, num_heads, n_subvec] - quantized codes
        pq_sub_dim: dimension of each subvector
        use_ip_metric: if True, centroids are augmented (d+1) and only first d dims are used

    Returns:
        Dict containing:
            - mse_error: Mean Squared Error
            - l2_error: L2 norm of difference
            - relative_error: L2 error normalized by original norm
    """
    bsz, num_heads, n_keys, head_dim = original_keys.shape
    n_subvec_per_head = head_dim // pq_sub_dim

    # Reshape original keys: [bsz, n_keys, num_heads, n_subvec, subvec_d]
    original_reshaped = original_keys.transpose(1, 2).reshape(
        bsz, n_keys, num_heads, n_subvec_per_head, pq_sub_dim
    )

    # Reconstruct keys from codebook
    reconstructed = torch.zeros_like(original_reshaped)
    for b in range(bsz):
        for h in range(num_heads):
            for s in range(n_subvec_per_head):
                codes = codebook[b, :, h, s]  # [n_keys]

                if use_ip_metric:
                    # For IP metric, only use first pq_sub_dim dimensions (ignore augmented dim)
                    cents = centroids[b, h, s, :, :pq_sub_dim]  # [cent_cnt, subvec_d]
                else:
                    # For Euclidean metric, use all dimensions
                    cents = centroids[b, h, s, :, :]  # [cent_cnt, subvec_d]

                reconstructed[b, :, h, s, :] = cents[codes]  # Gather centroids

    # Calculate error metrics
    diff = original_reshaped - reconstructed
    mse_error = torch.mean(diff**2).item()
    l2_error = torch.norm(diff).item()
    original_norm = torch.norm(original_reshaped).item()
    relative_error = l2_error / original_norm if original_norm > 0 else 0.0

    return {
        "mse_error": mse_error,
        "l2_error": l2_error,
        "relative_error": relative_error,
    }
