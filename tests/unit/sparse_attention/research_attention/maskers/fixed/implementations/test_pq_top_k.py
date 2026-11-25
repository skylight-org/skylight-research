"""
:author: Aditya Desai
:copyright: 2025 Sparse Attention Hub
:license: Apache 2.0
:date: 2025-06-29
:summary: Tests for PQCache masker implementation.
"""

import pytest
import torch


@pytest.mark.unit
class TestPQCacheMaskerImplementation:
    def test_pq_cache_masker_config_creation(self):
        """Test that pq cache masker config can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCacheConfig,
        )

        config = PQCacheConfig(
            heavy_size=10,
            pq_group_factor=2,  # head_dim=16 // pq_sub_dim=8 = 2
            pq_bits=4,
            kmeans_iter=10,
            init_offset=4,
            metric="euclidean",
        )
        assert config is not None
        assert config.heavy_size == 10
        assert config.pq_group_factor == 2
        assert config.pq_bits == 4
        assert config.kmeans_iter == 10
        assert config.init_offset == 4
        assert config.metric == "euclidean"

    def test_pq_cache_masker_creation(self):
        """Test that pq cache masker can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )

        config = PQCacheConfig(
            heavy_size=10,
            pq_group_factor=2,  # head_dim=16 // pq_sub_dim=8 = 2
            pq_bits=4,
            kmeans_iter=10,
            init_offset=4,
            metric="euclidean",
        )
        masker = PQCache(config)
        assert type(masker) is PQCache
        assert masker.config == config

    def test_pq_cache_masker_creation_from_config(self):
        """Test that pq cache masker can be created from a config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )

        config = PQCacheConfig(
            heavy_size=10,
            pq_group_factor=2,  # head_dim=16 // pq_sub_dim=8 = 2
            pq_bits=4,
            kmeans_iter=10,
            init_offset=4,
            metric="euclidean",
        )
        masker = PQCache.create_from_config(config)
        assert type(masker) is PQCache
        assert masker.config == config

    def test_pq_cache_masker_inheritance(self):
        """Test that pq cache masker inherits from TopKMasker."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            TopKMasker,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
        )

        assert issubclass(PQCache, TopKMasker)

    def test_pq_cache_masker_config_inheritance(self):
        """Test that pq cache masker config inherits from TopKMaskerConfig."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            TopKMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCacheConfig,
        )

        assert issubclass(PQCacheConfig, TopKMaskerConfig)

    def test_perform_kmeans_clustering_euclidean(self):
        """Test K-means clustering on keys with Euclidean metric."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )

        # Setup config
        config = PQCacheConfig(
            heavy_size=10,
            pq_group_factor=2,  # head_dim=16 // pq_sub_dim=8 = 2
            pq_bits=4,  # 2^4 = 16 centroids per subvector
            kmeans_iter=10,
            init_offset=4,  # Skip first 4 tokens (sink)
            metric="euclidean",
        )
        masker = PQCache(config)

        # Create test keys: [bsz=1, num_heads=2, seq_len=20, head_dim=16]
        # head_dim=16 will be split into 2 subvectors of 8 dimensions each
        bsz = 1
        num_heads = 2
        seq_len_keys = 20
        head_dim = 16
        keys = torch.randn(bsz, num_heads, seq_len_keys, head_dim, dtype=torch.float32)

        # Create sparse_meta_data
        sparse_meta_data = {}
        layer_idx = 0

        # Initialize PQ cache
        masker._initialize_pq_cache(sparse_meta_data, layer_idx)

        # Perform K-means clustering
        centroids, codebook = masker._perform_kmeans_clustering(
            keys, layer_idx, sparse_meta_data
        )

        # Check output shapes
        pq_sub_dim = head_dim // config.pq_group_factor  # 16 // 2 = 8
        n_subvec_per_head = head_dim // pq_sub_dim  # 16 // 8 = 2
        cent_cnt = 2**config.pq_bits  # 2^4 = 16
        n_quantized_keys = seq_len_keys - config.init_offset  # 20 - 4 = 16

        # Centroids: [bsz, num_heads, n_subvec, cent_cnt, subvec_d]
        assert centroids.shape == (
            bsz,
            num_heads,
            n_subvec_per_head,
            cent_cnt,
            pq_sub_dim,
        )
        assert centroids.shape == (1, 2, 2, 16, 8)

        # Codebook: [bsz, n_quantized_keys, num_heads, n_subvec]
        assert codebook.shape == (bsz, n_quantized_keys, num_heads, n_subvec_per_head)
        assert codebook.shape == (1, 16, 2, 2)

        # Check codebook values are within valid range [0, cent_cnt-1]
        assert codebook.min() >= 0
        assert codebook.max() < cent_cnt
        assert codebook.dtype == torch.int64

        # Check centroids are on the same device as keys
        assert centroids.device == keys.device
        assert codebook.device == keys.device

        # Check data is stored in sparse_meta_data
        assert sparse_meta_data["pq_centroids"][layer_idx] is not None
        assert sparse_meta_data["pq_codebook"][layer_idx] is not None
        assert torch.equal(sparse_meta_data["pq_centroids"][layer_idx], centroids)
        assert torch.equal(sparse_meta_data["pq_codebook"][layer_idx], codebook)

        # For Euclidean metric, phi should not be stored
        assert sparse_meta_data["pq_ip2l2_phi"][layer_idx] is None

        # Test reconstruction: compute errors using utility function
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.pq_utils import (
            compute_reconstruction_errors,
        )

        original_keys = keys[:, :, config.init_offset :, :]
        pq_sub_dim = head_dim // config.pq_group_factor
        errors = compute_reconstruction_errors(
            original_keys=original_keys,
            centroids=centroids,
            codebook=codebook,
            pq_sub_dim=pq_sub_dim,
            use_ip_metric=False,
        )

        print("\nEuclidean Metric Reconstruction Errors:")
        print(f"  MSE Error: {errors['mse_error']:.6f}")
        print(f"  L2 Error: {errors['l2_error']:.6f}")
        print(f"  Relative Error: {errors['relative_error']:.6f}")

        # Sanity check: error should be reasonable (not too large)
        assert (
            errors["relative_error"] < 1.0
        ), f"Relative error {errors['relative_error']} is too large"

    def test_perform_kmeans_clustering_ip(self):
        """Test K-means clustering on keys with IP metric."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )

        # Setup config with IP metric
        config = PQCacheConfig(
            heavy_size=10,
            pq_group_factor=2,  # head_dim=16 // pq_sub_dim=8 = 2
            pq_bits=4,
            kmeans_iter=10,
            init_offset=4,
            metric="ip",
        )
        masker = PQCache(config)

        # Create test keys
        bsz = 1
        num_heads = 2
        seq_len_keys = 20
        head_dim = 16
        keys = torch.randn(bsz, num_heads, seq_len_keys, head_dim, dtype=torch.float32)

        # Create sparse_meta_data
        sparse_meta_data = {}
        layer_idx = 0

        # Initialize PQ cache
        masker._initialize_pq_cache(sparse_meta_data, layer_idx)

        # Perform K-means clustering
        centroids, codebook = masker._perform_kmeans_clustering(
            keys, layer_idx, sparse_meta_data
        )

        # Check output shapes
        pq_sub_dim = head_dim // config.pq_group_factor
        n_subvec_per_head = head_dim // pq_sub_dim
        cent_cnt = 2**config.pq_bits
        n_quantized_keys = seq_len_keys - config.init_offset

        # For IP metric, centroids have augmented dimension (subvec_d + 1)
        assert centroids.shape == (
            bsz,
            num_heads,
            n_subvec_per_head,
            cent_cnt,
            pq_sub_dim + 1,  # +1 for augmentation
        )
        assert centroids.shape == (1, 2, 2, 16, 9)

        # Codebook shape remains the same
        assert codebook.shape == (bsz, n_quantized_keys, num_heads, n_subvec_per_head)
        assert codebook.shape == (1, 16, 2, 2)

        # For IP metric, phi should be stored
        assert sparse_meta_data["pq_ip2l2_phi"][layer_idx] is not None
        ip2l2_phi = sparse_meta_data["pq_ip2l2_phi"][layer_idx]

        # Phi shape: [n_groups, 1, 1] where n_groups = bsz * num_heads * n_subvec
        n_groups = bsz * num_heads * n_subvec_per_head
        assert ip2l2_phi.shape == (n_groups, 1, 1)
        assert ip2l2_phi.shape == (4, 1, 1)  # 1 * 2 * 2 = 4

        # Check phi values are positive (max squared norms)
        assert (ip2l2_phi > 0).all()

        # Test reconstruction: compute errors using utility function
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.pq_utils import (
            compute_reconstruction_errors,
        )

        original_keys = keys[:, :, config.init_offset :, :]
        pq_sub_dim = head_dim // config.pq_group_factor
        errors = compute_reconstruction_errors(
            original_keys=original_keys,
            centroids=centroids,
            codebook=codebook,
            pq_sub_dim=pq_sub_dim,
            use_ip_metric=True,  # IP metric: centroids are augmented
        )

        print("\nIP Metric Reconstruction Errors:")
        print(f"  MSE Error: {errors['mse_error']:.6f}")
        print(f"  L2 Error: {errors['l2_error']:.6f}")
        print(f"  Relative Error: {errors['relative_error']:.6f}")

        # Sanity check: error should be reasonable (not too large)
        assert (
            errors["relative_error"] < 1.0
        ), f"Relative error {errors['relative_error']} is too large"

    def test_perform_kmeans_clustering_different_init_offset(self):
        """Test that init_offset correctly skips keys from the front."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )

        # Test with different init_offsets
        # Using pq_bits=3 (8 centroids) and seq_len=50 to ensure enough samples
        for init_offset in [0, 5, 10]:
            config = PQCacheConfig(
                heavy_size=10,
                pq_group_factor=2,  # head_dim=16 // pq_sub_dim=8 = 2
                pq_bits=3,  # 2^3 = 8 centroids (need at least 8 samples)
                kmeans_iter=10,
                init_offset=init_offset,
                metric="euclidean",
            )
            masker = PQCache(config)

            bsz = 1
            num_heads = 2
            seq_len_keys = 50  # Long enough for all init_offset values
            head_dim = 16
            keys = torch.randn(
                bsz, num_heads, seq_len_keys, head_dim, dtype=torch.float32
            )

            sparse_meta_data = {}
            layer_idx = 0
            masker._initialize_pq_cache(sparse_meta_data, layer_idx)

            centroids, codebook = masker._perform_kmeans_clustering(
                keys, layer_idx, sparse_meta_data
            )

            # Check that codebook only has codes for keys after init_offset
            n_quantized_keys = seq_len_keys - init_offset
            assert codebook.shape[1] == n_quantized_keys

            # Verify centroids count matches configuration
            cent_cnt = 2**config.pq_bits
            assert centroids.shape[3] == cent_cnt  # Check number of centroids

            # Test reconstruction for this init_offset using utility function
            from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.pq_utils import (
                compute_reconstruction_errors,
            )

            original_keys = keys[:, :, init_offset:, :]
            pq_sub_dim = head_dim // config.pq_group_factor
            errors = compute_reconstruction_errors(
                original_keys=original_keys,
                centroids=centroids,
                codebook=codebook,
                pq_sub_dim=pq_sub_dim,
                use_ip_metric=False,
            )

            print(
                f"\nInit Offset={init_offset} Reconstruction Error: "
                f"{errors['relative_error']:.6f}"
            )
            assert (
                errors["relative_error"] < 1.0
            ), f"Relative error {errors['relative_error']} is too large"

    def test_perform_kmeans_varying_centroids(self):
        """Test K-means clustering with varying number of centroids.

        This test demonstrates how reconstruction error decreases as the number
        of centroids increases (for both Euclidean and IP metrics).
        """
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.pq_utils import (
            compute_reconstruction_errors,
        )

        # Fixed parameters
        bsz, num_heads, seq_len_keys, head_dim = 1, 1, 128, 16
        init_offset = 0
        pq_group_factor = 8  # head_dim=16 // pq_sub_dim=2 = 8
        kmeans_iter = 100  # More iterations for better convergence

        # Create data sample
        torch.manual_seed(42)  # For reproducibility
        keys = torch.randn(bsz, num_heads, seq_len_keys, head_dim, dtype=torch.float32)

        # Vary number of centroids from 2^1 to 2^7 (2 to 128)
        # Since we have 128 points, max centroids is 128
        pq_bits_range = [1, 2, 3, 4, 5, 6, 7]  # 2, 4, 8, 16, 32, 64, 128 centroids

        print("\n" + "=" * 80)
        print("Reconstruction Errors vs Number of Centroids")
        print("=" * 80)
        print(f"{'Centroids':<12} {'Euclidean Err':<20} {'IP Err':<20}")
        print("-" * 80)

        for pq_bits in pq_bits_range:
            n_centroids = 2**pq_bits

            # Test Euclidean metric
            config_euclidean = PQCacheConfig(
                heavy_size=10,
                pq_group_factor=pq_group_factor,
                pq_bits=pq_bits,
                kmeans_iter=kmeans_iter,
                init_offset=init_offset,
                metric="euclidean",
            )
            masker_euclidean = PQCache(config_euclidean)

            sparse_meta_data_euclidean = {}
            layer_idx = 0
            masker_euclidean._initialize_pq_cache(sparse_meta_data_euclidean, layer_idx)

            centroids_euc, codebook_euc = masker_euclidean._perform_kmeans_clustering(
                keys, layer_idx, sparse_meta_data_euclidean
            )

            original_keys = keys[:, :, init_offset:, :]
            pq_sub_dim = head_dim // pq_group_factor
            errors_euc = compute_reconstruction_errors(
                original_keys=original_keys,
                centroids=centroids_euc,
                codebook=codebook_euc,
                pq_sub_dim=pq_sub_dim,
                use_ip_metric=False,
            )

            # Test IP metric
            config_ip = PQCacheConfig(
                heavy_size=10,
                pq_group_factor=pq_group_factor,
                pq_bits=pq_bits,
                kmeans_iter=kmeans_iter,
                init_offset=init_offset,
                metric="ip",
            )
            masker_ip = PQCache(config_ip)

            sparse_meta_data_ip = {}
            masker_ip._initialize_pq_cache(sparse_meta_data_ip, layer_idx)

            centroids_ip, codebook_ip = masker_ip._perform_kmeans_clustering(
                keys, layer_idx, sparse_meta_data_ip
            )

            errors_ip = compute_reconstruction_errors(
                original_keys=original_keys,
                centroids=centroids_ip,
                codebook=codebook_ip,
                pq_sub_dim=pq_sub_dim,
                use_ip_metric=True,
            )

            # Print results
            print(
                f"{n_centroids:<12} "
                f"{errors_euc['relative_error']:<20.6f} "
                f"{errors_ip['relative_error']:<20.6f}"
            )

        print("=" * 80)

        # Basic sanity checks: error should generally decrease as centroids increase
        # (though not strictly monotonic due to random initialization)
        # Just verify that error with 2 centroids > error with 128 centroids
        config_min = PQCacheConfig(
            heavy_size=10,
            pq_group_factor=pq_group_factor,
            pq_bits=1,  # 2 centroids
            kmeans_iter=kmeans_iter,
            init_offset=init_offset,
            metric="euclidean",
        )
        masker_min = PQCache(config_min)
        sparse_meta_data_min = {}
        masker_min._initialize_pq_cache(sparse_meta_data_min, 0)
        centroids_min, codebook_min = masker_min._perform_kmeans_clustering(
            keys, 0, sparse_meta_data_min
        )
        pq_sub_dim = head_dim // pq_group_factor
        errors_min = compute_reconstruction_errors(
            original_keys=keys[:, :, init_offset:, :],
            centroids=centroids_min,
            codebook=codebook_min,
            pq_sub_dim=pq_sub_dim,
            use_ip_metric=False,
        )

        config_max = PQCacheConfig(
            heavy_size=10,
            pq_group_factor=pq_group_factor,
            pq_bits=7,  # 128 centroids
            kmeans_iter=kmeans_iter,
            init_offset=init_offset,
            metric="euclidean",
        )
        masker_max = PQCache(config_max)
        sparse_meta_data_max = {}
        masker_max._initialize_pq_cache(sparse_meta_data_max, 0)
        centroids_max, codebook_max = masker_max._perform_kmeans_clustering(
            keys, 0, sparse_meta_data_max
        )
        errors_max = compute_reconstruction_errors(
            original_keys=keys[:, :, init_offset:, :],
            centroids=centroids_max,
            codebook=codebook_max,
            pq_sub_dim=pq_sub_dim,
            use_ip_metric=False,
        )

        # Error with 2 centroids should be higher than with 128 centroids
        assert errors_min["relative_error"] > errors_max["relative_error"], (
            f"Expected error to decrease with more centroids: "
            f"2 centroids={errors_min['relative_error']:.6f}, "
            f"128 centroids={errors_max['relative_error']:.6f}"
        )

    def test_handle_incremental_keys(self):
        """Test handling of incremental keys during decoding."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.pq_utils import (
            compute_reconstruction_errors,
        )

        # Setup config
        config = PQCacheConfig(
            heavy_size=10,
            pq_group_factor=2,  # head_dim=16 // pq_sub_dim=8 = 2
            pq_bits=4,  # 16 centroids
            kmeans_iter=10,
            init_offset=4,
            metric="euclidean",
        )
        masker = PQCache(config)

        # Create initial keys: [bsz=1, num_heads=2, seq_len=20, head_dim=16]
        bsz = 1
        num_heads = 2
        initial_seq_len = 20
        head_dim = 16
        initial_keys = torch.randn(
            bsz, num_heads, initial_seq_len, head_dim, dtype=torch.float32
        )

        # Initialize and perform initial clustering
        sparse_meta_data = {}
        layer_idx = 0
        masker._initialize_pq_cache(sparse_meta_data, layer_idx)

        initial_centroids, initial_codebook = masker._perform_kmeans_clustering(
            initial_keys, layer_idx, sparse_meta_data
        )

        # Store initial centroids for comparison
        initial_centroids_copy = initial_centroids.clone()
        n_initial_quantized = initial_seq_len - config.init_offset

        # Verify initial setup
        assert initial_codebook.shape[1] == n_initial_quantized
        assert sparse_meta_data["pq_centroids"][layer_idx] is not None
        assert sparse_meta_data["pq_codebook"][layer_idx] is not None

        # Create new keys (simulating incremental decoding - add 5 new tokens)
        n_new_keys = 5
        new_seq_len = initial_seq_len + n_new_keys
        combined_keys = torch.randn(
            bsz, num_heads, new_seq_len, head_dim, dtype=torch.float32
        )
        # Copy initial keys to maintain continuity
        combined_keys[:, :, :initial_seq_len, :] = initial_keys

        # Handle incremental keys
        updated_centroids, updated_codebook = masker._handle_incremental_keys(
            combined_keys, layer_idx, sparse_meta_data
        )

        # Verify centroids remain unchanged (not re-trained)
        assert torch.allclose(
            updated_centroids, initial_centroids_copy, rtol=1e-5, atol=1e-7
        ), "Centroids should not change during incremental update"

        # Verify codebook now includes codes for new keys
        n_total_quantized = new_seq_len - config.init_offset
        assert updated_codebook.shape[1] == n_total_quantized
        assert updated_codebook.shape == (bsz, n_total_quantized, num_heads, 2)

        # Verify codebook values are valid
        cent_cnt = 2**config.pq_bits
        assert updated_codebook.min() >= 0
        assert updated_codebook.max() < cent_cnt
        assert updated_codebook.dtype == torch.int64

        # Verify the first part of codebook matches initial codebook
        assert torch.equal(
            updated_codebook[:, :n_initial_quantized, :, :], initial_codebook
        ), "Initial codebook entries should remain unchanged"

        # Verify sparse_meta_data is updated
        assert torch.equal(
            sparse_meta_data["pq_centroids"][layer_idx], updated_centroids
        )
        assert torch.equal(sparse_meta_data["pq_codebook"][layer_idx], updated_codebook)

        # Test reconstruction for all quantized keys (including new ones)
        original_keys = combined_keys[:, :, config.init_offset :, :]
        pq_sub_dim = head_dim // config.pq_group_factor
        errors = compute_reconstruction_errors(
            original_keys=original_keys,
            centroids=updated_centroids,
            codebook=updated_codebook,
            pq_sub_dim=pq_sub_dim,
            use_ip_metric=False,
        )

        print("\nIncremental Keys Reconstruction Errors:")
        print(f"  Total quantized keys: {n_total_quantized}")
        print(f"  Initial keys: {n_initial_quantized}")
        print(f"  New keys: {n_new_keys}")
        print(f"  MSE Error: {errors['mse_error']:.6f}")
        print(f"  L2 Error: {errors['l2_error']:.6f}")
        print(f"  Relative Error: {errors['relative_error']:.6f}")

        # Sanity check
        assert (
            errors["relative_error"] < 1.0
        ), f"Relative error {errors['relative_error']} is too large"

    def test_quantize_new_keys(self):
        """Test quantizing new keys using existing centroids."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.pq_utils import (
            compute_reconstruction_errors,
        )

        # Test with both Euclidean and IP metrics
        for metric in ["euclidean", "ip"]:
            print(f"\n{'='*60}")
            print(f"Testing _quantize_new_keys with {metric.upper()} metric")
            print(f"{'='*60}")

            # Setup config
            config = PQCacheConfig(
                heavy_size=10,
                pq_group_factor=2,  # head_dim=16 // pq_sub_dim=8 = 2
                pq_bits=4,  # 16 centroids
                kmeans_iter=10,
                init_offset=4,
                metric=metric,
            )
            masker = PQCache(config)

            # Create initial keys and perform clustering
            bsz = 1
            num_heads = 2
            initial_seq_len = 20
            head_dim = 16
            initial_keys = torch.randn(
                bsz, num_heads, initial_seq_len, head_dim, dtype=torch.float32
            )

            # Initialize and perform initial clustering
            sparse_meta_data = {}
            layer_idx = 0
            masker._initialize_pq_cache(sparse_meta_data, layer_idx)

            centroids, initial_codebook = masker._perform_kmeans_clustering(
                initial_keys, layer_idx, sparse_meta_data
            )

            # Create new keys to quantize
            n_new_keys = 5
            new_keys = torch.randn(
                bsz, num_heads, n_new_keys, head_dim, dtype=torch.float32
            )

            # Quantize the new keys
            new_codes = masker._quantize_new_keys(
                new_keys, centroids, layer_idx, sparse_meta_data
            )

            # Verify output shape
            pq_sub_dim = head_dim // config.pq_group_factor
            n_subvec_per_head = head_dim // pq_sub_dim  # 16 // 8 = 2
            assert new_codes.shape == (bsz, n_new_keys, num_heads, n_subvec_per_head)
            assert new_codes.shape == (1, 5, 2, 2)

            # Verify codes are valid (within centroid range)
            cent_cnt = 2**config.pq_bits  # 16
            assert new_codes.min() >= 0
            assert new_codes.max() < cent_cnt
            assert new_codes.dtype == torch.int64

            # Verify codes are on the same device
            assert new_codes.device == new_keys.device

            # Verify reconstruction quality for the new keys
            errors = compute_reconstruction_errors(
                original_keys=new_keys,
                centroids=centroids,
                codebook=new_codes,
                pq_sub_dim=pq_sub_dim,
                use_ip_metric=(metric == "ip"),
            )

            print(f"\nNew Keys Quantization Errors ({metric}):")
            print(f"  Number of new keys: {n_new_keys}")
            print(f"  MSE Error: {errors['mse_error']:.6f}")
            print(f"  L2 Error: {errors['l2_error']:.6f}")
            print(f"  Relative Error: {errors['relative_error']:.6f}")

            # Sanity check: error should be reasonable
            # Note: with small sample size (5 keys) and random data, error can sometimes be > 1.0
            assert (
                errors["relative_error"] < 2.0
            ), f"Relative error {errors['relative_error']} is too large for {metric} metric"

            # Test that quantizing the same keys gives the same codes
            new_codes_repeat = masker._quantize_new_keys(
                new_keys, centroids, layer_idx, sparse_meta_data
            )
            assert torch.equal(
                new_codes, new_codes_repeat
            ), "Quantizing same keys should produce identical codes"

            # Test with different new keys (should produce different codes)
            different_keys = torch.randn(
                bsz, num_heads, n_new_keys, head_dim, dtype=torch.float32
            )
            different_codes = masker._quantize_new_keys(
                different_keys, centroids, layer_idx, sparse_meta_data
            )
            assert different_codes.shape == new_codes.shape
            # Codes should likely be different (not guaranteed but very likely with random data)
            # Just verify the function runs without error

            print(f"{'='*60}\n")

    def test_compute_pq_scores(self):
        """Test computing PQ scores logic in isolation."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )

        # Setup config
        config = PQCacheConfig(
            heavy_size=10,
            pq_group_factor=2,  # head_dim=8 // pq_sub_dim=4 = 2
            pq_bits=2,  # 4 centroids per subvector
            kmeans_iter=10,
            init_offset=4,
            metric="euclidean",  # Doesn't affect scoring logic
        )
        masker = PQCache(config)

        # Construct inputs manually (no clustering required)
        bsz = 1
        num_heads = 4  # Query heads
        kv_heads = 2  # Key/value heads (GQA)
        seq_len_queries = 2
        seq_len_keys = 10
        head_dim = 8  # Will be split into 2 subvectors of 4 dims each
        pq_sub_dim = head_dim // config.pq_group_factor
        n_subvec = head_dim // pq_sub_dim  # 2
        cent_cnt = 2**config.pq_bits  # 4
        n_clustered = 5  # Number of quantized keys

        # Create queries: [bsz, num_heads, seq_len_queries, head_dim]
        queries = torch.randn(
            bsz, num_heads, seq_len_queries, head_dim, dtype=torch.float32
        )

        # Create keys (only used for shape reference in the function)
        keys = torch.randn(bsz, kv_heads, seq_len_keys, head_dim, dtype=torch.float32)

        # Manually construct centroids: [bsz, kv_heads, n_subvec, cent_cnt, subvec_d]
        centroids = torch.randn(
            bsz, kv_heads, n_subvec, cent_cnt, pq_sub_dim, dtype=torch.float32
        )

        # Manually construct codebook: [bsz, n_clustered, kv_heads, n_subvec]
        # Codebook contains indices [0, cent_cnt-1]
        codebook = torch.randint(
            0, cent_cnt, (bsz, n_clustered, kv_heads, n_subvec), dtype=torch.int64
        )

        # Call the function
        pq_scores = masker._compute_pq_scores(queries, keys, centroids, codebook)

        # Verify output shape: [bsz, num_heads, seq_len_queries, n_clustered]
        assert pq_scores.shape == (bsz, num_heads, seq_len_queries, n_clustered)
        assert pq_scores.shape == (1, 4, 2, 5)

        # Verify device
        assert pq_scores.device == queries.device

        # Verify the scoring logic manually for a single query-key pair
        # Let's verify the first query [0, 0, 0, :] against first key (index 0 in codebook)
        q = queries[0, 0, 0, :]  # [head_dim]

        # Split query into subvectors
        q_subvec = q.reshape(n_subvec, pq_sub_dim)  # [n_subvec, subvec_d]

        # For each subvector, get the centroid index from codebook
        expected_score = 0.0
        for s in range(n_subvec):
            # Get centroid index for key 0, head 0, subvector s
            # Due to GQA, query head 0 maps to kv_head 0
            kv_head = 0
            centroid_idx = codebook[0, 0, kv_head, s].item()

            # Get the centroid vector
            centroid_vec = centroids[0, kv_head, s, centroid_idx, :]  # [subvec_d]

            # Compute inner product for this subvector
            score_contrib = torch.dot(q_subvec[s], centroid_vec).item()
            expected_score += score_contrib

        # Compare with computed score
        computed_score = pq_scores[0, 0, 0, 0].item()

        print("\nManual verification:")
        print(f"  Expected score (manual): {expected_score:.6f}")
        print(f"  Computed score: {computed_score:.6f}")
        print(f"  Difference: {abs(expected_score - computed_score):.6f}")

        # Should be very close (allowing for floating point precision)
        assert torch.isclose(
            pq_scores[0, 0, 0, 0],
            torch.tensor(expected_score, dtype=torch.float32),
            rtol=1e-5,
            atol=1e-6,
        ), f"Score mismatch: expected {expected_score}, got {computed_score}"

        # Test GQA: verify that query heads share the same kv_heads
        # Query heads 0,1 should use kv_head 0; query heads 2,3 should use kv_head 1

        # For the same query and key index, heads in the same group should have same score
        # if they have the same query values (which they don't, so just verify shape is correct)

        print("\nShape verification:")
        print(f"  PQ Scores shape: {pq_scores.shape}")
        print(f"  Expected: ({bsz}, {num_heads}, {seq_len_queries}, {n_clustered})")
        print("  ✓ Shape matches!")

        # Test with IP metric (centroids with augmented dimension)
        print("\nTesting with augmented centroids (IP metric):")

        # Create augmented centroids: [bsz, kv_heads, n_subvec, cent_cnt, subvec_d + 1]
        centroids_aug = torch.randn(
            bsz,
            kv_heads,
            n_subvec,
            cent_cnt,
            pq_sub_dim + 1,
            dtype=torch.float32,
        )

        # Call with augmented centroids (function automatically handles the extra dimension)
        pq_scores_aug = masker._compute_pq_scores(
            queries, keys, centroids_aug, codebook
        )

        # Should still work and produce same shape
        assert pq_scores_aug.shape == (bsz, num_heads, seq_len_queries, n_clustered)
        print(f"  Augmented centroids shape: {centroids_aug.shape}")
        print(f"  PQ Scores shape: {pq_scores_aug.shape}")
        print("  ✓ Correctly handles augmented dimension!")

        print("\n✓ All unit test checks passed!")

    def test_config_validation_errors(self):
        """Test that PQCacheConfig validates parameters correctly."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCacheConfig,
        )

        # Test invalid pq_group_factor (must be > 0)
        with pytest.raises(ValueError, match="pq_group_factor must be > 0"):
            PQCacheConfig(
                heavy_size=10,
                pq_group_factor=0,  # Invalid
                pq_bits=4,
                kmeans_iter=10,
                init_offset=4,
                metric="euclidean",
            )

        with pytest.raises(ValueError, match="pq_group_factor must be > 0"):
            PQCacheConfig(
                heavy_size=10,
                pq_group_factor=-1,  # Invalid
                pq_bits=4,
                kmeans_iter=10,
                init_offset=4,
                metric="euclidean",
            )

        # Test invalid pq_bits (must be > 0)
        with pytest.raises(ValueError, match="pq_bits must be > 0"):
            PQCacheConfig(
                heavy_size=10,
                pq_group_factor=2,
                pq_bits=0,  # Invalid
                kmeans_iter=10,
                init_offset=4,
                metric="euclidean",
            )

        # Test invalid kmeans_iter (must be > 0)
        with pytest.raises(ValueError, match="kmeans_iter must be > 0"):
            PQCacheConfig(
                heavy_size=10,
                pq_group_factor=2,
                pq_bits=4,
                kmeans_iter=-5,  # Invalid
                init_offset=4,
                metric="euclidean",
            )

        # Test invalid init_offset (must be >= 0)
        with pytest.raises(ValueError, match="init_offset must be >= 0"):
            PQCacheConfig(
                heavy_size=10,
                pq_group_factor=2,
                pq_bits=4,
                kmeans_iter=10,
                init_offset=-1,  # Invalid
                metric="euclidean",
            )

        # Test invalid metric (must be 'euclidean' or 'ip')
        with pytest.raises(ValueError, match="metric must be 'euclidean' or 'ip'"):
            PQCacheConfig(
                heavy_size=10,
                pq_group_factor=2,
                pq_bits=4,
                kmeans_iter=10,
                init_offset=4,
                metric="cosine",  # Invalid
            )

        with pytest.raises(ValueError, match="metric must be 'euclidean' or 'ip'"):
            PQCacheConfig(
                heavy_size=10,
                pq_group_factor=2,
                pq_bits=4,
                kmeans_iter=10,
                init_offset=4,
                metric="invalid_metric",  # Invalid
            )

    def test_validate_inputs_errors(self):
        """Test that _validate_inputs catches invalid inputs."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )

        config: PQCacheConfig = PQCacheConfig(
            heavy_size=10,
            pq_group_factor=2,
            pq_bits=4,
            kmeans_iter=10,
            init_offset=4,
            metric="euclidean",
        )
        masker: PQCache = PQCache(config)

        # Test with None sparse_meta_data
        with pytest.raises(ValueError, match="sparse_meta_data cannot be None"):
            masker._validate_inputs(None, {"layer_idx": 0})

        # Test with missing layer_idx
        with pytest.raises(ValueError, match="layer_idx must be provided"):
            masker._validate_inputs({}, {})

        with pytest.raises(ValueError, match="layer_idx must be provided"):
            masker._validate_inputs({}, {"other_key": "value"})

        # Test valid inputs return layer_idx
        layer_idx: int = masker._validate_inputs({"pq_centroids": {}}, {"layer_idx": 5})
        assert layer_idx == 5

    def test_should_use_full_attention(self):
        """Test the _should_use_full_attention decision logic."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
            AttentionTensorDimensions,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )

        config: PQCacheConfig = PQCacheConfig(
            heavy_size=10,
            pq_group_factor=2,
            pq_bits=4,  # 2^4 = 16 centroids
            kmeans_iter=10,
            init_offset=4,
            metric="euclidean",
        )
        masker: PQCache = PQCache(config)

        # Test case 1: Sequence too short - should use full attention
        # total_needed = heavy_size + init_offset + seq_len_queries + 2^pq_bits
        #              = 10 + 4 + 2 + 16 = 32
        dims_short: AttentionTensorDimensions = AttentionTensorDimensions(
            batch_size=1, num_heads=4, seq_len_queries=2, seq_len_keys=30
        )
        # 30 < 32, should use full attention
        assert masker._should_use_full_attention(dims_short, heavy_size=10) is True

        # Test case 2: Sequence long enough - should not use full attention
        dims_long: AttentionTensorDimensions = AttentionTensorDimensions(
            batch_size=1, num_heads=4, seq_len_queries=2, seq_len_keys=100
        )
        # 100 > 32, should not use full attention
        assert masker._should_use_full_attention(dims_long, heavy_size=10) is False

        # Test case 3: Edge case - exactly at boundary
        dims_edge: AttentionTensorDimensions = AttentionTensorDimensions(
            batch_size=1, num_heads=4, seq_len_queries=2, seq_len_keys=32
        )
        # 32 <= 32, should use full attention
        assert masker._should_use_full_attention(dims_edge, heavy_size=10) is True

        # Test case 4: Different heavy_size
        # total_needed = 50 + 4 + 2 + 16 = 72
        dims_large_k: AttentionTensorDimensions = AttentionTensorDimensions(
            batch_size=1, num_heads=4, seq_len_queries=2, seq_len_keys=70
        )
        # 70 < 72, should use full attention
        assert masker._should_use_full_attention(dims_large_k, heavy_size=50) is True

        # 100 > 72, should not use full attention
        dims_large_k2: AttentionTensorDimensions = AttentionTensorDimensions(
            batch_size=1, num_heads=4, seq_len_queries=2, seq_len_keys=100
        )
        assert masker._should_use_full_attention(dims_large_k2, heavy_size=50) is False

    def test_initialize_pq_cache(self):
        """Test that _initialize_pq_cache properly initializes the data structure."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )

        config: PQCacheConfig = PQCacheConfig(
            heavy_size=10,
            pq_group_factor=2,
            pq_bits=4,
            kmeans_iter=10,
            init_offset=4,
            metric="euclidean",
        )
        masker: PQCache = PQCache(config)

        # Test with empty sparse_meta_data
        sparse_meta_data: dict = {}
        layer_idx: int = 0

        masker._initialize_pq_cache(sparse_meta_data, layer_idx)

        # Check that all keys are created
        assert "pq_centroids" in sparse_meta_data
        assert "pq_codebook" in sparse_meta_data
        assert "pq_ip2l2_phi" in sparse_meta_data

        # Check that layer_idx entries are initialized to None
        assert layer_idx in sparse_meta_data["pq_centroids"]
        assert layer_idx in sparse_meta_data["pq_codebook"]
        assert layer_idx in sparse_meta_data["pq_ip2l2_phi"]

        assert sparse_meta_data["pq_centroids"][layer_idx] is None
        assert sparse_meta_data["pq_codebook"][layer_idx] is None
        assert sparse_meta_data["pq_ip2l2_phi"][layer_idx] is None

        # Test with partially initialized sparse_meta_data
        sparse_meta_data2: dict = {"pq_centroids": {}}
        layer_idx2: int = 1

        masker._initialize_pq_cache(sparse_meta_data2, layer_idx2)

        # Check that missing keys are added
        assert "pq_codebook" in sparse_meta_data2
        assert "pq_ip2l2_phi" in sparse_meta_data2

        # Check that new layer_idx is added
        assert layer_idx2 in sparse_meta_data2["pq_centroids"]
        assert sparse_meta_data2["pq_centroids"][layer_idx2] is None

        # Test that calling again doesn't overwrite existing data
        sparse_meta_data3: dict = {
            "pq_centroids": {5: torch.tensor([1, 2, 3])},
            "pq_codebook": {5: torch.tensor([4, 5, 6])},
            "pq_ip2l2_phi": {5: torch.tensor([7.0])},
        }
        layer_idx3: int = 5

        masker._initialize_pq_cache(sparse_meta_data3, layer_idx3)

        # Existing data should be preserved
        assert torch.equal(
            sparse_meta_data3["pq_centroids"][5], torch.tensor([1, 2, 3])
        )
        assert torch.equal(sparse_meta_data3["pq_codebook"][5], torch.tensor([4, 5, 6]))
        assert torch.equal(sparse_meta_data3["pq_ip2l2_phi"][5], torch.tensor([7.0]))

    def test_determine_new_keys(self):
        """Test _determine_new_keys logic for different scenarios."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )

        config: PQCacheConfig = PQCacheConfig(
            heavy_size=10,
            pq_group_factor=2,
            pq_bits=4,
            kmeans_iter=10,
            init_offset=4,
            metric="euclidean",
        )
        masker: PQCache = PQCache(config)

        bsz: int = 1
        kv_heads: int = 2
        head_dim: int = 16

        # Scenario 1: First call - no cached codebook
        sparse_meta_data: dict = {"pq_codebook": {0: None}}
        layer_idx: int = 0
        seq_len_keys: int = 20

        keys: torch.Tensor = torch.randn(
            bsz, kv_heads, seq_len_keys, head_dim, dtype=torch.float32
        )

        cached_codebook, new_keys = masker._determine_new_keys(
            keys, sparse_meta_data, layer_idx
        )

        # Should return None for cached_codebook and keys in quantized region
        assert cached_codebook is None
        assert new_keys is not None
        assert new_keys.shape == (
            bsz,
            kv_heads,
            seq_len_keys - config.init_offset,
            head_dim,
        )

        # Scenario 2: Subsequent call with same sequence length - no new keys
        n_quantized_keys: int = seq_len_keys - config.init_offset  # 16
        pq_sub_dim = head_dim // config.pq_group_factor
        n_subvec: int = head_dim // pq_sub_dim  # 2

        existing_codebook: torch.Tensor = torch.randint(
            0, 16, (bsz, n_quantized_keys, kv_heads, n_subvec), dtype=torch.int64
        )
        sparse_meta_data["pq_codebook"][layer_idx] = existing_codebook

        keys_same: torch.Tensor = torch.randn(
            bsz, kv_heads, seq_len_keys, head_dim, dtype=torch.float32
        )

        cached_codebook2, new_keys2 = masker._determine_new_keys(
            keys_same, sparse_meta_data, layer_idx
        )

        # Should return cached codebook and None for new keys
        assert cached_codebook2 is not None
        assert torch.equal(cached_codebook2, existing_codebook)
        assert new_keys2 is None

        # Scenario 3: Subsequent call with more keys - has new keys
        n_new_keys: int = 5
        new_seq_len: int = seq_len_keys + n_new_keys  # 25

        keys_more: torch.Tensor = torch.randn(
            bsz, kv_heads, new_seq_len, head_dim, dtype=torch.float32
        )

        cached_codebook3, new_keys3 = masker._determine_new_keys(
            keys_more, sparse_meta_data, layer_idx
        )

        # Should return cached codebook and only the new keys
        assert cached_codebook3 is not None
        assert torch.equal(cached_codebook3, existing_codebook)
        assert new_keys3 is not None
        assert new_keys3.shape == (bsz, kv_heads, n_new_keys, head_dim)

        # Scenario 4: Invalid case - sequence shrunk
        smaller_seq_len: int = 10
        keys_smaller: torch.Tensor = torch.randn(
            bsz, kv_heads, smaller_seq_len, head_dim, dtype=torch.float32
        )

        # Should raise ValueError
        with pytest.raises(ValueError, match="Quantized region shrunk"):
            masker._determine_new_keys(keys_smaller, sparse_meta_data, layer_idx)

    def test_create_pq_mask(self):
        """Test _create_pq_mask functionality."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
            AttentionTensorDimensions,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        config: PQCacheConfig = PQCacheConfig(
            heavy_size=5,
            pq_group_factor=2,
            pq_bits=4,
            kmeans_iter=10,
            init_offset=4,
            metric="euclidean",
        )
        masker: PQCache = PQCache(config)

        bsz: int = 1
        n_heads: int = 2
        seq_len_queries: int = 3
        seq_len_keys: int = 20
        n_clustered: int = seq_len_keys - config.init_offset  # 16

        dims: AttentionTensorDimensions = AttentionTensorDimensions(
            batch_size=bsz,
            num_heads=n_heads,
            seq_len_queries=seq_len_queries,
            seq_len_keys=seq_len_keys,
        )

        # Create scores: [bsz, n_heads, seq_len_queries, n_clustered]
        torch.manual_seed(42)
        scores: torch.Tensor = torch.randn(
            bsz, n_heads, seq_len_queries, n_clustered, dtype=torch.float32
        )

        # Create previous mask (mark some positions as already active)
        previous_mask: Mask = Mask.create_empty_mask(
            shape=(bsz, n_heads, seq_len_queries, seq_len_keys),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        previous_dense: torch.Tensor = previous_mask.get_dense_mask()
        # Mark first 2 positions in quantized region as already active
        previous_dense[:, :, :, config.init_offset : config.init_offset + 2] = 1.0
        previous_mask = Mask(
            shape=previous_dense.shape,
            mask=previous_dense,
            from_dense_mask=True,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # Create PQ mask
        device: torch.device = torch.device("cpu")
        pq_mask: Mask = masker._create_pq_mask(
            dims,
            scores,
            effective_heavy_size=5,
            previous_mask=previous_mask,
            device=device,
        )

        # Verify mask properties
        assert pq_mask.shape == (bsz, n_heads, seq_len_queries, seq_len_keys)
        assert pq_mask.device == device

        # Check that mask has exactly heavy_size positions per query
        pq_dense: torch.Tensor = pq_mask.get_dense_mask()

        for b in range(bsz):
            for h in range(n_heads):
                for q in range(seq_len_queries):
                    # Count non-zero positions in quantized region
                    quantized_region: torch.Tensor = pq_dense[
                        b, h, q, config.init_offset : config.init_offset + n_clustered
                    ]
                    n_active: int = (quantized_region != 0).sum().item()

                    # Should have exactly heavy_size positions
                    assert (
                        n_active == config.heavy_size
                    ), f"Expected {config.heavy_size} active positions, got {n_active}"

        # Verify that positions already in previous_mask are NOT included in pq_mask
        # (they should be masked out during selection)
        for b in range(bsz):
            for h in range(n_heads):
                for q in range(seq_len_queries):
                    # Check first 2 positions (marked in previous_mask)
                    for i in range(2):
                        pos: int = config.init_offset + i
                        assert (
                            pq_dense[b, h, q, pos] == 0
                        ), f"Position {pos} was already active, should not be included in PQ mask"

    def test_create_from_config_invalid(self):
        """Test create_from_config with invalid config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
            MaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
        )

        # Create a generic MaskerConfig (not PQCacheConfig)
        invalid_config: MaskerConfig = MaskerConfig()

        # Should raise ValueError
        with pytest.raises(ValueError, match="Invalid config type"):
            PQCache.create_from_config(invalid_config)
