"""
:author: Aditya Desai
:copyright: 2025 Sparse Attention Hub
:license: Apache 2.0
:date: 2025-07-03
:summary: Tests for mask. This file is part of the Sparse Attention Hub project.
"""

import numpy as np
import pytest
import torch

from sparse_attention_hub.sparse_attention.utils.mask import Mask


@pytest.mark.unit
class TestMask:
    def test_create_mask_from_dense_mask(self):
        shape = (3, 5)
        mask = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 1.0],
            ]
        )
        mask_object = Mask.create_mask_from_dense_mask(shape, mask, dtype=torch.float32)
        assert mask_object.shape == shape
        assert torch.allclose(mask_object.mask, mask)
        assert mask_object.from_dense_mask

    def test_get_index_mask(self):
        shape = (3, 5)
        ptr = torch.tensor([0, 5, 7, 9])
        indices = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3])
        data = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        index_mask_object = Mask.create_mask_from_indices(
            shape, indices, ptr, data, dtype=torch.float32
        )
        index_mask, index_ptr, index_data = index_mask_object.get_index_mask()
        assert torch.allclose(index_mask, indices)
        assert torch.allclose(index_ptr, ptr)
        assert torch.allclose(index_data, data)
        assert index_mask_object.from_index

    def test_getters_dense_index(self):
        shape = (3, 5)
        mask = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 1.0],
            ]
        )
        dense_mask_object = Mask.create_mask_from_dense_mask(
            shape, mask, dtype=torch.float32
        )
        indices, ptr, data = dense_mask_object.get_index_mask()

        index_mask_object = Mask.create_mask_from_indices(
            shape, indices, ptr, data, dtype=torch.float32
        )
        dense_mask = index_mask_object.get_dense_mask()
        assert torch.allclose(dense_mask, mask)

    def test_getters_index_dense(self):
        shape = (3, 5)
        ptr = torch.tensor([0, 5, 7, 9])
        indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 12, 13])
        data = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        index_mask_object = Mask.create_mask_from_indices(
            shape, indices, ptr, data, dtype=torch.float32
        )

        dense_mask = index_mask_object.get_dense_mask()
        dense_mask_object = Mask.create_mask_from_dense_mask(
            shape, dense_mask, dtype=torch.float32
        )
        _indices, _ptr, _data = dense_mask_object.get_index_mask()
        assert _indices.shape == indices.shape
        assert _ptr.shape == ptr.shape
        assert _data.shape == data.shape
        assert torch.allclose(indices, _indices)
        assert torch.allclose(_ptr, ptr)
        assert torch.allclose(_data, data)

    def test_getters_dense_index_n_dims(self):
        n_dims = 3
        shape = tuple([5] * n_dims)
        mask = (torch.rand(shape) > 0.5) * torch.rand(shape)
        dense_mask_object = Mask.create_mask_from_dense_mask(
            shape, mask, dtype=torch.float32
        )
        indices, ptr, data = dense_mask_object.get_index_mask()

        for i in range(np.prod(shape[:-1])):
            start = ptr[i]
            end = ptr[i + 1]
            data = data[start:end]
            indices = indices[start:end]
            assert torch.allclose(data, mask.view(-1)[indices])

    def test_create_empty_mask_dense(self):
        shape = (3, 5, 7)
        mask_object = Mask.create_empty_mask(
            shape, dtype=torch.float32, device=torch.device("cpu")
        )
        assert mask_object.shape == shape
        assert mask_object.is_empty
        # Empty mask optimization - no actual data is stored
        assert mask_object.mask is None
        assert mask_object.indices is None
        assert mask_object.ptr is None

    def test_create_empty_mask_index(self):
        shape = (3, 5, 7)
        mask_object = Mask.create_empty_mask(
            shape, dtype=torch.float32, device=torch.device("cpu")
        )
        mask = mask_object.get_dense_mask()
        assert mask.shape == shape
        assert mask.sum() == 0
        assert mask_object.is_empty
        # Empty mask optimization - stored as flag, not actual sparse data
        assert mask_object.mask is None
        assert mask_object.indices is None
        assert mask_object.ptr is None

    def test_apply_mask(self):
        shape = (3, 5, 7)
        input = torch.rand(shape)
        dense_mask = (torch.rand(shape) > 0.5).float()

        true_answer = input * dense_mask

        dense_mask_object = Mask.create_mask_from_dense_mask(
            shape, dense_mask, dtype=torch.float32
        )
        indices, ptr, data = dense_mask_object.get_index_mask()
        index_mask_object = Mask.create_mask_from_indices(
            shape, indices, ptr, data, dtype=torch.float32
        )

        dense_masked_data = dense_mask_object.apply_mask(input, mode="dense")
        index_masked_data = index_mask_object.apply_mask(input, mode="dense")

        assert torch.allclose(dense_masked_data, true_answer)
        assert torch.allclose(index_masked_data, true_answer)

    def test_apply_inv_mask(self):
        shape = (3, 5, 7)
        input = torch.rand(shape)
        # Create a mask with non-zero values to test inverse operation
        dense_mask = torch.rand(shape) * 0.5 + 0.5  # Values in [0.5, 1.0]

        # Set some values to zero to test zero handling
        zero_mask = (torch.rand(shape) > 0.3).float()
        dense_mask = dense_mask * zero_mask

        # Calculate expected result
        true_answer = torch.zeros_like(input)
        non_zero_mask = dense_mask != 0
        true_answer[non_zero_mask] = input[non_zero_mask] * (
            1.0 / dense_mask[non_zero_mask]
        )

        dense_mask_object = Mask.create_mask_from_dense_mask(
            shape, dense_mask, dtype=torch.float32
        )
        indices, ptr, data = dense_mask_object.get_index_mask()
        index_mask_object = Mask.create_mask_from_indices(
            shape, indices, ptr, data, dtype=torch.float32
        )

        dense_inv_masked_data = dense_mask_object.apply_inv_mask(input, mode="dense")
        index_inv_masked_data = index_mask_object.apply_inv_mask(input, mode="dense")

        assert torch.allclose(dense_inv_masked_data, true_answer)
        assert torch.allclose(index_inv_masked_data, true_answer)

    def test_apply_inv_mask_empty(self):
        shape = (2, 3)
        input = torch.rand(shape)

        empty_mask = Mask.create_empty_mask(
            shape, dtype=torch.float32, device=torch.device("cpu")
        )
        result = empty_mask.apply_inv_mask(input, mode="dense")

        # Empty mask now returns input tensor directly (no masking applied)
        assert torch.allclose(result, input)

    def test_apply_inv_mask_all_ones(self):
        shape = (2, 3)
        input = torch.rand(shape)

        ones_mask = torch.ones(shape)
        mask_object = Mask.create_mask_from_dense_mask(
            shape, ones_mask, dtype=torch.float32
        )
        result = mask_object.apply_inv_mask(input, mode="dense")

        # When mask is all ones, inverse should be the same as input
        assert torch.allclose(result, input)

    def test_create_from_row_wise_idx_basic_dense(self):
        """Test basic functionality for dense mode."""
        shape = (2, 5)
        row_wise_idx = torch.tensor([[0, 2, 4], [1, 3, 4]])
        data = torch.tensor([[1.0, 0.5, 0.8], [0.6, 0.9, 0.3]])

        mask = Mask.create_from_row_wise_idx(
            shape, row_wise_idx, data, mask_type="dense", dtype=torch.float32
        )

        assert mask.shape == shape
        assert mask.from_dense_mask
        assert not mask.from_index

        # Check values are set correctly
        dense_mask = mask.get_dense_mask()
        expected = torch.tensor([[1.0, 0.0, 0.5, 0.0, 0.8], [0.0, 0.6, 0.0, 0.9, 0.3]])
        assert torch.allclose(dense_mask, expected)

    def test_create_from_row_wise_idx_basic_sparse(self):
        """Test basic functionality for sparse mode."""
        shape = (2, 5)
        row_wise_idx = torch.tensor([[0, 2, 4], [1, 3, 4]])
        data = torch.tensor([[1.0, 0.5, 0.8], [0.6, 0.9, 0.3]])

        mask = Mask.create_from_row_wise_idx(
            shape,
            row_wise_idx,
            data,
            mask_type="index",
            dtype=torch.float32,
            mode="sparse",
        )

        assert mask.shape == shape
        assert not mask.from_dense_mask
        assert mask.from_index

        # Check sparse representation
        indices, ptr, values = mask.get_index_mask()
        expected_indices = torch.tensor([0, 2, 4, 6, 8, 9])  # Flat indices
        expected_ptr = torch.tensor([0, 3, 6])
        expected_values = torch.tensor([1.0, 0.5, 0.8, 0.6, 0.9, 0.3])

        assert torch.allclose(indices, expected_indices)
        assert torch.allclose(ptr, expected_ptr)
        assert torch.allclose(values, expected_values)

    def test_create_from_row_wise_idx_sparse_index_preserves_unique_input_order(self):
        """Sparse index mode should preserve row-wise order when indices are unique."""
        shape = (2, 6)
        row_wise_idx = torch.tensor([[3, 1, 4], [2, 0, 5]])
        data = torch.tensor([[0.3, 0.1, 0.4], [1.2, 1.0, 1.5]])

        mask = Mask.create_from_row_wise_idx(
            shape,
            row_wise_idx,
            data,
            mask_type="index",
            dtype=torch.float32,
            mode="sparse",
        )

        indices, ptr, values = mask.get_index_mask()

        torch.testing.assert_close(indices, torch.tensor([3, 1, 4, 8, 6, 11]))
        torch.testing.assert_close(ptr, torch.tensor([0, 3, 6]))
        torch.testing.assert_close(values, data.reshape(-1))

    def test_create_from_row_wise_idx_single_element(self):
        """Test with single element per row."""
        shape = (3, 4)
        row_wise_idx = torch.tensor([[1], [0], [3]])
        data = torch.tensor([[0.5], [0.8], [0.2]])

        mask = Mask.create_from_row_wise_idx(
            shape, row_wise_idx, data, mask_type="dense", dtype=torch.float32
        )

        dense_mask = mask.get_dense_mask()
        expected = torch.tensor(
            [[0.0, 0.5, 0.0, 0.0], [0.8, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.2]]
        )
        assert torch.allclose(dense_mask, expected)

    def test_create_from_row_wise_idx_with_invalid_indices(self):
        """Test handling of invalid indices (e.g., -1 padding)."""
        shape = (2, 5)
        row_wise_idx = torch.tensor([[0, 2, -1], [1, -1, -1]])  # -1 as padding
        data = torch.tensor([[1.0, 0.5, 0.0], [0.6, 0.0, 0.0]])

        mask = Mask.create_from_row_wise_idx(
            shape,
            row_wise_idx,
            data,
            mask_type="dense",
            dtype=torch.float32,
            use_padding=True,
            mode="sparse",
        )

        dense_mask = mask.get_dense_mask()
        expected = torch.tensor([[1.0, 0.0, 0.5, 0.0, 0.0], [0.0, 0.6, 0.0, 0.0, 0.0]])
        assert torch.allclose(dense_mask, expected)

    def test_create_from_row_wise_idx_sparse_padding_preserves_valid_input_order(self):
        """Padding mode should filter -1 values without reordering valid entries."""
        shape = (2, 6)
        row_wise_idx = torch.tensor([[3, -1, 1, 4], [2, 0, -1, 5]])
        data = torch.tensor([[0.3, 0.0, 0.1, 0.4], [1.2, 1.0, 0.0, 1.5]])

        mask = Mask.create_from_row_wise_idx(
            shape,
            row_wise_idx,
            data,
            mask_type="index",
            dtype=torch.float32,
            use_padding=True,
            mode="sparse",
        )

        indices, ptr, values = mask.get_index_mask()

        torch.testing.assert_close(indices, torch.tensor([3, 1, 4, 8, 6, 11]))
        torch.testing.assert_close(ptr, torch.tensor([0, 3, 6]))
        torch.testing.assert_close(
            values, torch.tensor([0.3, 0.1, 0.4, 1.2, 1.0, 1.5])
        )

    def test_create_from_row_wise_idx_multidimensional(self):
        """Test with multi-dimensional batch shape."""
        shape = (2, 3, 4)  # batch_size=2, seq_len=3, feature_dim=4
        row_wise_idx = torch.tensor(
            [[[0, 2], [1, 3], [0, 1]], [[1, 3], [0, 2], [2, 3]]]
        )
        data = torch.tensor(
            [[[1.0, 0.5], [0.8, 0.3], [0.2, 0.7]], [[0.4, 0.9], [0.6, 0.1], [0.8, 0.4]]]
        )

        mask = Mask.create_from_row_wise_idx(
            shape, row_wise_idx, data, mask_type="dense", dtype=torch.float32
        )

        dense_mask = mask.get_dense_mask()

        # Check first batch
        assert torch.allclose(dense_mask[0, 0], torch.tensor([1.0, 0.0, 0.5, 0.0]))
        assert torch.allclose(dense_mask[0, 1], torch.tensor([0.0, 0.8, 0.0, 0.3]))
        assert torch.allclose(dense_mask[0, 2], torch.tensor([0.2, 0.7, 0.0, 0.0]))

        # Check second batch
        assert torch.allclose(dense_mask[1, 0], torch.tensor([0.0, 0.4, 0.0, 0.9]))
        assert torch.allclose(dense_mask[1, 1], torch.tensor([0.6, 0.0, 0.1, 0.0]))
        assert torch.allclose(dense_mask[1, 2], torch.tensor([0.0, 0.0, 0.8, 0.4]))

    def test_create_from_row_wise_idx_consistency(self):
        """Test that dense and sparse modes produce equivalent results."""
        shape = (3, 6)
        row_wise_idx = torch.tensor([[0, 2, 5], [1, 3, 4], [0, 1, 2]])
        data = torch.tensor([[1.0, 0.5, 0.8], [0.6, 0.9, 0.3], [0.4, 0.7, 0.2]])

        dense_mask = Mask.create_from_row_wise_idx(
            shape, row_wise_idx, data, mask_type="dense", dtype=torch.float32
        )
        sparse_mask = Mask.create_from_row_wise_idx(
            shape, row_wise_idx, data, mask_type="index", dtype=torch.float32
        )

        # Convert both to dense representation and compare
        dense_from_dense = dense_mask.get_dense_mask()
        dense_from_sparse = sparse_mask.get_dense_mask()

        assert torch.allclose(dense_from_dense, dense_from_sparse)

    def test_create_from_row_wise_idx_error_invalid_shape(self):
        """Test error handling for invalid shapes."""
        shape = ()  # Empty shape
        row_wise_idx = torch.tensor([[0, 1]])
        data = torch.tensor([[1.0, 0.5]])

        with pytest.raises(ValueError, match="shape must have at least one dimension"):
            Mask.create_from_row_wise_idx(
                shape, row_wise_idx, data, mask_type="dense", dtype=torch.float32
            )

    def test_create_from_row_wise_idx_error_mismatched_shapes(self):
        """Test error handling for mismatched input shapes."""
        shape = (2, 5)
        row_wise_idx = torch.tensor([[0, 2, 4]])  # Wrong batch size
        data = torch.tensor([[1.0, 0.5, 0.8], [0.6, 0.9, 0.3]])

        with pytest.raises(ValueError, match="row_wise_idx.shape must be"):
            Mask.create_from_row_wise_idx(
                shape, row_wise_idx, data, mask_type="dense", dtype=torch.float32
            )

    def test_create_from_row_wise_idx_error_data_shape_mismatch(self):
        """Test error handling when data shape doesn't match row_wise_idx shape."""
        shape = (2, 5)
        row_wise_idx = torch.tensor([[0, 2, 4], [1, 3, 4]])
        data = torch.tensor([[1.0, 0.5]])  # Wrong shape

        with pytest.raises(
            ValueError, match="data.shape must match row_wise_idx.shape"
        ):
            Mask.create_from_row_wise_idx(
                shape, row_wise_idx, data, mask_type="dense", dtype=torch.float32
            )

    # why comment this test?
    # The original check is removed since it unnecessarily causes GPU-CPU sync when running on GPU
    # in most cases, the indices are valid, so this check is unnecessary and causes unnecessary sync.

    # def test_create_from_row_wise_idx_error_out_of_bounds(self):
    #     """Test error handling for out-of-bounds indices."""
    #     shape = (2, 5)
    #     row_wise_idx = torch.tensor([[0, 2, 5], [1, 3, 4]])  # 5 is out of bounds
    #     data = torch.tensor([[1.0, 0.5, 0.8], [0.6, 0.9, 0.3]])

    #     with pytest.raises(
    #         ValueError, match="All valid indices in row_wise_idx must be in range"
    #     ):
    #         Mask.create_from_row_wise_idx(shape, row_wise_idx, data)

    def test_create_from_row_wise_idx_error_invalid_type(self):
        """Test error handling for invalid mask_type parameter."""
        shape = (2, 5)
        row_wise_idx = torch.tensor([[0, 2, 4], [1, 3, 4]])
        data = torch.tensor([[1.0, 0.5, 0.8], [0.6, 0.9, 0.3]])

        with pytest.raises(ValueError, match="type must be 'index' or 'dense'"):
            Mask.create_from_row_wise_idx(
                shape, row_wise_idx, data, mask_type="invalid", dtype=torch.float32
            )

    def test_create_from_row_wise_idx_empty_mask(self):
        """Test creating empty mask with all -1 indices."""
        shape = (2, 5)
        row_wise_idx = torch.tensor([[-1, -1, -1], [-1, -1, -1]])
        data = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        mask = Mask.create_from_row_wise_idx(
            shape,
            row_wise_idx,
            data,
            mask_type="dense",
            dtype=torch.float32,
            use_padding=True,
            mode="sparse",
        )

        # is_empty is set only when creating the mask empty
        # assert mask.is_empty
        dense_mask = mask.get_dense_mask()
        assert torch.allclose(dense_mask, torch.zeros(shape))

    def test_create_from_row_wise_idx_device_consistency(self):
        """Test that output tensors are on the same device as input."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        shape = (2, 5)
        row_wise_idx = torch.tensor([[0, 2, 4], [1, 3, 4]], device=device)
        data = torch.tensor([[1.0, 0.5, 0.8], [0.6, 0.9, 0.3]], device=device)

        mask = Mask.create_from_row_wise_idx(
            shape, row_wise_idx, data, mask_type="dense", dtype=torch.float32
        )
        assert mask.get_dense_mask().device == row_wise_idx.device

    def test_merge_mask_basic(self):
        """Test basic merge functionality."""
        shape = (2, 5)

        # Create first mask
        mask1 = Mask.create_mask_from_dense_mask(
            shape,
            torch.tensor(
                [[1.0, 0.0, 0.5, 0.0, 0.0], [0.0, 0.8, 0.0, 0.0, 0.3]],
                dtype=torch.float32,
            ),
            dtype=torch.float32,
        )

        # Create second mask
        mask2 = Mask.create_mask_from_dense_mask(
            shape,
            torch.tensor(
                [[0.0, 0.2, 0.5, 0.0, 0.4], [0.1, 0.0, 0.0, 0.6, 0.0]],
                dtype=torch.float32,
            ),
            dtype=torch.float32,
        )

        # Merge masks
        merged = mask1.merge_mask(mask2, inplace=False, mode="dense")

        # Expected result: union of indices with data addition
        expected_dense = torch.tensor(
            [[1.0, 0.2, 1.0, 0.0, 0.4], [0.1, 0.8, 0.0, 0.6, 0.3]]  # 0.5 + 0.5 = 1.0
        )

        assert torch.allclose(merged.get_dense_mask(), expected_dense)
        assert merged.from_dense_mask
        assert not merged.from_index

        merged = mask1.merge_mask(mask2, inplace=False, mode="sparse")

        # Expected result: union of indices with data addition
        expected_dense = torch.tensor(
            [[1.0, 0.2, 1.0, 0.0, 0.4], [0.1, 0.8, 0.0, 0.6, 0.3]]  # 0.5 + 0.5 = 1.0
        )

        assert torch.allclose(merged.get_dense_mask(), expected_dense)
        assert merged.from_index
        assert not merged.from_dense_mask

    def test_merge_mask_with_capping(self):
        """Test merge with capping (always [0.0, 1.0])."""
        shape = (2, 5)

        # Create first mask
        mask1 = Mask.create_mask_from_dense_mask(
            shape,
            torch.tensor(
                [[1.0, 0.0, 0.5, 0.0, 0.0], [0.0, 0.8, 0.0, 0.0, 0.3]],
                dtype=torch.float32,
            ),
            dtype=torch.float32,
        )

        # Create second mask
        mask2 = Mask.create_mask_from_dense_mask(
            shape,
            torch.tensor(
                [[0.0, 0.2, 0.8, 0.0, 0.4], [0.1, 0.0, 0.0, 0.6, 0.0]],
                dtype=torch.float32,
            ),
            dtype=torch.float32,
        )

        # Merge masks with capping
        merged = mask1.merge_mask(mask2, inplace=False, mode="dense")

        # Expected result with capping (now always [0.0, 1.0])
        expected_dense = torch.tensor(
            [
                [1.0, 0.2, 1.0, 0.0, 0.4],  # 0.5 + 0.8 = 1.3 -> capped to 1.0
                [0.1, 0.8, 0.0, 0.6, 0.3],
            ]
        )

        assert torch.allclose(merged.get_dense_mask(), expected_dense)

    def test_merge_mask_inplace(self):
        """Test in-place merge."""
        shape = (2, 5)

        # Create first mask
        mask1 = Mask.create_mask_from_dense_mask(
            shape,
            torch.tensor(
                [[1.0, 0.0, 0.5, 0.0, 0.0], [0.0, 0.8, 0.0, 0.0, 0.3]],
                dtype=torch.float32,
            ),
            dtype=torch.float32,
        )

        # Create second mask
        mask2 = Mask.create_mask_from_dense_mask(
            shape,
            torch.tensor(
                [[0.0, 0.2, 0.5, 0.0, 0.4], [0.1, 0.0, 0.0, 0.6, 0.0]],
                dtype=torch.float32,
            ),
            dtype=torch.float32,
        )

        # Merge masks in-place
        original_id = id(mask1)
        result = mask1.merge_mask(mask2, inplace=True, mode="sparse")

        # Check that the same object is returned
        assert id(result) == original_id
        assert result is mask1

        # Check that mask1 is updated
        assert mask1.from_index
        assert not mask1.from_dense_mask

        # Expected result
        expected_dense = torch.tensor(
            [[1.0, 0.2, 1.0, 0.0, 0.4], [0.1, 0.8, 0.0, 0.6, 0.3]]
        )

        assert torch.allclose(mask1.get_dense_mask(), expected_dense)

    def test_merge_mask_empty_masks(self):
        """Test merging with empty masks."""
        shape = (2, 3)

        # Create empty mask
        empty_mask = Mask.create_empty_mask(
            shape, dtype=torch.float32, device=torch.device("cpu")
        )

        # Create non-empty mask
        non_empty_mask = Mask.create_mask_from_dense_mask(
            shape,
            torch.tensor([[1.0, 0.0, 0.5], [0.0, 0.8, 0.0]], dtype=torch.float32),
            dtype=torch.float32,
        )

        # Merge empty with non-empty
        merged1 = empty_mask.merge_mask(non_empty_mask, inplace=False, mode="dense")
        merged2 = non_empty_mask.merge_mask(empty_mask, inplace=False, mode="dense")

        # Both should be equal to the non-empty mask
        assert torch.allclose(merged1.get_dense_mask(), non_empty_mask.get_dense_mask())
        assert torch.allclose(merged2.get_dense_mask(), non_empty_mask.get_dense_mask())

        # Merge two empty masks
        merged_empty = empty_mask.merge_mask(
            Mask.create_empty_mask(
                shape, dtype=torch.float32, device=torch.device("cpu")
            ),
            inplace=False,
            mode="dense",
        )
        assert merged_empty.is_empty

    def test_merge_mask_multidimensional(self):
        """Test merge with multi-dimensional masks."""
        shape = (2, 3, 4)

        # Create masks with known patterns
        mask1_dense = torch.tensor(
            [
                [[1.0, 0.0, 0.5, 0.0], [0.0, 0.8, 0.0, 0.3], [0.2, 0.0, 0.0, 0.0]],
                [[0.0, 0.4, 0.0, 0.0], [0.6, 0.0, 0.0, 0.0], [0.0, 0.0, 0.9, 0.0]],
            ]
        )

        mask2_dense = torch.tensor(
            [
                [[0.0, 0.2, 0.5, 0.0], [0.1, 0.0, 0.0, 0.3], [0.0, 0.0, 0.4, 0.0]],
                [[0.3, 0.0, 0.0, 0.7], [0.0, 0.0, 0.5, 0.0], [0.0, 0.8, 0.0, 0.0]],
            ]
        )

        mask1 = Mask.create_mask_from_dense_mask(
            shape, mask1_dense, dtype=torch.float32
        )
        mask2 = Mask.create_mask_from_dense_mask(
            shape, mask2_dense, dtype=torch.float32
        )

        # Merge
        merged = mask1.merge_mask(mask2, inplace=False, mode="dense")

        # Expected: element-wise addition with capping to [0.0, 1.0]
        expected_dense = torch.clamp(mask1_dense + mask2_dense, min=0.0, max=1.0)

        assert torch.allclose(merged.get_dense_mask(), expected_dense)

    def test_merge_mask_sparse_representations(self):
        """Test merge with sparse representations."""
        shape = (2, 5)

        # Create sparse mask 1
        indices1 = torch.tensor([0, 2, 6, 8])
        ptr1 = torch.tensor([0, 2, 4])
        data1 = torch.tensor([1.0, 0.5, 0.8, 0.3])
        mask1 = Mask.create_mask_from_indices(
            shape, indices1, ptr1, data1, dtype=torch.float32
        )

        # Create sparse mask 2
        indices2 = torch.tensor([1, 2, 5, 7])
        ptr2 = torch.tensor([0, 2, 4])
        data2 = torch.tensor([0.2, 0.5, 0.1, 0.6])
        mask2 = Mask.create_mask_from_indices(
            shape, indices2, ptr2, data2, dtype=torch.float32
        )

        # Merge
        merged = mask1.merge_mask(mask2, inplace=False, mode="dense")

        # Expected dense representation (capped to [0.0, 1.0])
        expected_dense = torch.clamp(
            mask1.get_dense_mask() + mask2.get_dense_mask(), min=0.0, max=1.0
        )
        assert torch.allclose(merged.get_dense_mask(), expected_dense)

    def test_merge_mask_shape_mismatch(self):
        """Test error handling for shape mismatch."""
        shape1 = (2, 5)
        shape2 = (3, 5)

        mask1 = Mask.create_empty_mask(
            shape1, dtype=torch.float32, device=torch.device("cpu")
        )
        mask2 = Mask.create_empty_mask(
            shape2, dtype=torch.float32, device=torch.device("cpu")
        )

        with pytest.raises(
            ValueError, match="Cannot merge masks with different shapes"
        ):
            mask1.merge_mask(mask2, inplace=False, mode="dense")

    def test_merge_mask_negative_capping(self):
        """Test merge with negative values and capping (always [0.0, 1.0])."""
        shape = (2, 3)

        # Create masks with negative values
        mask1 = Mask.create_mask_from_dense_mask(
            shape,
            torch.tensor([[1.0, -0.5, 0.0], [0.0, 0.8, -0.3]], dtype=torch.float32),
            dtype=torch.float32,
        )

        mask2 = Mask.create_mask_from_dense_mask(
            shape,
            torch.tensor([[0.0, 0.2, -0.4], [-0.1, 0.0, 0.6]], dtype=torch.float32),
            dtype=torch.float32,
        )

        # Merge with capping (now always [0.0, 1.0])
        merged = mask1.merge_mask(mask2, inplace=False, mode="dense")

        # Expected result (capped to [0.0, 1.0])
        expected_dense = torch.clamp(
            mask1.get_dense_mask() + mask2.get_dense_mask(), min=0.0, max=1.0
        )

        assert torch.allclose(merged.get_dense_mask(), expected_dense)

    def test_merge_mask_extreme_capping(self):
        """Test merge with large values and capping (always [0.0, 1.0])."""
        shape = (2, 3)

        mask1 = Mask.create_mask_from_dense_mask(
            shape,
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32),
            dtype=torch.float32,
        )

        mask2 = Mask.create_mask_from_dense_mask(
            shape,
            torch.tensor([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]], dtype=torch.float32),
            dtype=torch.float32,
        )

        # Test with capping (now always [0.0, 1.0])
        merged = mask1.merge_mask(mask2, inplace=False, mode="dense")

        # Expected: all values clamped to [0.0, 1.0]
        expected_dense = torch.clamp(
            mask1.get_dense_mask() + mask2.get_dense_mask(), min=0.0, max=1.0
        )

        assert torch.allclose(merged.get_dense_mask(), expected_dense)

    def test_merge_mask_single_element(self):
        """Test merge with single element masks."""
        shape = (1, 1)

        mask1 = Mask.create_mask_from_dense_mask(
            shape, torch.tensor([[0.5]], dtype=torch.float32), dtype=torch.float32
        )
        mask2 = Mask.create_mask_from_dense_mask(
            shape, torch.tensor([[0.3]], dtype=torch.float32), dtype=torch.float32
        )

        merged = mask1.merge_mask(mask2, inplace=False, mode="dense")

        expected_dense = torch.tensor([[0.8]])  # 0.5 + 0.3 = 0.8
        assert torch.allclose(merged.get_dense_mask(), expected_dense)

    def test_merge_mask_device_consistency(self):
        """Test that merged masks maintain device consistency."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        shape = (2, 3)

        mask1 = Mask.create_mask_from_dense_mask(
            shape,
            torch.tensor(
                [[1.0, 0.0, 0.5], [0.0, 0.8, 0.0]], device=device, dtype=torch.float32
            ),
            dtype=torch.float32,
        )
        true_device = mask1.get_dense_mask().device
        mask2 = Mask.create_mask_from_dense_mask(
            shape,
            torch.tensor(
                [[0.0, 0.2, 0.5], [0.1, 0.0, 0.6]], device=device, dtype=torch.float32
            ),
            dtype=torch.float32,
        )

        merged = mask1.merge_mask(mask2, inplace=False, mode="dense")

        assert merged.get_dense_mask().device == true_device
        indices, ptr, data = merged.get_index_mask()
        assert indices.device == true_device
        assert ptr.device == true_device
        assert data.device == true_device

    # ========== Full Mask Tests ==========

    def test_create_full_mask_basic(self):
        """Test basic full mask creation."""
        shape = (2, 3)
        full_mask = Mask.create_full_mask(
            shape, dtype=torch.float32, device=torch.device("cpu")
        )

        assert full_mask.shape == shape
        assert full_mask.dtype == torch.float32
        assert full_mask.is_full
        assert full_mask.is_full_mask()
        assert not full_mask.from_dense_mask
        assert not full_mask.from_index
        assert not full_mask.is_empty

        # Verify no actual data is stored
        assert full_mask.mask is None
        assert full_mask.indices is None
        assert full_mask.ptr is None
        assert full_mask.data is None

    def test_create_full_mask_different_shapes(self):
        """Test full mask creation with different shapes."""
        shapes = [(5,), (2, 4), (3, 2, 5), (2, 3, 4, 5)]

        for shape in shapes:
            full_mask = Mask.create_full_mask(
                shape, dtype=torch.float32, device=torch.device("cpu")
            )
            assert full_mask.shape == shape
            assert full_mask.is_full_mask()

            # Verify get_dense_mask returns all ones
            dense = full_mask.get_dense_mask()
            assert dense.shape == shape
            assert torch.all(dense == 1.0)

    def test_create_full_mask_different_dtypes(self):
        """Test full mask creation with different data types."""
        shape = (2, 3)
        dtypes = [torch.float32, torch.float64, torch.float16]

        for dtype in dtypes:
            full_mask = Mask.create_full_mask(
                shape, dtype=dtype, device=torch.device("cpu")
            )
            assert full_mask.dtype == dtype
            assert full_mask.is_full_mask()

            # Verify get_dense_mask respects dtype
            dense = full_mask.get_dense_mask()
            assert dense.dtype == dtype

    @pytest.mark.skip(
        reason="Auto detection is removed since it causes unnecessary sync when running on GPU"
    )
    def test_is_full_mask_various_cases(self):
        """Test is_full_mask() method with various mask types."""
        shape = (2, 3)

        # Test with full mask
        full_mask = Mask.create_full_mask(
            shape, dtype=torch.float32, device=torch.device("cpu")
        )
        assert full_mask.is_full_mask()

        # Test with empty mask
        empty_mask = Mask.create_empty_mask(shape)
        assert not empty_mask.is_full_mask()

        # Test with partial mask
        partial_mask = Mask.create_mask_from_dense_mask(
            shape,
            torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
            dtype=torch.float32,
        )
        assert not partial_mask.is_full_mask()

        # Test with all-ones mask (should be detected as full)
        ones_mask = Mask.create_mask_from_dense_mask(
            shape, torch.ones(shape, dtype=torch.float32)
        )
        assert ones_mask.is_full_mask()

    @pytest.mark.skip(
        reason="Auto detection is removed since it causes unnecessary sync when running on GPU"
    )
    def test_full_mask_auto_detection_dense(self):
        """Test auto-detection of full masks from dense representation."""
        shape = (2, 3)

        # Create dense mask with all 1.0 values
        ones_tensor = torch.ones(shape)
        mask = Mask.create_mask_from_dense_mask(shape, ones_tensor, dtype=torch.float32)

        # Should be auto-detected as full
        assert mask.is_full_mask()
        assert mask.is_full

        # Original mask data should be freed
        assert mask.mask is None

        # Create dense mask with some non-1.0 values
        partial_tensor = torch.tensor([[1.0, 0.5, 1.0], [1.0, 1.0, 1.0]])
        mask = Mask.create_mask_from_dense_mask(
            shape, partial_tensor, dtype=torch.float32
        )

        # Should NOT be detected as full
        assert not mask.is_full_mask()
        assert not mask.is_full
        assert mask.mask is not None

    @pytest.mark.skip(
        reason="Auto detection is removed since it causes unnecessary sync when running on GPU"
    )
    def test_full_mask_auto_detection_sparse(self):
        """Test auto-detection of full masks from sparse representation."""
        shape = (2, 3)
        total_size = 6

        # Create sparse representation of full mask (all positions with 1.0)
        indices = torch.arange(total_size, dtype=torch.long)
        ptr = torch.tensor([0, 3, 6], dtype=torch.long)
        data = torch.ones(total_size, dtype=torch.float32)

        mask = Mask.create_mask_from_indices(
            shape, indices, ptr, data, dtype=torch.float32
        )

        # Should be auto-detected as full
        assert mask.is_full_mask()
        assert mask.is_full

        # Original sparse data should be freed
        assert mask.indices is None
        assert mask.ptr is None
        assert mask.data is None

        # Create sparse representation with partial coverage
        partial_indices = torch.tensor([0, 2, 4], dtype=torch.long)
        partial_ptr = torch.tensor([0, 2, 3], dtype=torch.long)
        partial_data = torch.ones(3, dtype=torch.float32)

        mask = Mask.create_mask_from_indices(
            shape, partial_indices, partial_ptr, partial_data, dtype=torch.float32
        )

        # Should NOT be detected as full
        assert not mask.is_full_mask()
        assert not mask.is_full
        assert mask.indices is not None

    def test_full_mask_get_dense_mask(self):
        """Test get_dense_mask() optimization for full masks."""
        shape = (2, 3)
        full_mask = Mask.create_full_mask(
            shape, dtype=torch.float32, device=torch.device("cpu")
        )

        dense = full_mask.get_dense_mask()

        assert dense.shape == shape
        assert dense.dtype == full_mask.dtype
        assert torch.all(dense == 1.0)

        # Should be efficient (no actual data stored)
        assert full_mask.mask is None

    def test_full_mask_get_index_mask(self):
        """Test get_index_mask() optimization for full masks."""
        shape = (2, 3)
        full_mask = Mask.create_full_mask(
            shape, dtype=torch.float32, device=torch.device("cpu")
        )

        indices, ptr, data = full_mask.get_index_mask()

        # Should generate complete sparse representation
        expected_indices = torch.arange(6, dtype=torch.long)  # All positions
        expected_ptr = torch.tensor([0, 3, 6], dtype=torch.long)
        expected_data = torch.ones(6, dtype=full_mask.dtype)

        assert torch.equal(indices, expected_indices)
        assert torch.equal(ptr, expected_ptr)
        assert torch.equal(data, expected_data)

        # Should be generated on-demand (no actual data stored)
        assert full_mask.indices is None

    def test_full_mask_apply_mask_no_op(self):
        """Test that apply_mask() is a no-op for full masks."""
        shape = (2, 3)
        full_mask = Mask.create_full_mask(
            shape, dtype=torch.float32, device=torch.device("cpu")
        )

        input_tensor = torch.randn(shape)
        result = full_mask.apply_mask(input_tensor, mode="dense")

        # Should return input tensor directly (no-op)
        assert torch.equal(result, input_tensor)
        assert result is input_tensor  # Same object

    def test_full_mask_apply_inv_mask_no_op(self):
        """Test that apply_inv_mask() is a no-op for full masks."""
        shape = (2, 3)
        full_mask = Mask.create_full_mask(
            shape, dtype=torch.float32, device=torch.device("cpu")
        )

        input_tensor = torch.randn(shape)
        result = full_mask.apply_inv_mask(input_tensor, mode="dense")

        # Should return input tensor directly (no-op)
        assert torch.equal(result, input_tensor)
        assert result is input_tensor  # Same object

    def test_full_mask_is_empty_false(self):
        """Test that is_empty() returns False for full masks."""
        shape = (2, 3)
        full_mask = Mask.create_full_mask(
            shape, dtype=torch.float32, device=torch.device("cpu")
        )

        assert not full_mask.is_empty

    def test_full_mask_merge_optimization(self):
        """Test merge_mask() optimization when one mask is full."""
        shape = (2, 3)
        full_mask = Mask.create_full_mask(
            shape, dtype=torch.float32, device=torch.device("cpu")
        )

        # Create a partial mask
        partial_mask = Mask.create_mask_from_dense_mask(
            shape,
            torch.tensor([[1.0, 0.0, 0.5], [0.0, 0.8, 0.0]], dtype=torch.float32),
            dtype=torch.float32,
        )

        # Merge full with partial - result should be full
        merged1 = full_mask.merge_mask(partial_mask, inplace=False, mode="dense")
        assert merged1.is_full_mask()

        # Merge partial with full - result should be full
        merged2 = partial_mask.merge_mask(full_mask, inplace=False, mode="dense")
        assert merged2.is_full_mask()

        # Merge full with full - result should be full
        merged3 = full_mask.merge_mask(full_mask, inplace=False, mode="dense")
        assert merged3.is_full_mask()

    def test_full_mask_merge_inplace_optimization(self):
        """Test in-place merge optimization with full masks."""
        shape = (2, 3)
        full_mask = Mask.create_full_mask(
            shape, dtype=torch.float32, device=torch.device("cpu")
        )

        # Create a partial mask
        partial_mask = Mask.create_mask_from_dense_mask(
            shape,
            torch.tensor([[1.0, 0.0, 0.5], [0.0, 0.8, 0.0]], dtype=torch.float32),
            dtype=torch.float32,
        )

        # In-place merge with full mask
        original_id = id(partial_mask)
        result = partial_mask.merge_mask(full_mask, inplace=True, mode="dense")

        # Should return the same object, now converted to full
        assert id(result) == original_id
        assert result.is_full_mask()
        assert result.is_full

    @pytest.mark.skip(
        reason="Auto detection is removed since it causes unnecessary sync when running on GPU"
    )
    def test_full_mask_merge_result_detection(self):
        """Test detection of full mask results during merge."""
        shape = (2, 3)

        # Create two masks that together cover all positions with 1.0
        mask1 = Mask.create_mask_from_dense_mask(
            shape,
            torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=torch.float32),
            dtype=torch.float32,
        )
        mask2 = Mask.create_mask_from_dense_mask(
            shape,
            torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
            dtype=torch.float32,
        )

        # Merge should detect the result is full
        merged = mask1.merge_mask(mask2, inplace=False, mode="dense")
        assert merged.is_full_mask()

    def test_full_mask_repr(self):
        """Test string representation includes is_full flag."""
        shape = (2, 3)
        full_mask = Mask.create_full_mask(
            shape, dtype=torch.float32, device=torch.device("cpu")
        )

        repr_str = repr(full_mask)
        assert "is_full=True" in repr_str
        assert f"shape={shape}" in repr_str

    def test_full_mask_multidimensional(self):
        """Test full mask functionality with multi-dimensional tensors."""
        shape = (2, 3, 4)
        full_mask = Mask.create_full_mask(
            shape, dtype=torch.float32, device=torch.device("cpu")
        )

        # Test basic properties
        assert full_mask.shape == shape
        assert full_mask.is_full_mask()

        # Test get_dense_mask
        dense = full_mask.get_dense_mask()
        assert dense.shape == shape
        assert torch.all(dense == 1.0)

        # Test get_index_mask
        indices, ptr, data = full_mask.get_index_mask()
        expected_size = 2 * 3 * 4
        assert indices.numel() == expected_size
        assert torch.all(data == 1.0)

        # Test apply_mask no-op
        input_tensor = torch.randn(shape)
        result = full_mask.apply_mask(input_tensor, mode="dense")
        assert torch.equal(result, input_tensor)

    def test_full_mask_device_consistency(self):
        """Test full mask device consistency."""
        shape = (2, 3)
        full_mask = Mask.create_full_mask(
            shape, dtype=torch.float32, device=torch.device("cpu")
        )

        # Test get_dense_mask device
        dense = full_mask.get_dense_mask()
        # Full masks create new tensors, so they use default device
        assert dense.device.type == "cpu"  # Default device for torch.ones()

        # Test get_index_mask device
        indices, ptr, data = full_mask.get_index_mask()
        assert indices.device.type == "cpu"  # Default device
        assert ptr.device.type == "cpu"
        assert data.device.type == "cpu"

    def test_full_mask_edge_cases(self):
        """Test full mask edge cases."""
        # Single element mask
        shape = (1, 1)
        full_mask = Mask.create_full_mask(
            shape, dtype=torch.float32, device=torch.device("cpu")
        )
        assert full_mask.is_full_mask()

        dense = full_mask.get_dense_mask()
        assert dense.shape == shape
        assert torch.all(dense == 1.0)

        # Large shape
        shape = (10, 20)
        full_mask = Mask.create_full_mask(
            shape, dtype=torch.float32, device=torch.device("cpu")
        )
        assert full_mask.is_full_mask()

        # Apply mask should still be no-op
        input_tensor = torch.randn(shape)
        result = full_mask.apply_mask(input_tensor, mode="dense")
        assert torch.equal(result, input_tensor)

    @pytest.mark.skip(
        reason="Auto detection is removed since it causes unnecessary sync when running on GPU"
    )
    def test_full_mask_auto_detection_edge_cases(self):
        """Test auto-detection edge cases."""
        # Empty shape should not crash
        shape = (0, 5)
        try:
            empty_tensor = torch.ones(shape)
            mask = Mask.create_mask_from_dense_mask(
                shape, empty_tensor, dtype=torch.float32
            )
            # Should handle gracefully
        except Exception:
            # If it fails, that's also acceptable for edge cases
            pass

        # Single dimension
        shape = (5,)
        ones_tensor = torch.ones(shape)
        mask = Mask.create_mask_from_dense_mask(shape, ones_tensor, dtype=torch.float32)
        assert mask.is_full_mask()

        # Very small values that are not exactly 1.0
        shape = (2, 2)
        almost_ones = torch.tensor([[1.0, 0.999999], [1.0, 1.0]])
        mask = Mask.create_mask_from_dense_mask(shape, almost_ones, dtype=torch.float32)
        assert not mask.is_full_mask()  # Should not be detected as full

    def test_create_from_row_wise_idx_sparse_dense_equivalence(self):
        """Stress test to ensure sparse and dense modes produce identical results."""
        print("\n" + "=" * 80)
        print(
            "Stress Testing: create_from_row_wise_idx - Sparse vs Dense Mode Equivalence"
        )
        print("=" * 80)

        test_cases_passed: int = 0
        test_cases_total: int = 0

        # Test configuration matrix
        test_configs: list = [
            # (shape, k, description)
            ((10, 20), 5, "Small 2D"),
            ((50, 100), 20, "Medium 2D"),
            ((100, 200), 50, "Large 2D"),
            ((5, 8, 16), 4, "Small 3D"),
            ((10, 16, 32), 8, "Medium 3D"),
            ((4, 8, 16, 32), 6, "Small 4D"),
            ((2, 4, 8, 64), 16, "Medium 4D"),
        ]

        # Test different mask types
        mask_types: list = ["dense", "index"]

        # Test different dtypes
        dtypes: list = [torch.float32, torch.float64]

        for shape, k, desc in test_configs:
            n: int = shape[-1]
            batch_dims: tuple = shape[:-1]

            for mask_type in mask_types:
                for dtype in dtypes:
                    test_cases_total += 1

                    # Generate random row-wise indices
                    row_wise_idx: torch.Tensor = torch.randint(
                        0, n, size=batch_dims + (k,), dtype=torch.long
                    )

                    # Generate random data values
                    data: torch.Tensor = torch.rand(batch_dims + (k,), dtype=dtype)

                    # Create mask using sparse mode
                    mask_sparse: Mask = Mask.create_from_row_wise_idx(
                        shape,
                        row_wise_idx,
                        data,
                        mask_type=mask_type,
                        dtype=dtype,
                        mode="sparse",
                    )

                    # Create mask using dense mode
                    mask_dense: Mask = Mask.create_from_row_wise_idx(
                        shape,
                        row_wise_idx,
                        data,
                        mask_type=mask_type,
                        dtype=dtype,
                        mode="dense",
                    )

                    # Compare dense representations (most reliable comparison)
                    dense_mask_sparse: torch.Tensor = mask_sparse.get_dense_mask()
                    dense_mask_dense: torch.Tensor = mask_dense.get_dense_mask()

                    # Verify they are identical
                    if not torch.allclose(
                        dense_mask_sparse, dense_mask_dense, rtol=1e-5, atol=1e-7
                    ):
                        print(f"❌ FAILED: {desc}, mask_type={mask_type}, dtype={dtype}")
                        print(
                            f"   Max diff: {(dense_mask_sparse - dense_mask_dense).abs().max().item()}"
                        )
                        assert (
                            False
                        ), f"Sparse and dense modes produce different results for {desc}"

                    # Note: We don't compare index representations directly because:
                    # - Sparse mode creates index representation directly from row_wise_idx
                    # - Dense mode creates dense mask first (which handles duplicates via scatter_),
                    #   then converts to index (extracting non-zero elements)
                    # - With random indices, duplicates can occur, leading to different index counts
                    # - The important thing is that the dense representations match (which they do!)

                    test_cases_passed += 1

        print(f"\n✅ All {test_cases_passed}/{test_cases_total} test cases passed!")
        print(
            f"   Tested configurations: {len(test_configs)} shapes × {len(mask_types)} mask types × {len(dtypes)} dtypes"
        )
        print("=" * 80 + "\n")

    def test_create_from_row_wise_idx_sparse_dense_edge_cases(self):
        """Test edge cases for sparse vs dense mode equivalence."""
        # Set seed for reproducibility
        torch.manual_seed(42)

        print("\n" + "=" * 80)
        print("Edge Case Testing: create_from_row_wise_idx - Sparse vs Dense Mode")
        print("=" * 80)

        # Edge Case 1: Single element per row
        print("\nTest 1: Single element per row")
        shape: tuple = (10, 50)
        row_wise_idx: torch.Tensor = torch.randint(
            0, 50, size=(10, 1), dtype=torch.long
        )
        data: torch.Tensor = torch.rand(10, 1, dtype=torch.float32)

        mask_sparse: Mask = Mask.create_from_row_wise_idx(
            shape,
            row_wise_idx,
            data,
            mask_type="dense",
            dtype=torch.float32,
            mode="sparse",
        )
        mask_dense: Mask = Mask.create_from_row_wise_idx(
            shape,
            row_wise_idx,
            data,
            mask_type="dense",
            dtype=torch.float32,
            mode="dense",
        )

        assert torch.allclose(mask_sparse.get_dense_mask(), mask_dense.get_dense_mask())
        print("   ✅ Passed")

        # Edge Case 2: Many elements per row (high density)
        print("\nTest 2: High density (90% of elements)")
        shape: tuple = (5, 100)
        k: int = 90
        row_wise_idx: torch.Tensor = torch.stack(
            [torch.randperm(100)[:k] for _ in range(5)]
        )
        data: torch.Tensor = torch.rand(5, k, dtype=torch.float32)

        mask_sparse: Mask = Mask.create_from_row_wise_idx(
            shape,
            row_wise_idx,
            data,
            mask_type="index",
            dtype=torch.float32,
            mode="sparse",
        )
        mask_dense: Mask = Mask.create_from_row_wise_idx(
            shape,
            row_wise_idx,
            data,
            mask_type="index",
            dtype=torch.float32,
            mode="dense",
        )

        assert torch.allclose(mask_sparse.get_dense_mask(), mask_dense.get_dense_mask())
        print("   ✅ Passed")

        # Edge Case 3: Duplicate indices (last value should win)
        print("\nTest 3: Duplicate indices (scatter_ behavior)")
        shape: tuple = (2, 10)
        row_wise_idx: torch.Tensor = torch.tensor([[0, 0, 0], [5, 5, 5]])
        data: torch.Tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        mask_sparse: Mask = Mask.create_from_row_wise_idx(
            shape,
            row_wise_idx,
            data,
            mask_type="dense",
            dtype=torch.float32,
            mode="sparse",
        )
        mask_dense: Mask = Mask.create_from_row_wise_idx(
            shape,
            row_wise_idx,
            data,
            mask_type="dense",
            dtype=torch.float32,
            mode="dense",
        )

        # Both should have the last value at the duplicate index
        dense_sparse: torch.Tensor = mask_sparse.get_dense_mask()
        dense_dense: torch.Tensor = mask_dense.get_dense_mask()

        assert torch.allclose(dense_sparse, dense_dense)
        print("   ✅ Passed")

        # Edge Case 4: Very large batch dimensions
        print("\nTest 4: Large batch dimensions")
        shape: tuple = (32, 32, 32, 128)
        k: int = 64
        row_wise_idx: torch.Tensor = torch.randint(
            0, shape[-1], size=shape[:-1] + (k,), dtype=torch.long
        )
        data: torch.Tensor = torch.rand(shape[:-1] + (k,), dtype=torch.float32)

        mask_sparse: Mask = Mask.create_from_row_wise_idx(
            shape,
            row_wise_idx,
            data,
            mask_type="index",
            dtype=torch.float32,
            mode="sparse",
        )
        mask_dense: Mask = Mask.create_from_row_wise_idx(
            shape,
            row_wise_idx,
            data,
            mask_type="dense",
            dtype=torch.float32,
            mode="dense",
        )

        assert torch.allclose(mask_sparse.get_dense_mask(), mask_dense.get_dense_mask())
        print("   ✅ Passed")

        # Edge Case 5: Small values near zero
        print("\nTest 5: Small values near zero")
        shape: tuple = (5, 20)
        row_wise_idx: torch.Tensor = torch.randint(0, 20, size=(5, 3), dtype=torch.long)
        data: torch.Tensor = (
            torch.randn(5, 3, dtype=torch.float64) * 1e-8
        )  # Very small values

        mask_sparse: Mask = Mask.create_from_row_wise_idx(
            shape,
            row_wise_idx,
            data,
            mask_type="dense",
            dtype=torch.float64,
            mode="sparse",
        )
        mask_dense: Mask = Mask.create_from_row_wise_idx(
            shape,
            row_wise_idx,
            data,
            mask_type="dense",
            dtype=torch.float64,
            mode="dense",
        )

        assert torch.allclose(
            mask_sparse.get_dense_mask(),
            mask_dense.get_dense_mask(),
            rtol=1e-10,
            atol=1e-15,
        )
        print("   ✅ Passed")

        print("\n✅ All edge case tests passed!")
        print("=" * 80 + "\n")

    def test_create_from_row_wise_idx_sparse_index_duplicates_keep_last_value(self):
        """Sparse index mode should match dense scatter_ behavior for duplicates."""
        shape = (2, 5)
        row_wise_idx = torch.tensor([[1, 3, 1, 1], [2, 2, 4, 2]])
        data = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

        mask = Mask.create_from_row_wise_idx(
            shape,
            row_wise_idx,
            data,
            mask_type="index",
            dtype=torch.float32,
            mode="sparse",
        )

        expected = torch.zeros(shape, dtype=torch.float32)
        expected.scatter_(dim=-1, index=row_wise_idx, src=data)

        indices, ptr, index_data = mask.get_index_mask()

        torch.testing.assert_close(indices, torch.tensor([3, 1, 9, 7]))
        torch.testing.assert_close(ptr, torch.tensor([0, 2, 4]))
        torch.testing.assert_close(mask.get_dense_mask(), expected)
        torch.testing.assert_close(index_data, torch.tensor([2.0, 4.0, 7.0, 8.0]))
