"""Mask class for representing attention masks in both dense and sparse formats."""

import warnings
from typing import Literal, Optional, Tuple

import numpy as np
import torch


class Mask:
    """
    Mask object represents a mask over a tensor of shape (..., n).

    There are three representations of a Mask Object:
    1. mask: mask.shape == shape (dense representation)
    2. sparse matrix format: stores indices and ptr of mask (sparse representation)
    3. full mask: all elements are 1.0 (optimized representation)

    Both representations store floating point values. A value of zero means that the
    element is not active. A non-zero value represents the weight assigned to this
    element in tensor.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        mask: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        ptr: Optional[torch.Tensor] = None,
        data: Optional[torch.Tensor] = None,
        from_dense_mask: bool = False,
        from_index: bool = False,
        is_full: bool = False,
        is_empty: bool = False,
    ):
        """
        Initialize a Mask object.

        Args:
            shape: Shape of the mask (*, n)
            dtype: Data type for the mask values
            mask: Dense mask tensor (if from_dense_mask=True)
            indices: Indices for sparse representation (if from_index=True)
            ptr: Pointer array for sparse representation (if from_index=True)
            data: Data for sparse representation (if from_index=True)
            from_dense_mask: Whether creating from dense mask
            from_index: Whether creating from sparse indices
            is_full: Whether this is a full mask (all elements are 1.0)
            is_empty: Whether this is an empty mask (all elements are 0.0)
            device: Device for tensors (required for is_full=True or is_empty=True, inferred otherwise)
        """
        self.shape = shape
        self.dtype = dtype
        self.from_dense_mask = from_dense_mask
        self.from_index = from_index
        self.is_full = is_full
        self.is_empty = is_empty
        self.device: torch.device = device
        self.mask: Optional[torch.Tensor] = None
        self.indices: Optional[torch.Tensor] = None
        self.ptr: Optional[torch.Tensor] = None
        self.data: Optional[torch.Tensor] = None

        if is_full:
            # Full mask optimization - don't store any actual data
            # Device must be provided for full masks since we have no tensors to infer from
            if device is None:
                raise ValueError("device must be specified for full masks")
            self.device = device
        elif is_empty:
            # Empty mask optimization - don't store any actual data
            # Device must be provided for empty masks since we have no tensors to infer from
            if device is None:
                raise ValueError("device must be specified for empty masks")
            self.device = device
        elif from_dense_mask and mask is not None:
            self.mask = mask.to(dtype)
            self.device = mask.device
            # Check if this is actually a full mask
            if self._detect_full_mask():
                self.is_full = True
                self.mask = None  # Free memory
            # Check if this is actually an empty mask
            elif self._detect_empty_mask():
                self.is_empty = True
                self.mask = None  # Free memory
        elif from_index and indices is not None and ptr is not None:
            self.mask = None
            self.device = indices.device
            self.indices = indices
            self.ptr = ptr
            self.data = data
            # Check if this is actually a full mask
            if self._detect_full_mask():
                self.is_full = True
                self.indices = None
                self.ptr = None
                self.data = None
            # Check if this is actually an empty mask
            elif self._detect_empty_mask():
                self.is_empty = True
                self.indices = None
                self.ptr = None
                self.data = None
        else:
            raise ValueError(
                "Must specify either from_dense_mask with mask, from_index with indices and ptr, is_full=True, or is_empty=True"
            )

    def _detect_full_mask(self, only_check_flag: bool = True) -> bool:
        """
        Detect if the current mask represents a full mask (all elements are 1.0).
        include only_check_flag = False only if you need it . not for fast path. default is True.
        Returns:
            True if all elements are 1.0, False otherwise
        """
        if self.is_full:
            return True

        # A mask cannot be both full and empty
        if self.is_empty:
            return False

        if only_check_flag:
            return False

        # use only_check_flag = False only if you need it . not for fast path

        if self.from_dense_mask and self.mask is not None:
            return bool(torch.all(self.mask == 1.0).item())
        elif self.from_index and self.indices is not None and self.ptr is not None:
            # Check if sparse representation has all positions filled with 1.0
            expected_size = int(np.prod(self.shape))
            if self.indices.numel() != expected_size:
                return False
            if self.data is not None:
                return bool(torch.all(self.data == 1.0).item())
            else:
                # If no data is provided, assume 1.0 values
                return True

        return False

    def _detect_empty_mask(self, only_check_flag: bool = True) -> bool:
        """
        Detect if the current mask represents an empty mask (all elements are 0.0).
        include only_check_flag = False only if you need it . not for fast path. default is True.
        Returns:
            True if all elements are 0.0, False otherwise
        """
        if self.is_empty:
            return True

        # A mask cannot be both full and empty
        if self.is_full:
            return False

        if only_check_flag:
            return False

        # use only_check_flag = False only if you need it . not for fast path

        if self.from_dense_mask and self.mask is not None:
            return bool(torch.all(self.mask == 0.0).item())
        elif self.from_index and self.indices is not None:
            # For sparse representation, empty means no indices
            return int(self.indices.numel()) == 0

        return False

    # CREATE METHODS
    @classmethod
    def create_empty_mask(
        cls,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> "Mask":
        """
        Create a mask with is_empty set to True

        Args:
            shape: Shape of the mask (*, n)
            dtype: Data type for the mask values
            device: Device for the mask tensors

        Returns:
            Mask object with is_empty=True (optimized representation)
        """
        # Use optimized empty mask representation regardless of mask_type
        return cls(shape=shape, dtype=dtype, is_empty=True, device=device)

    @classmethod
    def create_full_mask(
        cls,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> "Mask":
        """
        Create a full mask (all elements are 1.0) with optimized representation.

        Args:
            shape: Shape of the mask (*, n)
            dtype: Data type for the mask values
            device: Device for the mask tensors

        Returns:
            Mask object with is_full=True
        """
        return cls(shape=shape, dtype=dtype, is_full=True, device=device)

    def is_full_mask(self) -> bool:
        """
        Check if this is a full mask (all elements are 1.0).

        Returns:
            True if all elements are 1.0, False otherwise
        """
        return self.is_full

    def is_empty_mask(self) -> bool:
        """
        Check if this is an empty mask (all elements are 0.0).

        Returns:
            True if all elements are 0.0, False otherwise
        """
        return self.is_empty

    @classmethod
    def create_mask_from_indices(
        cls,
        shape: Tuple[int, ...],
        indices: torch.Tensor,
        ptr: torch.Tensor,
        data: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> "Mask":
        """
        Create a Mask from sparse indices and pointer representation.

        Args:
            shape: Shape of the mask (*, n)
            indices: Indices for sparse representation
            ptr: Pointer array for sparse representation
            data: Data for sparse representation | if not provided (None), assume 1.0 values
            dtype: Data type for the mask values

        Returns:
            Mask object with from_index=True
        """
        # For multi-dimensional tensors, ptr should have size equal to the product of all dimensions except the last one, plus 1
        expected_ptr_size: int = int(np.prod(shape[:-1]) + 1)
        if ptr.numel() != expected_ptr_size:
            raise ValueError(
                f"ptr.numel() must be {expected_ptr_size}, got {ptr.numel()}"
            )

        return cls(
            shape=shape,
            dtype=dtype,
            indices=indices,
            ptr=ptr,
            data=data,
            from_index=True,
            device=indices.device,
        )

    @classmethod
    def create_mask_from_dense_mask(
        cls,
        shape: Tuple[int, ...],
        mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> "Mask":
        """
        Create a Mask from dense mask tensor.

        Args:
            shape: Shape of the mask (*, n)
            mask: Dense mask tensor of shape (*, n)
            dtype: Data type for the mask values

        Returns:
            Mask object with from_dense_mask=True
        """
        if mask.shape != shape:
            raise ValueError(f"mask.shape must be {shape}, got {mask.shape}")

        return cls(
            shape=shape,
            dtype=dtype,
            mask=mask,
            from_dense_mask=True,
            device=mask.device,
        )

    @classmethod
    def create_from_row_wise_idx(
        cls,
        shape: Tuple[int, ...],
        row_wise_idx: torch.Tensor,
        data: torch.Tensor,
        mask_type: Literal["index", "dense"],
        dtype: torch.dtype,
        use_padding: bool = False,
        mode: Literal["sparse", "dense"] = "dense",
    ) -> "Mask":
        if mode == "sparse":
            if mask_type == "dense":
                warnings.warn(
                    "Creating dense mask in sparse mode is SLOW. Unless you know what you are doing, use dense mode instead.",
                    UserWarning,
                    stacklevel=2,
                )
            return cls.create_from_row_wise_idx_sparse(
                shape, row_wise_idx, data, mask_type, dtype, use_padding
            )
        elif mode == "dense":
            if mask_type == "index":
                warnings.warn(
                    "Creating index mask in dense mode is SLOW. Unless you know what you are doing, use sparse mode instead.",
                    UserWarning,
                    stacklevel=2,
                )
            return cls.create_from_row_wise_idx_dense(
                shape, row_wise_idx, data, mask_type, dtype, use_padding
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

    @classmethod
    def create_from_row_wise_idx_dense(
        cls,
        shape: Tuple[int, ...],
        row_wise_idx: torch.Tensor,
        data: torch.Tensor,
        mask_type: Literal["index", "dense"],
        dtype: torch.dtype,
        use_padding: bool = False,
    ) -> "Mask":
        """
        Create a Mask from row-wise indices in dense representation.

        Args:
            shape: Shape of the mask (*, n)
            row_wise_idx: Row-wise indices tensor of shape (*, k). If use_padding=True,
                         -1 values are treated as padding and filtered out.
            data: Data tensor of shape (*, k) corresponding to the indices
            mask_type: Type of representation ("index" or "dense"). The mask that you need to return.

        Returns:
            Mask object with the specified representation

        Comments: For this kind of creation, it is easier to create the mask in dense representation.
        So we manipulate it that way. If required mask_type is index , we convert it to index representation.
        """
        if len(shape) < 1:
            raise ValueError("shape must have at least one dimension")

        # if empty row-wise idx, return empty mask
        if row_wise_idx.shape[-1] <= 0:
            raise ValueError("row_wise_idx must have at least one column")

        # Validate input shapes
        expected_row_wise_shape: Tuple[int, ...] = shape[:-1] + (
            row_wise_idx.shape[-1],
        )
        if row_wise_idx.shape != expected_row_wise_shape:
            raise ValueError(
                f"row_wise_idx.shape must be {expected_row_wise_shape}, got {row_wise_idx.shape}"
            )

        if data.shape != row_wise_idx.shape:
            raise ValueError(
                f"data.shape must match row_wise_idx.shape {row_wise_idx.shape}, got {data.shape}"
            )
        assert (
            not use_padding
        ), "use_padding is not supported for dense mode; use sparse mode instead"

        dense_mask: torch.Tensor = torch.zeros(
            shape, dtype=dtype, device=row_wise_idx.device
        )
        dense_mask.scatter_(dim=-1, index=row_wise_idx, src=data)
        if mask_type == "dense":
            return cls.create_mask_from_dense_mask(shape, dense_mask, dtype=dtype)
        elif mask_type == "index":
            # Convert dense to index representation
            temp_mask: Mask = cls.create_mask_from_dense_mask(
                shape, dense_mask, dtype=dtype
            )
            return temp_mask
        else:
            raise ValueError(f"type must be 'index' or 'dense', got {mask_type}")

    @classmethod
    def create_from_row_wise_idx_sparse(
        cls,
        shape: Tuple[int, ...],
        row_wise_idx: torch.Tensor,
        data: torch.Tensor,
        mask_type: Literal["index", "dense"],
        dtype: torch.dtype,
        use_padding: bool = False,
    ) -> "Mask":
        """
        Create a Mask from row-wise indices.

        Args:
            shape: Shape of the mask (*, n)
            row_wise_idx: Row-wise indices tensor of shape (*, k). If use_padding=True,
                         -1 values are treated as padding and filtered out.
            data: Data tensor of shape (*, k) corresponding to the indices
            mask_type: Type of representation ("index" or "dense"). The mask that you need to return.
            dtype: Data type for the mask values
            use_padding: If True, handles -1 padding (causes GPU-CPU sync). If False (default),
                        assumes all indices are valid and uses zero-sync fast path.

        Returns:
            Mask object with the specified representation


        Comments: For this kind of creation, it is easier to create the mask in index representation.
        So we manipulate it that way. If required mask_type is dense , we convert it to dense representation.
        """
        if len(shape) < 1:
            raise ValueError("shape must have at least one dimension")

        # if empty row-wise idx, return empty mask
        if row_wise_idx.shape[-1] <= 0:
            raise ValueError("row_wise_idx must have at least one column")

        # Validate input shapes
        expected_row_wise_shape: Tuple[int, ...] = shape[:-1] + (
            row_wise_idx.shape[-1],
        )
        if row_wise_idx.shape != expected_row_wise_shape:
            raise ValueError(
                f"row_wise_idx.shape must be {expected_row_wise_shape}, got {row_wise_idx.shape}"
            )

        if data.shape != row_wise_idx.shape:
            raise ValueError(
                f"data.shape must match row_wise_idx.shape {row_wise_idx.shape}, got {data.shape}"
            )

        n: int = shape[-1]
        batch_dims: Tuple[int, ...] = shape[:-1]
        batch_size: int = int(np.prod(batch_dims))
        k: int = row_wise_idx.shape[-1]

        flat_row_wise_idx: torch.Tensor = row_wise_idx.reshape(batch_size, k)
        flat_data: torch.Tensor = data.reshape(batch_size, k)

        row_offsets: torch.Tensor = (
            torch.arange(batch_size, device=flat_row_wise_idx.device).unsqueeze(1) * n
        )

        if use_padding:
            # Slow path: Handle -1 padding (causes GPU-CPU sync via boolean indexing)
            flat_indices_with_offset: torch.Tensor = flat_row_wise_idx + row_offsets

            # Filter out invalid indices (e.g., -1 padding)
            valid_mask: torch.Tensor = flat_row_wise_idx >= 0
            valid_flat_indices: torch.Tensor = flat_indices_with_offset[valid_mask]
            valid_values: torch.Tensor = flat_data[valid_mask]

            valid_counts_per_row: torch.Tensor = valid_mask.sum(dim=1)
            ptr: torch.Tensor = torch.cat(
                [
                    torch.zeros(1, dtype=torch.long, device=flat_row_wise_idx.device),
                    torch.cumsum(valid_counts_per_row, dim=0),
                ]
            )
        else:
            # Fast path: No padding handling.
            # Deduplicate indices within each row (last occurrence wins, matching scatter_
            # semantics used by create_from_row_wise_idx_dense) and build ptr from the
            # actual per-row unique counts to avoid variable-shadowing bugs.
            unique_indices: list = []
            unique_values: list = []
            row_unique_counts: list = []

            for i in range(batch_size):
                row_idx = flat_row_wise_idx[i]
                row_val = flat_data[i]

                seen: dict = {}
                for j in range(len(row_idx)):
                    seen[int(row_idx[j])] = row_val[j].item()  # last value wins

                for col_idx, v in seen.items():  # col_idx avoids shadowing outer k
                    unique_indices.append(i * n + col_idx)
                    unique_values.append(v)
                row_unique_counts.append(len(seen))

            valid_flat_indices = torch.tensor(
                unique_indices, dtype=torch.long, device=flat_row_wise_idx.device
            )
            valid_values = torch.tensor(
                unique_values, dtype=dtype, device=flat_row_wise_idx.device
            )

            # Build ptr from actual unique counts per row (handles duplicates correctly)
            counts_tensor = torch.tensor(
                row_unique_counts, dtype=torch.long, device=flat_row_wise_idx.device
            )
            ptr = torch.cat(
                [
                    torch.zeros(1, dtype=torch.long, device=flat_row_wise_idx.device),
                    torch.cumsum(counts_tensor, dim=0),
                ]
            )

        # Create sparse mask
        sparse_mask: Mask = cls.create_mask_from_indices(
            shape, valid_flat_indices, ptr, valid_values, dtype=dtype
        )

        if mask_type == "dense":
            # Convert to dense representation
            dense_mask: torch.Tensor = sparse_mask.get_dense_mask()
            return cls.create_mask_from_dense_mask(shape, dense_mask, dtype=dtype)
        elif mask_type == "index":
            return sparse_mask
        else:
            raise ValueError(f"type must be 'index' or 'dense', got {type}")

    def get_index_mask(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the sparse index representation.

        Returns:
            Tuple of (indices, ptr, data) for sparse representation
        Comments: This will return the mask in index representation.
        """
        if self.is_full:
            # For full masks, generate all indices with value 1.0
            total_size: int = int(np.prod(self.shape))
            indices: torch.Tensor = torch.arange(
                total_size, dtype=torch.long, device=self.device
            )
            data = torch.ones(total_size, dtype=self.dtype, device=self.device)

            # Create ptr array
            n_cols: int = self.shape[-1]
            ptr = torch.arange(
                0, total_size + 1, n_cols, dtype=torch.long, device=self.device
            )

            return indices, ptr, data
        elif self.is_empty:
            # For empty masks, return empty indices with zero data
            empty_indices: torch.Tensor = torch.empty(
                0, dtype=torch.long, device=self.device
            )
            ptr_size: int = int(np.prod(self.shape[:-1]) + 1)
            ptr = torch.zeros(ptr_size, dtype=torch.long, device=self.device)
            data = torch.empty(0, dtype=self.dtype, device=self.device)
            return empty_indices, ptr, data
        elif self.from_index:
            if self.indices is None or self.ptr is None:
                raise RuntimeError("Sparse mask indices or ptr is None")
            if self.data is None:
                # If no data is provided, assume 1.0 values
                data = torch.ones(
                    self.indices.numel(), dtype=self.dtype, device=self.indices.device
                )
            else:
                data = self.data
            return self.indices, self.ptr, data
        elif self.from_dense_mask:
            if self.mask is None:
                raise RuntimeError("Dense mask is None")
            # Convert dense representation to sparse representation
            # Find non-zero indices
            non_zero_indices: torch.Tensor = torch.nonzero(
                self.mask.view(-1), as_tuple=False
            ).squeeze(-1)
            data = self.mask.view(-1)[non_zero_indices]

            # Create ptr array with size equal to product of all dimensions except the last one, plus 1
            # Vectorized version: count nonzero per row and cumsum
            counts = torch.count_nonzero(self.mask.view(-1, self.shape[-1]), dim=1)
            ptr = torch.cat(
                [
                    torch.zeros(1, dtype=torch.long, device=self.mask.device),
                    torch.cumsum(counts, dim=0),
                ]
            )

            return non_zero_indices, ptr, data
        else:
            raise RuntimeError("Mask object is in an invalid state")

    def get_dense_mask(self) -> torch.Tensor:
        """
        Get the dense mask representation.

        Returns:
            Dense mask tensor
        Comments: This will return the mask object in dense representation.
        """
        if self.is_full:
            # For full masks, return tensor of ones
            return torch.ones(self.shape, dtype=self.dtype, device=self.device)
        elif self.is_empty:
            # For empty masks, return tensor of zeros
            return torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        elif self.from_dense_mask:
            if self.mask is None:
                raise RuntimeError("Dense mask is None")
            return self.mask
        elif self.from_index:
            if self.indices is None or self.ptr is None:
                raise RuntimeError("Sparse mask indices or ptr is None")
            # Convert sparse representation to dense mask
            # Ensure dense mask is on the same device as indices/data
            device: torch.device = self.indices.device
            dense_mask: torch.Tensor = torch.zeros(
                self.shape, dtype=self.dtype, device=device
            )
            if len(self.indices) > 0:
                if self.data is not None:
                    dense_mask.view(-1)[self.indices] = self.data
                else:
                    dense_mask.view(-1)[self.indices] = 1.0
            return dense_mask
        else:
            raise RuntimeError("Mask object is in an invalid state")

    # APPLY MASK
    def apply_mask(
        self, input_tensor: torch.Tensor, mode: Literal["sparse", "dense"] = "dense"
    ) -> torch.Tensor:
        """
        Apply the mask to an input tensor.

        Args:
            input_tensor: Input tensor with shape same as the mask
            mode: Mode of application ("sparse" or "dense"), defaults to "sparse"

        Returns:
            Output tensor after applying the mask.
            if full mask, return input tensor directly.
            if empty mask, return input tensor directly.
            otherwise, return the masked output tensor.

        """
        if mode == "sparse":
            return self.apply_mask_sparse(input_tensor)
        elif mode == "dense":
            return self.apply_mask_dense(input_tensor)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def apply_mask_dense(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the mask to an input tensor in dense format.

        Args:
            input_tensor: Input tensor with shape same as the mask

        Returns:
            Output tensor after applying the mask
        """
        if input_tensor.shape != self.shape:
            raise ValueError(
                f"input_tensor.shape must be {self.shape}, got {input_tensor.shape}"
            )

        if self.is_full or self.is_empty:
            # Full mask optimization - return input tensor directly (no-op)
            return input_tensor

        dense_mask = self.get_dense_mask()
        output = input_tensor * dense_mask
        return output

    def apply_mask_sparse(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the mask to an input tensor.

        Args:
            if mask is empty or full, return input_tensor # as if no mask is applied
            if mask is not empty, then 0=> inactive and >0 => active with the corresponding weight

            input_tensor: Input tensor with shape same as the mask

        Returns:
            Output tensor after applying the mask
        """
        if input_tensor.shape != self.shape:
            raise ValueError(
                f"input_tensor.shape must be {self.shape}, got {input_tensor.shape}"
            )

        if self.is_full or self.is_empty:
            # Full mask optimization - return input tensor directly (no-op)
            return input_tensor

        # Always work with index representation for efficiency
        indices: torch.Tensor
        ptr: torch.Tensor
        data: torch.Tensor
        indices, ptr, data = self.get_index_mask()

        # Apply mask using sparse operations
        output: torch.Tensor = torch.zeros_like(input_tensor)
        if indices.numel() > 0:
            output.view(-1)[indices] = input_tensor.view(-1)[indices] * data

        return output

    # APPLY INVERSE MASK

    def apply_inv_mask(
        self, input_tensor: torch.Tensor, mode: Literal["sparse", "dense"] = "dense"
    ) -> torch.Tensor:
        """
        Apply the inverse mask to an input tensor.

        Args:
            input_tensor: Input tensor with shape same as the mask
            mode: Mode of application ("sparse" or "dense"), defaults to "sparse"

        Returns:
            Output tensor after applying the inverse mask
        """
        if mode == "sparse":
            return self.apply_inv_mask_sparse(input_tensor)
        elif mode == "dense":
            return self.apply_inv_mask_dense(input_tensor)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def apply_inv_mask_dense(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the inverse mask to an input tensor in dense format.

        Args:
            input_tensor: Input tensor with shape same as the mask

        Returns:
            Output tensor after applying the inverse mask
        """
        if input_tensor.shape != self.shape:
            raise ValueError(
                f"input_tensor.shape must be {self.shape}, got {input_tensor.shape}"
            )

        if self.is_full or self.is_empty:
            # Full mask optimization - return input tensor directly (1.0 / 1.0 = 1.0)
            return input_tensor

        mask_dense = self.get_dense_mask()
        valid_mask = mask_dense != 0
        result = input_tensor * (1.0 / (mask_dense + 1e-6) * valid_mask)
        return result

    def apply_inv_mask_sparse(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the inverse mask to an input tensor.

        Args:
            input_tensor: Input tensor with shape same as the mask

        Returns:
            if mask is empty or full, return input_tensor # as if no mask is applied

            Output tensor after applying the inverse mask:
            - output[IDX] = 0 if mask[IDX] == 0
            - output[IDX] = input[IDX] * 1.0 / mask[IDX] if mask[IDX] != 0
        """
        if input_tensor.shape != self.shape:
            raise ValueError(
                f"input_tensor.shape must be {self.shape}, got {input_tensor.shape}"
            )

        if self.is_full or self.is_empty:
            # Full mask optimization - return input tensor directly (1.0 / 1.0 = 1.0)
            return input_tensor

        # Always work with index representation for efficiency
        indices: torch.Tensor
        ptr: torch.Tensor
        data: torch.Tensor
        indices, ptr, data = self.get_index_mask()

        # Apply inverse mask using sparse operations
        output: torch.Tensor = torch.zeros_like(input_tensor)
        if indices.numel() > 0:
            # For inverse mask: output[indices] = input[indices] * (1.0 / data)
            output.view(-1)[indices] = input_tensor.view(-1)[indices] * (
                1.0 / (data + 1e-6)
            )

        return output

    def __repr__(self) -> str:
        """String representation of the Mask object."""
        return f"Mask(shape={self.shape}, dtype={self.dtype}, from_dense_mask={self.from_dense_mask}, from_index={self.from_index}, is_full={self.is_full}, is_empty={self.is_empty})"

    def merge_mask(
        self,
        other_mask: "Mask",
        inplace: bool,
        mode: Literal["sparse", "dense"] = "dense",
    ) -> "Mask":
        """
        Merge this mask with another mask.

        Args:
            other_mask: The other mask to merge with
            inplace: Whether to modify this mask in place
            mode: Merge mode - "dense" (default) or "sparse"

        Returns:
            Merged mask
        """
        if mode == "dense":
            return self.merge_mask_dense(other_mask, inplace)
        elif mode == "sparse":
            return self.merge_mask_sparse(other_mask, inplace)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def merge_mask_dense(
        self,
        other_mask: "Mask",
        inplace: bool,
    ) -> "Mask":
        """
        Merge this mask with another mask in dense format.
        """
        if self.shape != other_mask.shape:
            raise ValueError(
                f"Cannot merge masks with different shapes: {self.shape} vs {other_mask.shape}"
            )

        # Full mask optimizations
        if self.is_full or other_mask.is_full:
            # If either mask is full, the result is full
            if inplace:
                self.is_full = True
                self.is_empty = False
                self.from_dense_mask = False
                self.from_index = False
                self.mask = None
                self.indices = None
                self.ptr = None
                self.data = None
                return self
            else:
                return Mask.create_full_mask(self.shape, self.dtype, self.device)

        if self.is_empty and other_mask.is_empty:
            # If both masks are empty, result is empty
            if inplace:
                return self
            else:
                return Mask.create_empty_mask(self.shape, self.dtype, self.device)

        # Empty mask optimizations
        if self.is_empty:
            # If this mask is empty, result is the other mask
            if inplace:
                # Copy other mask's state to self
                self.is_full = other_mask.is_full
                self.is_empty = other_mask.is_empty
                self.from_dense_mask = other_mask.from_dense_mask
                self.from_index = other_mask.from_index
                self.mask = other_mask.mask
                self.indices = other_mask.indices
                self.ptr = other_mask.ptr
                self.data = other_mask.data
                self.device = other_mask.device
                return self
            else:
                # Return a copy of the other mask (can use get_dense_mask and create new)
                return Mask.create_mask_from_dense_mask(
                    self.shape, other_mask.get_dense_mask(), dtype=self.dtype
                )

        if other_mask.is_empty:
            # If other mask is empty, result is this mask
            if inplace:
                return self
            else:
                return Mask.create_mask_from_dense_mask(
                    self.shape, self.get_dense_mask(), dtype=self.dtype
                )

        mask1 = self.get_dense_mask()
        mask2 = other_mask.get_dense_mask()
        merged_mask = torch.clamp(mask1 + mask2, 0.0, 1.0)
        if inplace:
            self.mask = merged_mask
            self.from_dense_mask = True
            self.from_index = False
            self.is_full = False
            self.is_empty = False
            self.indices = None
            self.ptr = None
            self.data = None
            return self
        else:
            return Mask.create_mask_from_dense_mask(
                self.shape, merged_mask, dtype=self.dtype
            )

    def merge_mask_sparse(
        self,
        other_mask: "Mask",
        inplace: bool,
    ) -> "Mask":
        """
        Merge this mask with another mask in sparse format.

        Args:
            other_mask: The other mask to merge with
            inplace: Whether to update the current mask in place

        Returns:
            Merged mask (new instance if inplace=False, self if inplace=True)
        """

        if self.shape != other_mask.shape:
            raise ValueError(
                f"Cannot merge masks with different shapes: {self.shape} vs {other_mask.shape}"
            )

        # Full mask optimizations
        if self.is_full or other_mask.is_full:
            # If either mask is full, the result is full
            if inplace:
                self.is_full = True
                self.is_empty = False
                self.from_dense_mask = False
                self.from_index = False
                self.mask = None
                self.indices = None
                self.ptr = None
                self.data = None
                return self
            else:
                return Mask.create_full_mask(self.shape, self.dtype, self.device)

        if self.is_empty and other_mask.is_empty:
            # If both masks are empty, result is empty
            if inplace:
                return self
            else:
                return Mask.create_empty_mask(self.shape, self.dtype, self.device)

        # Empty mask optimizations
        if self.is_empty:
            # If this mask is empty, result is the other mask
            if inplace:
                # Copy other mask's state to self
                self.is_full = other_mask.is_full
                self.is_empty = other_mask.is_empty
                self.from_dense_mask = other_mask.from_dense_mask
                self.from_index = other_mask.from_index
                self.mask = other_mask.mask
                self.indices = other_mask.indices
                self.ptr = other_mask.ptr
                self.data = other_mask.data
                self.device = other_mask.device
                return self
            else:
                # Return a copy by getting index representation and creating new
                indices, ptr, data = other_mask.get_index_mask()
                return Mask.create_mask_from_indices(
                    self.shape, indices, ptr, data, dtype=self.dtype
                )

        if other_mask.is_empty:
            # If other mask is empty, result is this mask
            if inplace:
                return self
            else:
                indices, ptr, data = self.get_index_mask()
                return Mask.create_mask_from_indices(
                    self.shape, indices, ptr, data, dtype=self.dtype
                )

        self_indices: torch.Tensor
        self_ptr: torch.Tensor
        self_data: torch.Tensor
        self_indices, self_ptr, self_data = self.get_index_mask()
        other_indices: torch.Tensor
        other_ptr: torch.Tensor
        other_data: torch.Tensor
        other_indices, other_ptr, other_data = other_mask.get_index_mask()

        device: torch.device = (
            self_indices.device
            if self_indices.numel() > 0
            else (
                other_indices.device
                if other_indices.numel() > 0
                else torch.device("cpu")
            )
        )

        if self_indices.numel() == 0 and other_indices.numel() == 0:
            final_indices: torch.Tensor = torch.empty(
                0, dtype=torch.long, device=device
            )
            final_data: torch.Tensor = torch.empty(0, dtype=self.dtype, device=device)
            num_rows = int(np.prod(self.shape[:-1]))
            final_ptr = torch.zeros(num_rows + 1, dtype=torch.long, device=device)
        elif self_indices.numel() == 0:
            final_indices = other_indices
            final_data = torch.clamp(other_data, 0.0, 1.0)
            final_ptr = other_ptr
        elif other_indices.numel() == 0:
            final_indices = self_indices
            final_data = torch.clamp(self_data, 0.0, 1.0)
            final_ptr = self_ptr
        else:
            all_indices: torch.Tensor = torch.cat([self_indices, other_indices])
            all_data: torch.Tensor = torch.cat([self_data, other_data])

            sorted_indices: torch.Tensor
            sort_order: torch.Tensor
            sorted_indices, sort_order = torch.sort(all_indices)
            sorted_data: torch.Tensor = all_data[sort_order]

            unique_indices: torch.Tensor
            inverse_indices: torch.Tensor
            unique_indices, inverse_indices = torch.unique_consecutive(
                sorted_indices, return_inverse=True
            )

            summed_data: torch.Tensor = torch.zeros(
                unique_indices.numel(), dtype=self.dtype, device=device
            )
            summed_data.scatter_add_(0, inverse_indices, sorted_data)

            final_data = torch.clamp(summed_data, 0.0, 1.0)
            final_indices = unique_indices

            # Reconstruct ptr array by counting elements per row
            num_rows = int(np.prod(self.shape[:-1]))
            n_cols = self.shape[-1]
            row_indices: torch.Tensor = final_indices // n_cols

            # Use scatter_add instead of bincount to avoid GPU-CPU synchronization
            row_counts: torch.Tensor = torch.zeros(
                num_rows, dtype=torch.long, device=device
            )
            ones = torch.ones_like(row_indices, dtype=torch.long)
            row_counts.scatter_add_(0, row_indices, ones)

            final_ptr = torch.cat(
                [
                    torch.zeros(1, dtype=torch.long, device=device),
                    torch.cumsum(row_counts, dim=0),
                ]
            )

        if inplace:
            self.indices = final_indices
            self.ptr = final_ptr
            self.data = final_data
            self.from_index = True
            self.from_dense_mask = False
            self.is_full = False
            self.is_empty = False
            self.mask = None
            return self
        else:
            return Mask.create_mask_from_indices(
                shape=self.shape,
                indices=final_indices,
                ptr=final_ptr,
                data=final_data,
                dtype=self.dtype,
            )

    def get_density(self) -> float:
        """
        Get the sparsity of the mask.
        """
        if self.is_full:
            return 1.0
        elif self.is_empty:
            return 0.0
        elif self.from_dense_mask:
            return float(torch.sum(self.mask > 0) / self.mask.numel())
        elif self.from_index:
            return float(len(self.indices)) / float(np.prod(self.shape))
        else:
            raise RuntimeError("Mask object is in an invalid state")
