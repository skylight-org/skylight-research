"""Encoded tensor records and individual stored chunks."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch

from ..context import AttentionContext


@dataclass
class QuantizedTensor:
    codes: torch.Tensor
    scales: torch.Tensor
    zero_points: Optional[torch.Tensor]
    shape: Tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    axis: int
    bits: int
    format_id: str
    group_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    norms: Optional[torch.Tensor] = None
    global_scale: Optional[torch.Tensor] = None
    codebook: Optional[torch.Tensor] = None
    outlier_indices: Optional[torch.Tensor] = None
    outlier_values: Optional[torch.Tensor] = None
    group_bits: Optional[torch.Tensor] = None  # bits per group, for mixed precision
    qjl_codes: Optional[torch.Tensor] = None  # Packed one-bit residual signs.
    residual_norms: Optional[torch.Tensor] = None  # Unit-vector residual magnitudes.

    def storage_bytes(self) -> int:
        """Tensor storage only; excludes Python objects and shared calibration data."""
        tensors = [
            self.codes,
            self.scales,
            self.zero_points,
            self.norms,
            self.global_scale,
            self.codebook,
            self.outlier_indices,
            self.outlier_values,
            self.group_bits,
            self.qjl_codes,
            self.residual_norms,
        ]
        return sum(t.numel() * t.element_size() for t in tensors if t is not None)


@dataclass
class StoredChunk:
    """A run of consecutive tokens in one region (``sink``, ``middle`` or ``local``).

    Exactly one representation is set: ``full_precision`` (kept as received),
    ``quantized`` (codes, PACKED mode) or ``dequantized`` (reconstruction, FAKE mode).
    """

    start_token: int
    token_count: int
    context: Optional[AttentionContext] = None
    transform_state: Any = None
    quantized: Optional[QuantizedTensor] = None
    dequantized: Optional[torch.Tensor] = None
    region: str = "middle"
    full_precision: Optional[torch.Tensor] = None
    # Packed chunks only: tokens a plan keeps exact, with flat indices into
    # [batch * heads * tokens] and values [kept, head_dim]. FAKE chunks already
    # contain their originals in dequantized and release these extra tensors.
    kept_index: Optional[torch.Tensor] = None
    kept_values: Optional[torch.Tensor] = None

    @property
    def is_encoded(self) -> bool:
        return self.full_precision is None
