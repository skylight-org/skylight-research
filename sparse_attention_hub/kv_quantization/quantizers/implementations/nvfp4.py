"""The NVFP4 format settings, shared by FakeNVFP4Quantizer."""

from dataclasses import dataclass

from ..base import KVQuantizerConfig


@dataclass(frozen=True)
class NVFP4QuantizerConfig(KVQuantizerConfig):
    """NVFP4: FP4 (E2M1) values in blocks of 16, one FP8 (E4M3) scale per block and
    one FP32 scale per tensor. The format fixes ``bits``, ``group_size`` and the scale
    types; ``scale_dtype`` and ``offset_dtype`` are unused, and ``symmetric`` must be
    True. ``granularity`` chooses the block direction: along channels (PER_TOKEN or
    PER_BLOCK, the standard layout) or along tokens (PER_CHANNEL, KIVI-style keys).
    """

    bits: int = 4
    group_size: int = 16
    symmetric: bool = True
