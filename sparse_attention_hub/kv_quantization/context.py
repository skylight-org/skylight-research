"""Shared attention metadata and opt-in tracing; contains no KV storage."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass(frozen=True)
class AttentionContext:
    layer_idx: int
    tensor_kind: str
    start_token: int
    phase: str = "decode"
    position_ids: Optional[torch.Tensor] = None


def record_trace(metadata: Dict[str, Any], stage: str, **details: Any) -> None:
    """Opt-in lightweight teaching trace; never stores tensors."""
    events = metadata.get("kv_quantization_trace")
    if events is not None:
        events.append({"stage": stage, **details})
