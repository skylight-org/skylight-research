"""KV cache quantization.

Two backends, named directly by the HuggingFace adapter's
``quantize_kv_cache`` flag:

- ``"nvfp4"`` — packed NVFP4 via ``nvidia-modelopt``. Saves memory, needs a CUDA
  GPU and modelopt installed.
- ``"fake_nvfp4"`` — simulated NVFP4: quantize and dequantize in place, in pure
  torch. Same accuracy loss, no memory saving, runs anywhere.

``quantize_kv_cache=False`` disables quantization entirely.

Only the factory and the constants are imported here; the backends are imported
on use, so the fake path never touches modelopt. Nothing in the core package
imports this one eagerly.
"""

from .constants import (
    AXIS_PER_CHANNEL,
    AXIS_PER_TOKEN,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_RESIDUAL_LENGTH,
    E2M1_MAX,
    E4M3_MAX,
    E4M3_MIN_SUBNORMAL,
    VALID_AXES,
)
from .factory import (
    BACKENDS,
    FAKE_NVFP4,
    NVFP4,
    create_quantized_kv_cache,
    describe_backends,
    validate_backend,
)

__all__ = [
    "AXIS_PER_CHANNEL",
    "AXIS_PER_TOKEN",
    "BACKENDS",
    "DEFAULT_BLOCK_SIZE",
    "DEFAULT_RESIDUAL_LENGTH",
    "E2M1_MAX",
    "E4M3_MAX",
    "E4M3_MIN_SUBNORMAL",
    "FAKE_NVFP4",
    "NVFP4",
    "VALID_AXES",
    "create_quantized_kv_cache",
    "describe_backends",
    "validate_backend",
]
