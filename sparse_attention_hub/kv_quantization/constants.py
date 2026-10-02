"""Constants shared by the real and fake NVFP4 backends.

Kept separate so importing them pulls in neither nvidia-modelopt nor torch.

Attributes:
    DEFAULT_BLOCK_SIZE (int): NVFP4's block size. 16 is the only value the
        hardware format defines, so other values produce a valid tensor that no
        kernel can consume.
    DEFAULT_RESIDUAL_LENGTH (int): How many trailing tokens stay unquantized.
    AXIS_PER_TOKEN (int): Block along ``head_dim`` — each block covers one
        token's channels.
    AXIS_PER_CHANNEL (int): Block along the token axis — each block covers one
        channel across tokens (the KIVI-style layout for keys).
    E2M1_MAX (float): Largest magnitude in the 4-bit element format, whose grid
        is exactly {0, .5, 1, 1.5, 2, 3, 4, 6}.
    E4M3_MAX (float): Largest magnitude of the 8-bit block scale. 448, not 480:
        the all-ones exponent with mantissa 111 is reserved for NaN.
    E4M3_MIN_SUBNORMAL (float): Smallest E4M3 value, 2^-9: the floor on block
        scales. It is what nvidia-modelopt (and so the real ``nvfp4`` backend)
        uses. Verified on molab 2026-09-28.
"""

from typing import Tuple

DEFAULT_BLOCK_SIZE: int = 16
DEFAULT_RESIDUAL_LENGTH: int = 128

AXIS_PER_TOKEN: int = -1
AXIS_PER_CHANNEL: int = 0
VALID_AXES: Tuple[int, int] = (AXIS_PER_TOKEN, AXIS_PER_CHANNEL)

E2M1_MAX: float = 6.0
E4M3_MAX: float = 448.0
E4M3_MIN_SUBNORMAL: float = 2.0**-9
