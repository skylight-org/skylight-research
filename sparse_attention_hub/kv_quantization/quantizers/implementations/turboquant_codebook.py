"""TurboQuant's scalar stage, with the random rotation moved to a KVTransform.

TurboQuant (arXiv 2504.19874v1, Algorithms 1 and 2) is: normalize each head
vector, rotate it randomly, snap every coordinate to a Lloyd-Max codebook of the
rotated-sphere coordinate distribution, and (``prod``) add 1-bit QJL signs of the
residual. ``TurboQuantQuantizer`` does all of it inside one codec. Here the
rotation is ``RandomRotation`` in the transform slot and this codec does the rest:

    original x --RandomRotation--> y = x R --this codec--> norm, codebook, QJL
    reconstruction <--R^T (inverse transform)-- decoded y

With the same rotation seed the two produce the same codebook indices (up to
FP32 rounding in the order of normalize/rotate). ``prod`` differs only in which
random projection the QJL signs use: the residual here is in rotated
coordinates, so the effective projection is S R instead of S. Both are i.i.d.
Gaussian matrices, so the estimator and its guarantees are the same.

The codebook assumes randomly rotated input; ``TurboQuantSplitKVConfig`` checks
that the transform slot holds ``RandomRotationConfig``.
"""

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch

from ...cache.block import QuantizedTensor
from ...context import AttentionContext
from ...numerics import finite_cast, require_finite
from ...policies.precision.base import PrecisionPlan
from ...registry import KVQuantizerRegistry
from ..base import KVQuantizer, KVQuantizerConfig, ParamDType, QuantGranularity
from ..common import check_input, pack_bitstream, unpack_bitstream
from .non_uniform import nearest_level
from .turboquant import (
    _DISTRIBUTIONS,
    TURBOQUANT_NUMERICS,
    TURBOQUANT_REFERENCE,
    _check_bits,
    _check_dimension,
    _fingerprint,
    stable_norm_and_unit,
    turboquant_codebook,
)

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@dataclass(frozen=True)
class TurboQuantCodebookQuantizerConfig(KVQuantizerConfig):
    """Norm + sphere codebook (+ QJL) for vectors that are already rotated.

    Attributes:
        bits: Nominal bits per coordinate; ``prod`` uses ``bits - 1`` codebook
            bits plus one QJL sign bit.
        variant: ``"mse"`` (Algorithm 1) or ``"prod"`` (Algorithm 2).
        codebook_distribution: ``"sphere_beta"`` (exact coordinate law) or
            ``"gaussian_limit"`` (its N(0, 1/d) limit).
        qjl_seed: Seed of the per-layer, per-kind QJL projection (``prod``).

    Norms are stored in FP32; ``group_size``, ``symmetric``, ``scale_dtype`` and
    ``offset_dtype`` are unused and must keep their defaults.
    """

    bits: int = 3
    variant: str = "mse"
    codebook_distribution: str = "sphere_beta"
    qjl_seed: int = 1
    parameter_dtype: str = field(default="fp32", init=False)
    reference: str = field(default=TURBOQUANT_REFERENCE, init=False)
    numerics_version: str = field(default=TURBOQUANT_NUMERICS, init=False)


@KVQuantizerRegistry.register(TurboQuantCodebookQuantizerConfig)
class TurboQuantCodebookQuantizer(KVQuantizer):
    """TurboQuant MSE/prod codec without its rotation (see the module docstring)."""

    config: TurboQuantCodebookQuantizerConfig

    def __init__(self, config: TurboQuantCodebookQuantizerConfig) -> None:
        super().__init__(config)
        _check_bits(config.bits)
        if config.variant not in ("mse", "prod"):
            raise ValueError("TurboQuant variant must be mse or prod")
        if config.codebook_distribution not in _DISTRIBUTIONS:
            raise ValueError("Unknown TurboQuant codebook distribution")
        if (
            QuantGranularity(config.granularity) != QuantGranularity.PER_TOKEN
            or type(config.group_size) is not int
            or config.group_size != 32
            or config.symmetric is not False
            or config.scale_dtype != ParamDType.FP16
            or config.offset_dtype != ParamDType.FP16
        ):
            raise ValueError(
                "TurboQuant normalizes whole token/head vectors in FP32; leave "
                "group_size, symmetric, scale_dtype and offset_dtype at defaults"
            )
        if type(config.qjl_seed) is not int or not 0 <= config.qjl_seed < 2**63:
            raise ValueError("Invalid TurboQuant QJL seed")
        self._books: Dict[int, torch.Tensor] = {}
        self._projections: Dict[Tuple[int, str, int], torch.Tensor] = {}

    @property
    def coarse_bits(self) -> int:
        return self.config.bits - (self.config.variant == "prod")

    def _levels(self, dimension: int, device: torch.device) -> torch.Tensor:
        if dimension not in self._books:
            self._books[dimension] = (
                turboquant_codebook(
                    dimension, self.coarse_bits, self.config.codebook_distribution
                )
                if self.coarse_bits
                else torch.zeros(1, dtype=torch.float32)
            )
        return self._books[dimension].to(device)

    def _projection(
        self, context: AttentionContext, dimension: int, device: torch.device
    ) -> torch.Tensor:
        """The QJL Gaussian matrix, seeded exactly as in TurboQuantQuantizer."""
        key = (context.layer_idx, context.tensor_kind, dimension)
        if key not in self._projections:
            tag = f"turboquant-qjl-v1:{self.config.qjl_seed}:{key}"
            seed = int.from_bytes(
                hashlib.sha256(tag.encode()).digest()[:8], "little"
            ) % (2**63)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            self._projections[key] = torch.randn(
                dimension, dimension, generator=generator, dtype=torch.float64
            ).float()
        return self._projections[key].to(device)

    @staticmethod
    def _validate_context(context: Optional[AttentionContext]) -> AttentionContext:
        if (
            context is None
            or type(context.layer_idx) is not int
            or not 0 <= context.layer_idx < 2**31
            or context.tensor_kind not in ("key", "value")
        ):
            raise ValueError(
                "TurboQuant prod requires a valid layer/key-or-value context"
            )
        return context

    def state_for(
        self, context: Optional[AttentionContext], dimension: int
    ) -> Dict[str, torch.Tensor]:
        """Owned copies of the codebook (and QJL projection for ``prod``)."""
        _check_dimension(dimension)
        result = {"codebook": self._levels(dimension, torch.device("cpu")).clone()}
        if self.config.variant == "prod":
            context = self._validate_context(context)
            result["projection"] = self._projection(
                context, dimension, torch.device("cpu")
            ).clone()
        return result

    def quantize(
        self,
        x: torch.Tensor,
        plan: Optional[PrecisionPlan] = None,
        *,
        context: Optional[AttentionContext] = None,
    ) -> QuantizedTensor:
        check_input(x, plan)
        if x.dtype not in _DTYPES:
            raise ValueError("TurboQuant inputs must be FP16, BF16 or FP32")
        if plan is not None and (
            type(plan.bits) is not int or plan.bits != self.config.bits
        ):
            raise ValueError("TurboQuant precision plan must match its configured bits")
        dimension = x.shape[-1]
        _check_dimension(dimension)
        levels = self._levels(dimension, x.device)
        # Algorithm 1 on the rotated vector: y / ||y||, nearest codebook entry.
        rho, unit = stable_norm_and_unit(x)
        if self.coarse_bits:
            indices = nearest_level(unit, levels).masked_fill(rho == 0, 0)
            codes, _ = pack_bitstream(indices, self.coarse_bits)
            coarse = levels[indices].masked_fill(rho == 0, 0)
        else:
            codes = torch.empty(0, dtype=torch.uint8, device=x.device)
            coarse = torch.zeros_like(unit)
        metadata: Dict[str, Any] = {
            "reference": TURBOQUANT_REFERENCE,
            "numerics_version": TURBOQUANT_NUMERICS,
            "variant": self.config.variant,
            "distribution": self.config.codebook_distribution,
            "dimension": dimension,
            "coarse_bits": self.coarse_bits,
            "code_count": x.numel() if self.coarse_bits else 0,
            "qjl_code_count": 0,
            "codebook_id": _fingerprint(levels),
        }
        qjl_codes, gamma = None, None
        if self.config.variant == "prod":
            # Algorithm 2: residual r = unit - coarse; signs of S r; ||r||.
            context = self._validate_context(context)
            projection = self._projection(context, dimension, x.device)
            residual = unit - coarse
            gamma = torch.linalg.vector_norm(residual, dim=-1, keepdim=True)
            require_finite(gamma, name="TurboQuant residual norms")
            qjl_codes, _ = pack_bitstream(residual @ projection.T >= 0, 1)
            metadata.update(
                layer_idx=context.layer_idx,
                tensor_kind=context.tensor_kind,
                qjl_seed=self.config.qjl_seed,
                qjl_code_count=x.numel(),
                projection_id=_fingerprint(projection),
            )
        return QuantizedTensor(
            codes=codes,
            scales=torch.empty(0, dtype=torch.float32, device=x.device),
            zero_points=None,
            shape=tuple(x.shape),
            dtype=x.dtype,
            device=x.device,
            axis=3,
            bits=self.config.bits,
            format_id=f"turboquant_codebook_{self.config.variant}_v1",
            group_size=dimension,
            metadata=metadata,
            norms=rho.detach().clone().contiguous(),
            qjl_codes=qjl_codes,
            residual_norms=None
            if gamma is None
            else gamma.detach().clone().contiguous(),
        )

    def dequantize(self, q: QuantizedTensor) -> torch.Tensor:
        if q.format_id != f"turboquant_codebook_{self.config.variant}_v1":
            raise ValueError(
                "TurboQuant record variant/format does not match this codec"
            )
        meta = q.metadata
        dimension = q.shape[-1]
        levels = self._levels(dimension, q.device)
        if (
            meta.get("dimension") != dimension
            or meta.get("coarse_bits") != self.coarse_bits
            or meta.get("distribution") != self.config.codebook_distribution
            or meta.get("codebook_id") != _fingerprint(levels)
        ):
            raise ValueError("TurboQuant record does not match this codec's codebook")
        if q.norms is None or q.norms.dtype != torch.float32:
            raise ValueError("TurboQuant records need FP32 norms")
        if self.coarse_bits:
            indices = unpack_bitstream(q.codes, self.coarse_bits, meta["code_count"])
            coarse = levels[indices.long().reshape(q.shape)]
            coarse = coarse.masked_fill(q.norms == 0, 0)
        else:
            if q.codes.numel():
                raise ValueError("Zero-bit coarse stage requires an empty index stream")
            coarse = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
        if self.config.variant == "prod":
            if q.qjl_codes is None or q.residual_norms is None:
                raise ValueError("TurboQuant prod requires residual sign codes")
            context = self._validate_context(
                AttentionContext(meta.get("layer_idx"), meta.get("tensor_kind"), 0)
            )
            projection = self._projection(context, dimension, q.device)
            if meta.get("projection_id") != _fingerprint(projection):
                raise ValueError("TurboQuant record does not match its QJL projection")
            signs = unpack_bitstream(q.qjl_codes, 1, meta["qjl_code_count"]).float()
            signs = (2 * signs - 1).reshape(q.shape)
            # DeQuant_prod: sqrt(pi/2) / d * gamma * S^T qjl.
            coarse = coarse + (
                (signs @ projection)
                * q.residual_norms
                * (math.sqrt(math.pi / 2) / dimension)
            )
        return finite_cast(coarse * q.norms, q.dtype)
