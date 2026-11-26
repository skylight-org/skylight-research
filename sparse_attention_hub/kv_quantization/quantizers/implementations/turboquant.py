"""TurboQuant Algorithms 1/2 with original-coordinate dense reconstruction.

Reference: https://arxiv.org/pdf/2504.19874v1, Sections 3.1 and 3.2.
Independent implementation of the mathematical algorithms; no community code is
copied. FP32 is our explicit runtime numerical policy, not an author kernel claim.
"""

import hashlib
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import torch

from ...cache.block import QuantizedTensor
from ...context import AttentionContext
from ...numerics import finite_cast, require_finite
from ...policies.precision.base import PrecisionPlan
from ...registry import KVQuantizerRegistry
from ...transforms.implementations.random_rotation import (
    RandomRotation,
    RandomRotationConfig,
    random_orthogonal,
)
from ..base import KVQuantizer, KVQuantizerConfig, ParamDType, QuantGranularity
from ..common import check_input, pack_bitstream, unpack_bitstream
from .non_uniform import lloyd_max_gaussian, nearest_level

TURBOQUANT_REFERENCE = "arxiv:2504.19874v1:algorithms-1-2"
TURBOQUANT_NUMERICS = "fp32-beta-gl96-lloyd2000-tol1e-11-qr64-qjl64-v1"
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_DISTRIBUTIONS = ("sphere_beta", "gaussian_limit")
# Channel groups of the outlier split; "" is the ordinary whole-vector codec.
_GROUPS = ("outlier", "normal")
_CENTER_KINDS = ("key", "value")


def _check_dimension(dimension: int) -> None:
    if type(dimension) is not int or dimension < 2:
        raise ValueError("TurboQuant requires a head dimension of at least 2")


def _check_bits(bits: int) -> None:
    if type(bits) is not int or bits not in (1, 2, 3, 4):
        raise ValueError("TurboQuant supports 1, 2, 3 or 4 nominal bits")


def sphere_lloyd_max(
    dimension: int,
    bits: int,
    *,
    tolerance: float = 1e-11,
    max_iterations: int = 2000,
    quadrature_order: int = 96,
) -> torch.Tensor:
    """Solve the sphere-coordinate scalar quantizer in FP64, returning CPU levels.

    With z=sin(theta), the density times dz is proportional to
    cos(theta)**(dimension-2). Conditional log weights avoid tiny tail masses.
    """
    _check_dimension(dimension)
    _check_bits(bits)
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or tolerance <= 0
        or type(max_iterations) is not int
        or max_iterations < 1
        or type(quadrature_order) is not int
        or quadrature_order < 16
    ):
        raise ValueError("Invalid Lloyd-Max convergence or quadrature settings")
    nodes_np, weights_np = np.polynomial.legendre.leggauss(quadrature_order)
    nodes = torch.tensor(nodes_np, dtype=torch.float64)
    log_weights = torch.tensor(weights_np, dtype=torch.float64).log()
    count = 2**bits
    positions = torch.arange(count, dtype=torch.float64)
    if dimension <= 4:
        centres = -1 + (2 * positions + 1) / count
    else:
        normal = torch.distributions.Normal(0.0, 1.0)
        centres = normal.icdf((positions + 0.5) / count) / math.sqrt(dimension)
    for _ in range(max_iterations):
        edges = torch.cat(
            (
                centres.new_tensor([-1.0]),
                (centres[:-1] + centres[1:]) / 2,
                centres.new_tensor([1.0]),
            )
        )
        if not (edges[1:] > edges[:-1]).all():
            raise ValueError("Sphere codebook intervals must have positive width")
        angles = edges.asin()
        mid = (angles[:-1] + angles[1:]) / 2
        half = (angles[1:] - angles[:-1]) / 2
        theta = mid[:, None] + half[:, None] * nodes
        log_mass = log_weights + (dimension - 2) * theta.cos().log()
        updated = (torch.softmax(log_mass, dim=-1) * theta.sin()).sum(dim=-1)
        updated = (updated - updated.flip(0)) / 2
        require_finite(updated, name="Sphere codebook centres")
        difference = (updated - centres).abs().max().item()
        centres = updated
        if difference <= tolerance:
            if not (centres[1:] > centres[:-1]).all() or (centres.abs() >= 1).any():
                raise ValueError("Invalid converged sphere codebook")
            return centres
    raise RuntimeError("Sphere Lloyd-Max solver did not converge")


@lru_cache(maxsize=64)
def _codebook_values(dimension: int, bits: int, distribution: str) -> Tuple[float, ...]:
    # Cache immutable Python values, never a mutable tensor shared with callers.
    if distribution == "sphere_beta":
        levels = sphere_lloyd_max(dimension, bits).float()
    else:
        levels = lloyd_max_gaussian(bits).clone() / math.sqrt(dimension)
    require_finite(levels, name="TurboQuant codebook")
    if not (levels[1:] > levels[:-1]).all():
        raise ValueError("TurboQuant codebook must be strictly ordered")
    return tuple(levels.tolist())


def turboquant_codebook(
    dimension: int, bits: int, distribution: str = "sphere_beta"
) -> torch.Tensor:
    """Owned FP32 copy of a dimension-specific table; no sample calibration."""
    _check_dimension(dimension)
    _check_bits(bits)
    if distribution not in _DISTRIBUTIONS:
        raise ValueError("Unknown TurboQuant codebook distribution")
    return torch.tensor(
        _codebook_values(dimension, bits, distribution), dtype=torch.float32
    )


def stable_norm_and_unit(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize original vectors in FP32 without squaring their raw magnitude."""
    data = x.detach().float()
    largest = data.abs().amax(dim=-1, keepdim=True)
    divisor = torch.where(largest > 0, largest, torch.ones_like(largest))
    rho = torch.linalg.vector_norm(data / divisor, dim=-1, keepdim=True) * largest
    require_finite(rho, name="TurboQuant original norms")
    unit = data / torch.where(rho > 0, rho, torch.ones_like(rho))
    require_finite(unit, name="TurboQuant normalized vectors")
    return rho, unit


def _fingerprint(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str((tuple(data.shape), data.dtype)).encode())
    digest.update(data.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class TurboQuantQuantizerConfig(KVQuantizerConfig):
    """Whole-head scalar indices and optional residual QJL correction.

    bits is the total nominal budget: prod uses bits-1 coarse bits plus one sign
    bit. Norms are stored in FP32; inherited scale/offset/group/symmetric settings
    must keep unused defaults. Rotation is internal so residuals use original
    coordinates and normalization occurs before rotation.

    A plan with ``channel_bits`` (ChannelBoost) splits each vector into its
    boosted ("outlier") and other ("normal") channels and runs two independent
    TurboQuant instances, each with its own norm, rotation, codebook and QJL
    matrix: the paper's fractional-bit KV setup (e.g. 32 channels at 3 bits and
    96 at 2 bits = 2.5 bits). ``bits`` is then the normal channels' width.

    ``center`` (not in the paper) lists tensor kinds, e.g. ``("key",)``, whose
    per-channel mean is removed before quantization and added back after. The
    mean is taken over the first chunk of a request (its prefill) per layer and
    head, reused for every later chunk, and stored in FP32 in the record's
    ``zero_points``. It removes the large offset every Qwen key shares, which
    attention ignores (softmax is shift-invariant) but which otherwise inflates
    TurboQuant's error, since that error scales with the whole vector norm.
    """

    supports_channel_bits: ClassVar[bool] = True
    channel_bits_need_per_channel: ClassVar[bool] = False

    bits: int = 3
    variant: str = "mse"
    codebook_distribution: str = "sphere_beta"
    rotation: RandomRotationConfig = field(default_factory=RandomRotationConfig)
    qjl_seed: int = 1
    center: Tuple[str, ...] = ()
    parameter_dtype: str = field(default="fp32", init=False)
    reference: str = field(default=TURBOQUANT_REFERENCE, init=False)
    numerics_version: str = field(default=TURBOQUANT_NUMERICS, init=False)


@dataclass(frozen=True)
class _Resources:
    rotation: torch.Tensor
    levels: torch.Tensor
    projection: Optional[torch.Tensor]
    identities: Tuple[str, str, Optional[str]]


@KVQuantizerRegistry.register(TurboQuantQuantizerConfig)
class TurboQuantQuantizer(KVQuantizer):
    """Paper MSE/prod codecs, independent of query or attention implementation."""

    config: TurboQuantQuantizerConfig

    def __init__(self, config: TurboQuantQuantizerConfig) -> None:
        super().__init__(config)
        _check_bits(config.bits)
        if config.variant not in ("mse", "prod"):
            raise ValueError("TurboQuant variant must be mse or prod")
        if config.codebook_distribution not in _DISTRIBUTIONS:
            raise ValueError("Unknown TurboQuant codebook distribution")
        if (
            QuantGranularity(config.granularity) != QuantGranularity.PER_TOKEN
            or config.group_size != 32
            or type(config.group_size) is not int
            or config.symmetric is not False
            or config.scale_dtype != ParamDType.FP16
            or config.offset_dtype != ParamDType.FP16
        ):
            raise ValueError(
                "TurboQuant normalizes whole token/head vectors in FP32; leave "
                "group_size, symmetric, scale_dtype and offset_dtype at defaults"
            )
        if (
            type(config.rotation) is not RandomRotationConfig
            or type(config.rotation.seed) is not int
            or not 0 <= config.rotation.seed < 2**31
            or type(config.qjl_seed) is not int
            or not 0 <= config.qjl_seed < 2**63
        ):
            raise ValueError("Invalid TurboQuant rotation or QJL seed")
        if (
            type(config.center) is not tuple
            or len(set(config.center)) != len(config.center)
            or any(kind not in _CENTER_KINDS for kind in config.center)
        ):
            raise ValueError(f"center must be a tuple of distinct kinds from {_CENTER_KINDS}")
        self._rotation = RandomRotation(config.rotation)
        self._books: Dict[Tuple[int, int], torch.Tensor] = {}
        self._cpu: Dict[Tuple[Any, ...], _Resources] = {}
        self._devices: Dict[Tuple[Any, ...], _Resources] = {}
        # (layer, kind) -> (channel_bits object, its outlier channel indices), so a
        # frozen ChannelBoost choice yields one shared index tensor.
        self._split_indices: Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor]] = {}
        # (layer, kind) -> (start of the last chunk seen, frozen FP32 mean [B,H,1,D]).
        self._centers: Dict[Tuple[int, str], Tuple[int, torch.Tensor]] = {}

    @property
    def coarse_bits(self) -> int:
        return self.config.bits - (self.config.variant == "prod")

    @staticmethod
    def _validate_context(context: Optional[AttentionContext]) -> AttentionContext:
        if (
            context is None
            or type(context.layer_idx) is not int
            or not 0 <= context.layer_idx < 2**31
            or context.tensor_kind not in ("key", "value")
        ):
            raise ValueError("TurboQuant requires a valid layer/key-or-value context")
        return context

    def _group_coarse_bits(self, bits: int) -> int:
        return bits - (self.config.variant == "prod")

    def _resources(
        self,
        context: AttentionContext,
        dimension: int,
        device: torch.device,
        group: str = "",
        coarse_bits: Optional[int] = None,
    ) -> _Resources:
        """Rotation, codebook and QJL matrix. ``group`` "" is the whole-vector
        codec (unchanged seeds); split groups get their own seeded matrices."""
        coarse = self.coarse_bits if coarse_bits is None else coarse_bits
        key = (context.layer_idx, context.tensor_kind, dimension)
        full_key = (*key, group, coarse)
        if full_key not in self._cpu:
            if group:
                tag = (
                    f"turboquant-group-rotation-v1:{self.config.rotation.seed}:"
                    f"{key}:{group}"
                )
                seed = int.from_bytes(
                    hashlib.sha256(tag.encode()).digest()[:8], "little"
                ) % (2**63)
                rotation = random_orthogonal(dimension, seed)
            else:
                rotation = self._rotation.matrix(*key)
            require_finite(rotation, name="TurboQuant rotation")
            if (dimension, coarse) not in self._books:
                self._books[(dimension, coarse)] = (
                    turboquant_codebook(
                        dimension, coarse, self.config.codebook_distribution
                    )
                    if coarse
                    else torch.zeros(1, dtype=torch.float32)
                )
            levels = self._books[(dimension, coarse)]
            projection = None
            if self.config.variant == "prod":
                tag = f"turboquant-qjl-v1:{self.config.qjl_seed}:{key}"
                if group:
                    tag = f"{tag}:{group}"
                seed = int.from_bytes(
                    hashlib.sha256(tag.encode()).digest()[:8], "little"
                ) % (2**63)
                generator = torch.Generator(device="cpu").manual_seed(seed)
                projection = torch.randn(
                    dimension, dimension, generator=generator, dtype=torch.float64
                ).float()
            identities = (
                _fingerprint(rotation),
                _fingerprint(levels),
                None if projection is None else _fingerprint(projection),
            )
            self._cpu[full_key] = _Resources(rotation, levels, projection, identities)
        device_key = (*full_key, device)
        if device_key not in self._devices:
            cpu = self._cpu[full_key]
            self._devices[device_key] = _Resources(
                cpu.rotation.to(device),
                cpu.levels.to(device),
                None if cpu.projection is None else cpu.projection.to(device),
                cpu.identities,
            )
        return self._devices[device_key]

    def state_for(
        self, context: AttentionContext, dimension: int
    ) -> Dict[str, torch.Tensor]:
        """Owned inspection copies, also usable as explicit reference inputs."""
        context = self._validate_context(context)
        _check_dimension(dimension)
        state = self._resources(context, dimension, torch.device("cpu"))
        result = {"rotation": state.rotation.clone(), "codebook": state.levels.clone()}
        if state.projection is not None:
            result["projection"] = state.projection.clone()
        return result

    def shared_state_bytes(self) -> int:
        """Owned matrix/table tensor allocations; distinct from request payload."""
        allocations = {}
        for state in (*self._cpu.values(), *self._devices.values()):
            for tensor in (state.rotation, state.levels, state.projection):
                if tensor is not None:
                    storage = tensor.untyped_storage()
                    allocations[(tensor.device, storage.data_ptr())] = storage.nbytes()
        return sum(allocations.values())

    def _metadata(
        self, context: AttentionContext, dimension: int, count: int, state: _Resources
    ) -> Dict[str, Any]:
        return {
            "reference": TURBOQUANT_REFERENCE,
            "numerics_version": TURBOQUANT_NUMERICS,
            "variant": self.config.variant,
            "distribution": self.config.codebook_distribution,
            "layer_idx": context.layer_idx,
            "tensor_kind": context.tensor_kind,
            "dimension": dimension,
            "rotation_seed": self.config.rotation.seed,
            "qjl_seed": self.config.qjl_seed,
            "coarse_bits": self.coarse_bits,
            "code_count": count if self.coarse_bits else 0,
            "qjl_code_count": count if self.config.variant == "prod" else 0,
            "rotation_id": state.identities[0],
            "codebook_id": state.identities[1],
            "projection_id": state.identities[2],
        }

    def quantize(
        self,
        x: torch.Tensor,
        plan: Optional[PrecisionPlan] = None,
        *,
        context: Optional[AttentionContext] = None,
    ) -> QuantizedTensor:
        check_input(x, plan, allow_channel_bits=True)
        context = self._validate_context(context)
        if x.dtype not in _DTYPES:
            raise ValueError("TurboQuant inputs must be FP16, BF16 or FP32")
        if plan is not None and (
            type(plan.bits) is not int or plan.bits != self.config.bits
        ):
            raise ValueError("TurboQuant precision plan must match its configured bits")
        dimension = x.shape[-1]
        _check_dimension(dimension)
        mean = self._center_for(x, context)
        source = x if mean is None else x.float() - mean
        if plan is not None and plan.channel_bits is not None:
            record = self._quantize_split(source, plan.channel_bits, context)
        else:
            record = self._quantize_whole(source, context)
        if mean is not None:
            record.dtype = x.dtype
            record.zero_points = mean
            record.metadata["centered"] = True
        return record

    def _center_for(
        self, x: torch.Tensor, context: AttentionContext
    ) -> Optional[torch.Tensor]:
        """Frozen per-channel mean for a centered kind; a chunk starting at or
        before the previous one (a new request), or a new shape/device, takes a
        new mean from its own tokens."""
        if context.tensor_kind not in self.config.center:
            return None
        key = (context.layer_idx, context.tensor_kind)
        start = int(context.start_token)
        cached = self._centers.get(key)
        if (
            cached is None
            or start <= cached[0]
            or cached[1].shape != (x.shape[0], x.shape[1], 1, x.shape[3])
            or cached[1].device != x.device
        ):
            mean = x.float().mean(dim=-2, keepdim=True).contiguous()
            require_finite(mean, name="TurboQuant channel mean")
        else:
            mean = cached[1]
        self._centers[key] = (start, mean)
        return mean

    def _check_center(self, q: QuantizedTensor) -> None:
        """A centered record carries its FP32 mean [B,H,1,D]; others carry none."""
        centered = q.metadata.get("centered")
        if q.zero_points is None:
            if centered is not None:
                raise ValueError("Centered TurboQuant record lacks its channel mean")
            return
        mean = q.zero_points
        if (
            centered is not True
            or q.metadata.get("tensor_kind") not in self.config.center
            or mean.dtype != torch.float32
            or mean.device != q.device
            or tuple(mean.shape) != (q.shape[0], q.shape[1], 1, q.shape[3])
        ):
            raise ValueError("Invalid TurboQuant channel mean")
        require_finite(mean, name="TurboQuant channel mean")

    @staticmethod
    def _uncenter(output: torch.Tensor, q: QuantizedTensor) -> torch.Tensor:
        return output if q.zero_points is None else output + q.zero_points

    def _quantize_whole(
        self, x: torch.Tensor, context: AttentionContext
    ) -> QuantizedTensor:
        dimension = x.shape[-1]
        state = self._resources(context, dimension, x.device)
        codes, rho, qjl_codes, gamma = self._encode(x, state, self.coarse_bits)
        return QuantizedTensor(
            codes=codes,
            scales=torch.empty(0, dtype=torch.float32, device=x.device),
            zero_points=None,
            shape=tuple(x.shape),
            dtype=x.dtype,
            device=x.device,
            axis=3,
            bits=self.config.bits,
            format_id=f"turboquant_{self.config.variant}_v1",
            group_size=dimension,
            metadata=self._metadata(context, dimension, x.numel(), state),
            norms=rho.detach().clone().contiguous(),
            qjl_codes=qjl_codes,
            residual_norms=None
            if gamma is None
            else gamma.detach().clone().contiguous(),
        )

    @staticmethod
    def _encode(
        x: torch.Tensor, state: _Resources, coarse_bits: int
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """Algorithms 1/2 on whole vectors: codes, norms, QJL signs, residual norms."""
        rho, unit = stable_norm_and_unit(x)
        if coarse_bits:
            indices = nearest_level(unit @ state.rotation, state.levels)
            indices = indices.masked_fill(rho == 0, 0)
            codes, _ = pack_bitstream(indices, coarse_bits)
            coarse = state.levels[indices] @ state.rotation.T
            coarse = coarse.masked_fill(rho == 0, 0)
        else:
            codes = torch.empty(0, dtype=torch.uint8, device=x.device)
            coarse = torch.zeros_like(unit)
        qjl_codes, gamma = None, None
        if state.projection is not None:
            residual = unit - coarse
            gamma = torch.linalg.vector_norm(residual, dim=-1, keepdim=True)
            require_finite(gamma, name="TurboQuant residual norms")
            qjl_codes, _ = pack_bitstream(residual @ state.projection.T >= 0, 1)
        return codes, rho, qjl_codes, gamma

    @staticmethod
    def _decode(
        codes: torch.Tensor,
        shape: Tuple[int, ...],
        norms: torch.Tensor,
        qjl_codes: Optional[torch.Tensor],
        residual_norms: Optional[torch.Tensor],
        state: _Resources,
        coarse_bits: int,
    ) -> torch.Tensor:
        """FP32 reconstruction of ``_encode``'s output (before the dtype cast)."""
        count = math.prod(shape)
        if coarse_bits:
            indices = unpack_bitstream(codes, coarse_bits, count)
            coarse = state.levels[indices.long().reshape(shape)] @ state.rotation.T
            coarse = coarse.masked_fill(norms == 0, 0)
        else:
            if codes.numel():
                raise ValueError("Zero-bit coarse stage requires an empty index stream")
            coarse = torch.zeros(shape, dtype=torch.float32, device=norms.device)
        if state.projection is not None:
            if qjl_codes is None or residual_norms is None:
                raise ValueError("TurboQuant prod requires residual sign codes")
            signs = unpack_bitstream(qjl_codes, 1, count).float()
            signs = (2 * signs - 1).reshape(shape)
            coarse = coarse + (
                (signs @ state.projection)
                * residual_norms
                * (math.sqrt(math.pi / 2) / shape[-1])
            )
        return coarse * norms

    # ----- outlier-channel split (two independent TurboQuant instances) -----

    def _split_widths(
        self, channel_bits: torch.Tensor, x: torch.Tensor
    ) -> Tuple[int, int]:
        """(outlier bits, outlier channels per head) of a two-width plan."""
        if (
            tuple(channel_bits.shape) != (x.shape[0], x.shape[1], x.shape[3])
            or channel_bits.dtype.is_floating_point
            or channel_bits.dtype == torch.bool
            or channel_bits.device != x.device
        ):
            raise ValueError(
                "channel_bits must be integer [batch, heads, head_dim] on the "
                "input device"
            )
        widths = sorted(int(w) for w in channel_bits.unique().tolist())
        high = widths[-1]
        if len(widths) != 2 or widths[0] != self.config.bits or high > 4:
            raise ValueError(
                "TurboQuant outlier split needs exactly two widths: the configured "
                f"bits ({self.config.bits}) and a larger one of at most 4"
            )
        counts = (channel_bits == high).sum(dim=-1)
        count = int(counts.flatten()[0])
        if not (counts == count).all():
            raise ValueError("Every head must boost the same number of channels")
        if count < 2 or x.shape[-1] - count < 2:
            raise ValueError("Both TurboQuant channel groups need at least 2 channels")
        return high, count

    def _outlier_indices(
        self,
        context: AttentionContext,
        channel_bits: torch.Tensor,
        high: int,
        count: int,
    ) -> torch.Tensor:
        """Ascending boosted channels [batch, heads, count] as int16. The same
        tensor is reused while the plan passes the same channel_bits object (a
        frozen ChannelBoost choice), so the store can count it once."""
        key = (context.layer_idx, context.tensor_kind)
        cached = self._split_indices.get(key)
        if cached is not None and cached[0] is channel_bits:
            return cached[1]
        order = torch.argsort(
            (channel_bits != high).to(torch.int8), dim=-1, stable=True
        )
        indices = order[..., :count].to(torch.int16).contiguous()
        self._split_indices[key] = (channel_bits, indices)
        return indices

    @staticmethod
    def _split_order(indices: torch.Tensor, dimension: int) -> torch.Tensor:
        """Permutation [batch, heads, dimension]: outlier channels first, then
        the others, each in ascending channel order."""
        mask = torch.zeros(
            (*indices.shape[:-1], dimension), dtype=torch.bool, device=indices.device
        )
        mask.scatter_(-1, indices.long(), True)
        return torch.argsort((~mask).to(torch.int8), dim=-1, stable=True)

    def _group_states(
        self,
        context: AttentionContext,
        high: int,
        count: int,
        dimension: int,
        device: torch.device,
    ) -> List[_Resources]:
        return [
            self._resources(context, size, device, name, self._group_coarse_bits(bits))
            for name, bits, size in zip(
                _GROUPS, (high, self.config.bits), (count, dimension - count)
            )
        ]

    def _split_metadata(
        self,
        context: AttentionContext,
        dimension: int,
        high: int,
        count: int,
        vectors: int,
        states: List[_Resources],
    ) -> Dict[str, Any]:
        groups: Dict[str, Any] = {}
        for name, bits, size, state in zip(
            _GROUPS, (high, self.config.bits), (count, dimension - count), states
        ):
            groups[name] = {
                "bits": bits,
                "coarse_bits": self._group_coarse_bits(bits),
                "dimension": size,
                "rotation_id": state.identities[0],
                "codebook_id": state.identities[1],
                "projection_id": state.identities[2],
            }
        return {
            "reference": TURBOQUANT_REFERENCE,
            "numerics_version": TURBOQUANT_NUMERICS,
            "variant": self.config.variant,
            "distribution": self.config.codebook_distribution,
            "layer_idx": context.layer_idx,
            "tensor_kind": context.tensor_kind,
            "dimension": dimension,
            "rotation_seed": self.config.rotation.seed,
            "qjl_seed": self.config.qjl_seed,
            "vectors": vectors,
            "groups": groups,
        }

    def _quantize_split(
        self, x: torch.Tensor, channel_bits: torch.Tensor, context: AttentionContext
    ) -> QuantizedTensor:
        high, count = self._split_widths(channel_bits, x)
        batch, heads, tokens, dimension = x.shape
        indices = self._outlier_indices(context, channel_bits, high, count)
        order = self._split_order(indices, dimension)
        permuted = x.gather(-1, order[:, :, None, :].expand(x.shape))
        parts = (permuted[..., :count], permuted[..., count:])
        states = self._group_states(context, high, count, dimension, x.device)
        encoded = [
            self._encode(part, state, self._group_coarse_bits(bits))
            for part, state, bits in zip(parts, states, (high, self.config.bits))
        ]
        metadata = self._split_metadata(
            context, dimension, high, count, batch * heads * tokens, states
        )
        metadata["outlier_code_bytes"] = int(encoded[0][0].numel())
        qjl_codes, gamma = None, None
        if self.config.variant == "prod":
            qjl_codes = torch.cat([e[2] for e in encoded])
            gamma = torch.cat([e[3] for e in encoded], dim=-1).detach().contiguous()
            metadata["outlier_qjl_bytes"] = int(encoded[0][2].numel())
        return QuantizedTensor(
            codes=torch.cat([e[0] for e in encoded]),
            scales=torch.empty(0, dtype=torch.float32, device=x.device),
            zero_points=None,
            shape=tuple(x.shape),
            dtype=x.dtype,
            device=x.device,
            axis=3,
            bits=self.config.bits,
            format_id=f"turboquant_{self.config.variant}_outlier_v1",
            group_size=dimension,
            metadata=metadata,
            norms=torch.cat([e[1] for e in encoded], dim=-1).detach().contiguous(),
            outlier_indices=indices,
            qjl_codes=qjl_codes,
            residual_norms=gamma,
        )

    def _dequantize_split(self, q: QuantizedTensor) -> torch.Tensor:
        if (
            len(q.shape) != 4
            or any(type(n) is not int or n < 1 for n in q.shape)
            or q.axis != 3
            or q.bits != self.config.bits
            or q.group_size != q.shape[-1]
            or q.dtype not in _DTYPES
            or q.scales.numel() != 0
            or q.codes.dtype != torch.uint8
            or q.codes.ndim != 1
        ):
            raise ValueError("Invalid TurboQuant outlier-split layout")
        self._check_center(q)
        batch, heads, tokens, dimension = q.shape
        indices = q.outlier_indices
        if (
            indices is None
            or indices.dtype != torch.int16
            or indices.ndim != 3
            or tuple(indices.shape[:2]) != (batch, heads)
            or indices.device != q.device
        ):
            raise ValueError(
                "TurboQuant outlier split needs [batch, heads, k] int16 indices"
            )
        count = indices.shape[-1]
        if (
            (indices < 0).any()
            or (indices >= dimension).any()
            or (count > 1 and not (indices[..., 1:] > indices[..., :-1]).all())
        ):
            raise ValueError("Outlier channel indices must be ascending and in range")
        meta = q.metadata
        groups = meta.get("groups")
        high = groups.get("outlier", {}).get("bits") if isinstance(groups, dict) else None
        if type(high) is not int or not self.config.bits < high <= 4:
            raise ValueError("TurboQuant outlier record lacks a valid outlier width")
        context = self._validate_context(
            AttentionContext(meta.get("layer_idx"), meta.get("tensor_kind"), 0)
        )
        states = self._group_states(context, high, count, dimension, q.device)
        expected = self._split_metadata(
            context, dimension, high, count, batch * heads * tokens, states
        )
        if any(meta.get(k) != v for k, v in expected.items()):
            raise ValueError(
                "TurboQuant record metadata/codebook/matrix fingerprint mismatch"
            )
        prod = self.config.variant == "prod"
        for parameter in [q.norms] + ([q.residual_norms] if prod else []):
            if (
                parameter is None
                or parameter.dtype != torch.float32
                or parameter.device != q.device
                or tuple(parameter.shape) != (batch, heads, tokens, 2)
            ):
                raise ValueError("Invalid TurboQuant split norm shape/dtype/device")
            require_finite(parameter, name="TurboQuant stored norms")
            if (parameter < 0).any():
                raise ValueError("TurboQuant norms must be nonnegative")
        if prod and (q.qjl_codes is None or q.qjl_codes.device != q.device):
            raise ValueError("TurboQuant prod requires residual sign codes")
        if not prod and (q.qjl_codes is not None or q.residual_norms is not None):
            raise ValueError("TurboQuant MSE cannot contain a QJL payload")
        split = meta.get("outlier_code_bytes")
        qjl_split = meta.get("outlier_qjl_bytes") if prod else 0
        if type(split) is not int or type(qjl_split) is not int:
            raise ValueError("TurboQuant outlier record lacks its stream split")
        code_parts = (q.codes[:split], q.codes[split:])
        qjl_parts: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (
            (q.qjl_codes[:qjl_split], q.qjl_codes[qjl_split:])
            if prod and q.qjl_codes is not None
            else (None, None)
        )
        decoded = []
        for i, (state, bits, size) in enumerate(
            zip(states, (high, self.config.bits), (count, dimension - count))
        ):
            decoded.append(
                self._decode(
                    code_parts[i],
                    (batch, heads, tokens, size),
                    q.norms[..., i : i + 1],
                    qjl_parts[i],
                    None if q.residual_norms is None else q.residual_norms[..., i : i + 1],
                    state,
                    self._group_coarse_bits(bits),
                )
            )
        order = self._split_order(indices, dimension)
        output = torch.empty(q.shape, dtype=torch.float32, device=q.device)
        output.scatter_(
            -1, order[:, :, None, :].expand(q.shape), torch.cat(decoded, dim=-1)
        )
        return finite_cast(self._uncenter(output, q), q.dtype)

    def _validate_record(
        self, q: QuantizedTensor
    ) -> Tuple[AttentionContext, _Resources]:
        if q.format_id != f"turboquant_{self.config.variant}_v1":
            raise ValueError(
                "TurboQuant record variant/format does not match this codec"
            )
        if (
            len(q.shape) != 4
            or any(type(n) is not int or n < 1 for n in q.shape)
            or q.shape[-1] < 2
            or type(q.bits) is not int
            or q.bits != self.config.bits
            or type(q.group_size) is not int
            or q.group_size != q.shape[-1]
            or q.axis != 3
            or q.dtype not in _DTYPES
        ):
            raise ValueError("Invalid TurboQuant stored layout")
        if q.scales.numel() != 0:
            raise ValueError("TurboQuant does not store scales")
        self._check_center(q)
        if q.scales.device != q.device or q.scales.dtype != torch.float32:
            raise ValueError("Invalid TurboQuant empty scale buffer")
        for extra in (
            q.global_scale,
            q.codebook,
            q.outlier_indices,
            q.outlier_values,
            q.group_bits,
        ):
            if extra is not None:
                raise ValueError("Unexpected TurboQuant payload field")
        context = self._validate_context(
            AttentionContext(
                q.metadata.get("layer_idx"), q.metadata.get("tensor_kind"), 0
            )
        )
        state = self._resources(context, q.shape[-1], q.device)
        expected = self._metadata(context, q.shape[-1], math.prod(q.shape), state)
        if any(
            type(q.metadata.get(k)) is not type(v) or q.metadata.get(k) != v
            for k, v in expected.items()
        ):
            raise ValueError(
                "TurboQuant record metadata/codebook/matrix fingerprint mismatch"
            )
        if (
            q.codes.device != q.device
            or q.codes.dtype != torch.uint8
            or q.codes.ndim != 1
        ):
            raise ValueError("Invalid TurboQuant index stream")
        norm_shape = (*q.shape[:-1], 1)
        parameters = [q.norms]
        if self.config.variant == "prod":
            parameters.append(q.residual_norms)
            if q.qjl_codes is None or q.qjl_codes.device != q.device:
                raise ValueError("TurboQuant prod requires residual sign codes")
        elif q.residual_norms is not None or q.qjl_codes is not None:
            raise ValueError("TurboQuant MSE cannot contain a QJL payload")
        for parameter in parameters:
            if (
                parameter is None
                or parameter.dtype != torch.float32
                or parameter.device != q.device
                or tuple(parameter.shape) != norm_shape
            ):
                raise ValueError("Invalid TurboQuant norm shape/dtype/device")
            require_finite(parameter, name="TurboQuant stored norms")
            if (parameter < 0).any():
                raise ValueError("TurboQuant norms must be nonnegative")
        if (
            q.residual_norms is not None
            and ((q.norms == 0) & (q.residual_norms != 0)).any()
        ):
            raise ValueError("Zero original vectors must have zero residual norms")
        return context, state

    def dequantize(self, q: QuantizedTensor) -> torch.Tensor:
        if q.format_id == f"turboquant_{self.config.variant}_outlier_v1":
            return self._dequantize_split(q)
        _, state = self._validate_record(q)
        reconstructed = self._decode(
            q.codes,
            tuple(q.shape),
            q.norms,
            q.qjl_codes,
            q.residual_norms,
            state,
            self.coarse_bits,
        )
        return finite_cast(self._uncenter(reconstructed, q), q.dtype)
