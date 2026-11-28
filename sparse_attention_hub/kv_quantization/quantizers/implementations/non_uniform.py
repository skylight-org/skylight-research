"""NonUniformQuantizer: values snap to a table of levels instead of an even grid."""

import hashlib
import math
from dataclasses import dataclass
from functools import lru_cache
from threading import RLock
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from ...cache.block import QuantizedTensor
from ...context import AttentionContext
from ...numerics import finite_cast
from ...policies.precision.base import PrecisionPlan
from ...registry import KVQuantizerRegistry
from ..base import KVQuantizer, KVQuantizerConfig, QuantGranularity
from ..common import check_input, pack_codes, store_param, unpack_codes

MODES = ("gaussian", "calibrated")
TableKey = Tuple[int, str]  # (layer_idx, "key" | "value")


@dataclass(frozen=True)
class NonUniformQuantizerConfig(KVQuantizerConfig):
    """Snap each value to the nearest entry of a table of ``2**bits`` levels.

    Attributes:
        mode: ``"gaussian"``: every token vector is divided by its L2 norm and scaled
            by sqrt(head_dim), so its coordinates are close to N(0, 1) (especially
            after a random rotation, as in TurboQuant), then snapped to the
            Lloyd-Max levels of N(0, 1); the norm is stored in ``scale_dtype``.
            ``"calibrated"``: one table per channel, layer and keys/values, learned
            by 1-D k-means from samples in ``calibrate()`` (KVQuant-style); nothing
            is stored per token besides the codes.
        codebook_path: Load calibrated tables from this file (``save`` writes it).
        calibration_iters: k-means iterations in ``calibrate()``.

    ``bits`` must be 1, 2, 4 or 8. ``group_size``, ``symmetric`` and
    ``offset_dtype`` are unused. ``"gaussian"`` needs PER_TOKEN granularity.
    Prepare all calibrated tables before encoding. The first successful encode
    freezes them for that quantizer's lifetime, including FAKE encoding. Use a
    fresh quantizer/cache to change tables; ``tables`` exposes inspection copies.
    Successfully decoding a matching calibrated record freezes the receiver too.
    """

    bits: int = 2
    mode: str = "gaussian"
    codebook_path: Optional[str] = None
    calibration_iters: int = 50


@lru_cache(maxsize=None)
def lloyd_max_gaussian(bits: int, iterations: int = 300) -> torch.Tensor:
    """Levels that minimise the mean squared error of quantizing N(0, 1), ascending."""
    count = 2**bits
    normal = torch.distributions.Normal(0.0, 1.0)
    probs = (torch.arange(count, dtype=torch.float64) + 0.5) / count
    levels = normal.icdf(probs)
    for _ in range(iterations):
        edges = torch.cat(
            [
                torch.tensor([-math.inf], dtype=torch.float64),
                (levels[1:] + levels[:-1]) / 2,
                torch.tensor([math.inf], dtype=torch.float64),
            ]
        )
        density = torch.exp(-(edges**2) / 2) / math.sqrt(2 * math.pi)
        density = torch.nan_to_num(density, nan=0.0)
        mass = normal.cdf(edges[1:]) - normal.cdf(edges[:-1])
        levels = (density[:-1] - density[1:]) / mass  # E[X | edge_i < X < edge_i+1]
    return levels.float()


def nearest_level(values: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Index of the nearest of the ascending ``levels`` (last axis), broadcasting."""
    midpoints = (levels[..., 1:] + levels[..., :-1]) / 2
    if midpoints.ndim == 1:
        return torch.bucketize(values.contiguous(), midpoints)
    # One table per channel: compare against each channel's midpoints.
    return (values[..., None] > midpoints).sum(dim=-1)


@KVQuantizerRegistry.register(NonUniformQuantizerConfig)
class NonUniformQuantizer(KVQuantizer):
    config: NonUniformQuantizerConfig

    def __init__(self, config: NonUniformQuantizerConfig) -> None:
        super().__init__(config)
        if config.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if type(config.bits) is not int or config.bits not in (1, 2, 4, 8):
            raise ValueError("NonUniformQuantizer supports 1, 2, 4 or 8 bits")
        if type(config.calibration_iters) is not int or config.calibration_iters < 1:
            raise ValueError("calibration_iters must be a positive integer")
        if (
            config.mode == "gaussian"
            and QuantGranularity(config.granularity) != QuantGranularity.PER_TOKEN
        ):
            raise ValueError("gaussian mode normalises whole tokens; use PER_TOKEN")
        self._tables: Dict[TableKey, torch.Tensor] = {}
        self._table_fingerprints: Dict[TableKey, str] = {}
        self._tables_frozen = False
        self._tables_lock = RLock()
        if config.codebook_path is not None:
            self.load(config.codebook_path)

    @property
    def tables(self) -> Mapping[TableKey, torch.Tensor]:
        """Read-only inspection snapshot; tensor edits cannot change live tables."""
        with self._tables_lock:
            return MappingProxyType(
                {key: table.clone() for key, table in self._tables.items()}
            )

    @property
    def tables_frozen(self) -> bool:
        """Whether calibrated tables have been used for encoding or decoding."""
        return self._tables_frozen

    def __getstate__(self) -> Dict[str, Any]:
        """Preserve table/freeze state when sending a codec to a worker."""
        with self._tables_lock:
            state = self.__dict__.copy()
            state.pop("_tables_lock")
            state["_tables"] = self._tables.copy()
            state["_table_fingerprints"] = self._table_fingerprints.copy()
            return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._tables_lock = RLock()

    # ------------------------------------------------------------------ encoding
    def quantize(
        self,
        x: torch.Tensor,
        plan: Optional[PrecisionPlan] = None,
        *,
        context: Optional[AttentionContext] = None,
    ) -> QuantizedTensor:
        check_input(x, plan, allow_keep_tokens=True)
        if plan is not None and plan.bits != self.config.bits:
            raise ValueError(
                f"NonUniform precision plan must use {self.config.bits} bits"
            )
        if plan is not None and plan.keep_tokens is not None:
            # Whole kept vectors do not need norms or codes derived from their
            # original magnitude. Their exact originals belong to the coordinator.
            x = x.masked_fill(plan.keep_tokens[..., None], 0.0)
        if self.config.mode == "gaussian":
            return self._quantize_gaussian(x)
        with self._tables_lock:
            return self._quantize_calibrated(x, context)

    def _quantize_gaussian(self, x: torch.Tensor) -> QuantizedTensor:
        dim = x.shape[-1]
        data = x.detach().float()
        exact = data.norm(dim=-1, keepdim=True)
        # Rounded up to the storage dtype; a zero vector keeps a zero norm.
        norms = torch.where(
            exact > 0,
            store_param(exact, self.config.scale_dtype),
            torch.zeros_like(exact),
        )
        unit = (
            data
            / torch.where(norms > 0, norms, torch.ones_like(norms))
            * math.sqrt(dim)
        )
        levels = lloyd_max_gaussian(self.config.bits).to(x.device)
        packed, count = pack_codes(nearest_level(unit, levels), self.config.bits)
        return self._record(x, packed, count, norms, mode="gaussian", key=None)

    def _quantize_calibrated(
        self, x: torch.Tensor, context: Optional[AttentionContext]
    ) -> QuantizedTensor:
        if context is None:
            raise ValueError(
                "calibrated mode needs the attention context (layer, kind)"
            )
        key = (context.layer_idx, context.tensor_kind)
        key = self._validate_table_key(key)
        if key not in self._tables:
            raise RuntimeError(
                f"NonUniformQuantizer has no table for layer {key[0]} {key[1]}s; "
                "run calibrate() or set codebook_path for all needed layers/kinds "
                "before encoding"
            )
        table = self._tables[key].to(x.device)  # [channels, levels]
        if table.shape[0] != x.shape[-1]:
            raise ValueError("Calibrated table does not match head_dim")
        codes = nearest_level(x.detach().float(), table)
        packed, count = pack_codes(codes, self.config.bits)
        encoded = self._record(x, packed, count, None, mode="calibrated", key=key)
        encoded.metadata["table_fingerprint"] = self._table_fingerprints[key]
        self._tables_frozen = True
        return encoded

    def _record(
        self,
        x: torch.Tensor,
        packed: torch.Tensor,
        count: int,
        norms: Optional[torch.Tensor],
        *,
        mode: str,
        key: Optional[TableKey],
    ) -> QuantizedTensor:
        return QuantizedTensor(
            codes=packed,
            scales=torch.empty(0, device=x.device),
            zero_points=None,
            shape=tuple(x.shape),
            dtype=x.dtype,
            device=x.device,
            axis=3,
            bits=self.config.bits,
            format_id=f"nonuniform_{mode}_v1",
            group_size=x.shape[-1],
            metadata={"code_count": count, "table_key": key},
            norms=(
                None
                if norms is None
                else norms.to(self.config.scale_dtype.torch_dtype).contiguous()
            ),
        )

    def dequantize(self, q: QuantizedTensor) -> torch.Tensor:
        codes = unpack_codes(q.codes, q.bits, q.metadata["code_count"]).long()
        codes = codes.reshape(q.shape)
        if q.format_id == "nonuniform_gaussian_v1":
            assert q.norms is not None
            levels = lloyd_max_gaussian(q.bits).to(q.codes.device)
            restored = levels[codes] / math.sqrt(q.shape[-1]) * q.norms.float()
        elif q.format_id == "nonuniform_calibrated_v1":
            with self._tables_lock:
                key = self._validate_table_key(q.metadata["table_key"])
                if (
                    q.bits != self.config.bits
                    or key not in self._tables
                    or q.metadata.get("table_fingerprint")
                    != self._table_fingerprints[key]
                ):
                    raise ValueError(
                        "Encoded data does not match this calibrated table"
                    )
                table = self._tables[key].to(q.codes.device)
                if table.shape[0] != q.shape[-1]:
                    raise ValueError("Calibrated table does not match encoded head_dim")
                channel = torch.arange(q.shape[-1], device=q.codes.device)
                restored = table[channel, codes]
                # A matching record also makes a fresh receiver depend on its
                # prepared table, even before it encodes anything of its own.
                self._tables_frozen = True
        else:
            raise ValueError(f"Unsupported non-uniform encoding: {q.format_id}")
        return finite_cast(restored, q.dtype, device=q.device)

    # --------------------------------------------------------------- calibration
    @staticmethod
    def _validate_table_key(key: object) -> TableKey:
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(
                "Table key must be (non-negative layer_idx, 'key' or 'value')"
            )
        layer, kind = key
        if (
            type(layer) is not int
            or layer < 0
            or not isinstance(kind, str)
            or kind not in ("key", "value")
        ):
            raise ValueError(
                "Table key must be (non-negative layer_idx, 'key' or 'value')"
            )
        return layer, kind

    def _check_tables_mutable(self) -> None:
        if self.config.mode != "calibrated":
            raise ValueError("Table preparation is for mode='calibrated'")
        if self._tables_frozen:
            raise RuntimeError(
                "Calibrated tables are frozen after encoding/decoding; "
                "create a fresh quantizer and cache to use different tables"
            )

    def _validate_table(self, table: object) -> torch.Tensor:
        if (
            not isinstance(table, torch.Tensor)
            or table.layout != torch.strided
            or not table.is_floating_point()
            or table.ndim != 2
            or table.shape[0] == 0
            or table.shape[1] != 2**self.config.bits
        ):
            raise ValueError(
                "Each table must be a dense floating [channels, 2**bits] tensor "
                "with at least one channel"
            )
        owned = (
            table.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        )
        if not torch.isfinite(owned).all():
            raise ValueError("Table levels must be finite in FP32")
        if (owned[:, 1:] < owned[:, :-1]).any():
            raise ValueError("Table levels must be sorted in nondecreasing order")
        return owned

    def _install_tables(self, tables: Dict[TableKey, torch.Tensor]) -> None:
        """Commit a complete validated update, with no externally owned aliases."""
        self._check_tables_mutable()
        fingerprints = {
            key: hashlib.sha256(
                str(tuple(table.shape)).encode() + table.numpy().tobytes()
            ).hexdigest()
            for key, table in tables.items()
        }
        self._tables.update(tables)
        self._table_fingerprints.update(fingerprints)

    def calibrate(self, samples: Mapping[TableKey, torch.Tensor]) -> None:
        """Learn per-channel tables for each ``(layer_idx, "key" | "value")``.

        Each value holds sample vectors with head_dim as the last axis, e.g. keys
        collected from calibration prompts. Only used in ``"calibrated"`` mode.
        All required layers/kinds must be prepared before the first encoding.
        Invalid sample batches leave previously prepared tables unchanged.
        """
        with self._tables_lock:
            self._calibrate(samples)

    def _calibrate(self, samples: Mapping[TableKey, torch.Tensor]) -> None:
        self._check_tables_mutable()
        if not isinstance(samples, Mapping) or not samples:
            raise ValueError("Calibration samples must be a nonempty mapping")
        prepared = {}
        for key, tensor in samples.items():
            key = self._validate_table_key(key)
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.layout != torch.strided
                or not tensor.is_floating_point()
                or tensor.ndim < 1
                or tensor.numel() == 0
                or tensor.shape[-1] == 0
            ):
                raise ValueError(
                    "Calibration samples must be nonempty floating vectors"
                )
            rows = tensor.detach().float().reshape(-1, tensor.shape[-1])
            if not torch.isfinite(rows).all():
                raise ValueError("Calibration samples must be finite in FP32")
            prepared[key] = self._validate_table(
                _kmeans_per_channel(
                    rows, 2**self.config.bits, self.config.calibration_iters
                )
            )
        self._install_tables(prepared)

    def save(self, path: str) -> None:
        """Save prepared tables before or after freezing (not live cache records)."""
        if self.config.mode != "calibrated" or not self._tables:
            raise ValueError("save() needs prepared calibrated tables")
        torch.save(
            {
                "bits": self.config.bits,
                "tables": {
                    f"{layer}:{kind}": t for (layer, kind), t in self.tables.items()
                },
            },
            path,
        )

    def load(self, path: str) -> None:
        """Validate all file contents before merging tables into an unused codec."""
        with self._tables_lock:
            self._load(path)

    def _load(self, path: str) -> None:
        self._check_tables_mutable()
        stored = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(stored, dict) or type(stored.get("bits")) is not int:
            raise ValueError(
                "Table file must contain integer bits and a tables mapping"
            )
        if stored["bits"] != self.config.bits:
            raise ValueError(
                f"Tables in {path} are {stored['bits']}-bit, not {self.config.bits}-bit"
            )
        tables = stored.get("tables")
        if not isinstance(tables, Mapping) or not tables:
            raise ValueError("Table file must contain a nonempty tables mapping")
        prepared = {}
        for name, table in tables.items():
            if not isinstance(name, str):
                raise ValueError(
                    "Table names must use 'layer_idx:key' or 'layer_idx:value'"
                )
            parts = name.split(":")
            if len(parts) != 2 or not parts[0].isdecimal():
                raise ValueError(
                    "Table names must use 'layer_idx:key' or 'layer_idx:value'"
                )
            key = self._validate_table_key((int(parts[0]), parts[1]))
            if name != f"{key[0]}:{key[1]}":
                raise ValueError(
                    "Table names must use canonical non-negative layer indices"
                )
            prepared[key] = self._validate_table(table)
        self._install_tables(prepared)


def _kmeans_per_channel(
    rows: torch.Tensor, levels: int, iterations: int
) -> torch.Tensor:
    """1-D k-means for every channel at once: ``rows`` [samples, channels] ->
    ascending centroids [channels, levels]. Starts at evenly spaced quantiles."""
    quantiles = (torch.arange(levels, dtype=torch.float32) + 0.5) / levels
    centroids = torch.quantile(rows, quantiles.to(rows.device), dim=0).T.contiguous()
    for _ in range(iterations):
        assign = nearest_level(rows, centroids)  # [samples, channels]
        one_hot = torch.nn.functional.one_hot(assign, levels).float()
        counts = one_hot.sum(dim=0)  # [channels, levels]
        sums = (one_hot * rows[..., None]).sum(dim=0)
        updated = sums / counts.clamp(min=1)
        centroids = torch.where(counts > 0, updated, centroids).sort(dim=-1).values
    return centroids
