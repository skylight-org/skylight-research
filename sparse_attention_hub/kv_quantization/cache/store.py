"""Own and reconstruct request-local KV histories."""

from typing import Any, Callable, Dict, List, Tuple

import torch

from ..numerics import require_finite
from ..policies.cache.base import CachePolicy
from .block import StoredChunk
from .state import LayerKVState, TensorCacheState

CACHE_KEY = "kv_quantization_store"
PENDING_LOCAL = "pending_local"

# encode(kind, region, tensor, start_token) -> StoredChunk; region is sink/middle/local.
Encoder = Callable[[str, str, torch.Tensor, int], StoredChunk]
# decode(kind, chunk) -> tensor in original coordinates, dtype and device.
Decoder = Callable[[str, StoredChunk], torch.Tensor]


class QuantizedKVStore:
    def __init__(
        self, owner: object, *, store_codes: bool, sink_tokens: Dict[str, int]
    ) -> None:
        self.owner = owner
        self.store_codes = store_codes
        self.sink_tokens = sink_tokens
        self.layers: Dict[int, LayerKVState] = {}

    def seen_tokens(self, layer_idx: int) -> int:
        state = self.layers.get(layer_idx)
        return 0 if state is None else state.seen_tokens

    @staticmethod
    def _schema(x: torch.Tensor) -> Tuple[Any, ...]:
        return x.shape[0], x.shape[1], x.shape[3], x.dtype, x.device

    def append(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        encode: Encoder,
    ) -> int:
        """Append only new tokens, independently of HF's current cache length.

        The first configured tokens go to the sink (encoded now, since sinks never
        move). The rest join the local window as pending full-precision chunks;
        ``compress_ready`` moves old ones to the middle and encodes the remainder.
        """
        self.validate_input(layer_idx, keys, values)
        new_count = int(keys.shape[-2])
        if new_count == 0:
            return 0
        # Check newly captured tokens even when policy keeps them uncompressed.
        require_finite(keys, name="New keys")
        require_finite(values, name="New values")
        state = self.layers.setdefault(layer_idx, LayerKVState())
        state.key_schema, state.value_schema = self._schema(keys), self._schema(values)
        for kind, new in (("key", keys.detach()), ("value", values.detach())):
            target: TensorCacheState = getattr(state, kind)
            sink_count = min(
                new_count, max(0, self.sink_tokens[kind] - target.sink_tokens)
            )
            if sink_count:
                target.sink.append(
                    encode(kind, "sink", new[..., :sink_count, :], state.seen_tokens)
                )
            rest = new[..., sink_count:, :]
            if rest.shape[-2]:
                target.recent.append(
                    StoredChunk(
                        state.seen_tokens + sink_count,
                        int(rest.shape[-2]),
                        region=PENDING_LOCAL,
                        full_precision=rest.clone(),
                    )
                )
        state.seen_tokens += new_count
        return new_count

    def validate_input(
        self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor
    ) -> None:
        """Check shapes and the layer's schema before reading or changing state."""
        if keys.ndim != 4 or values.ndim != 4 or keys.shape[:3] != values.shape[:3]:
            raise ValueError(
                "Expected matching [batch, kv_heads, tokens, channels] K/V"
            )
        if not keys.is_floating_point() or not values.is_floating_point():
            raise ValueError("Native K/V inputs must be floating point")
        state = self.layers.get(layer_idx)
        schemas = (self._schema(keys), self._schema(values))
        if (
            state is not None
            and state.key_schema is not None
            and schemas
            != (
                state.key_schema,
                state.value_schema,
            )
        ):
            raise ValueError(
                "Batch/head/channel shape, dtype or device changed; start a fresh store"
            )

    def compress_ready(
        self,
        layer_idx: int,
        policy: CachePolicy,
        encode: Encoder,
        decode: Decoder,
        *,
        phase: str = "before_attention",
    ) -> None:
        state = self.layers[layer_idx]
        for kind in ("key", "value"):
            target: TensorCacheState = getattr(state, kind)
            while True:
                count = policy.next_chunk_tokens(kind, target, phase=phase)
                if not count:
                    break
                if count < 0 or count > target.recent_tokens:
                    raise ValueError("Cache policy returned an invalid chunk length")
                start = target.recent[0].start_token
                moved = self._take_local(kind, target, count, encode, decode)
                page = policy.page_tokens(kind) or count
                for offset in range(0, count, page):
                    target.chunks.append(
                        encode(
                            kind,
                            "middle",
                            moved[..., offset : offset + page, :],
                            start + offset,
                        )
                    )
            # Tokens still in the local window are stored with the local quantizer.
            target.recent = [
                encode(kind, "local", chunk.full_precision, chunk.start_token)
                if chunk.region == PENDING_LOCAL and chunk.full_precision is not None
                else chunk
                for chunk in target.recent
            ]

    @staticmethod
    def _take_local(
        kind: str,
        target: TensorCacheState,
        count: int,
        encode: Encoder,
        decode: Decoder,
    ) -> torch.Tensor:
        """Remove the oldest ``count`` local tokens and return their values.

        Quantized local tokens are returned as decoded from their stored local copy,
        so moving them to the middle re-quantizes that copy (not the original).
        """
        taken: List[torch.Tensor] = []
        remaining = count
        while remaining:
            chunk = target.recent[0]
            values = decode(kind, chunk)
            if chunk.token_count <= remaining:
                taken.append(values)
                target.recent.pop(0)
                remaining -= chunk.token_count
                continue
            taken.append(values[..., :remaining, :])
            rest = values[..., remaining:, :].clone()
            rest_start = chunk.start_token + remaining
            if chunk.is_encoded:
                target.recent[0] = encode(kind, "local", rest, rest_start)
            else:
                target.recent[0] = StoredChunk(
                    rest_start,
                    int(rest.shape[-2]),
                    region=chunk.region,
                    full_precision=rest,
                )
            remaining = 0
        return torch.cat(taken, dim=-2)

    def get(self, layer_idx: int, decode: Decoder) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct temporary K/V, preserving original order and KV-head layout."""
        outputs = []
        layer = self.layers[layer_idx]
        for kind in ("key", "value"):
            state: TensorCacheState = getattr(layer, kind)
            parts = [decode(kind, chunk) for chunk in state.all_chunks()]
            if not parts:
                raise ValueError("Cannot read an empty KV history")
            full = torch.cat(parts, dim=-2)
            if full.shape[-2] != layer.seen_tokens:
                raise RuntimeError(
                    "Stored KV regions do not cover the complete token history"
                )
            outputs.append(full)
        return outputs[0], outputs[1]

    @staticmethod
    def _chunk_bytes(chunk: StoredChunk) -> int:
        for tensor in (chunk.full_precision, chunk.dequantized):
            if tensor is not None:  # already holds every token, kept ones included
                return int(tensor.numel() * tensor.element_size())
        if chunk.quantized is None:
            return 0
        kept = sum(
            t.numel() * t.element_size()
            for t in (chunk.kept_index, chunk.kept_values)
            if t is not None
        )
        return int(chunk.quantized.storage_bytes() + kept)

    @staticmethod
    def _shared_outlier_bytes(chunks: Any, seen: Dict[Any, int]) -> int:
        """Bytes of tensors that several records share (a frozen TurboQuant
        channel choice in ``outlier_indices``, a frozen channel mean in
        ``zero_points``): subtract them per record, add each distinct storage
        once through ``seen``."""
        repeated = 0
        for chunk in chunks:
            record = chunk.quantized
            if record is None or not record.format_id.startswith("turboquant_"):
                continue
            shared = []
            if record.outlier_indices is not None and record.format_id.endswith(
                "_outlier_v1"
            ):
                shared.append(record.outlier_indices)
            if record.zero_points is not None and record.metadata.get("centered"):
                shared.append(record.zero_points)
            for tensor in shared:
                repeated += int(tensor.numel() * tensor.element_size())
                storage = tensor.untyped_storage()
                seen[(tensor.device, storage.data_ptr())] = storage.nbytes()
        return repeated

    def stats(self) -> Dict[str, Any]:
        owned_bytes = 0
        shared: Dict[Any, int] = {}
        layers = {}
        for index, layer in self.layers.items():
            info = {"seen_tokens": layer.seen_tokens}
            for kind in ("key", "value"):
                state: TensorCacheState = getattr(layer, kind)
                info[f"{kind}_sink_tokens"] = state.sink_tokens
                info[f"{kind}_compressed_tokens"] = state.compressed_tokens
                info[f"{kind}_recent_tokens"] = state.recent_tokens
                info[f"{kind}_encoded_tokens"] = sum(
                    end - start for start, end in state.encoded_ranges()
                )
                chunks = list(state.all_chunks())
                owned_bytes += sum(self._chunk_bytes(chunk) for chunk in chunks)
                owned_bytes -= self._shared_outlier_bytes(chunks, shared)
            layers[index] = info
        owned_bytes += sum(shared.values())
        return {"store_bytes": owned_bytes, "layers": layers}

    def clear(self) -> None:
        self.layers.clear()
