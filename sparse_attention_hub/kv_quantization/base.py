"""KV quantization coordinator and its configuration."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, cast

import torch

from sparse_attention_hub.metric_logging.logger import MicroMetricLogger

from .cache.block import StoredChunk
from .cache.store import CACHE_KEY, Encoder, QuantizedKVStore
from .context import AttentionContext, record_trace
from .metrics.quant_error import QuantErrorMetric
from .policies.cache.base import CachePolicy, CachePolicyConfig
from .policies.cache.implementations.default import DefaultCachePolicyConfig
from .policies.precision.base import PrecisionPolicy, PrecisionPolicyConfig
from .policies.precision.implementations.fixed import FixedPrecisionConfig
from .quantizers.base import KVQuantizer, KVQuantizerConfig
from .quantizers.implementations.integer import IntQuantizerConfig
from .registry import (
    CachePolicyRegistry,
    KVQuantizationRegistry,
    PrecisionPolicyRegistry,
)
from .transforms.base import KVTransform, KVTransformConfig
from .transforms.implementations.identity import IdentityTransformConfig


@dataclass(frozen=True)
class KVQuantizationConfig:
    key_transform: KVTransformConfig = field(default_factory=IdentityTransformConfig)
    value_transform: KVTransformConfig = field(default_factory=IdentityTransformConfig)
    key_quantizer: KVQuantizerConfig = field(default_factory=IntQuantizerConfig)
    value_quantizer: KVQuantizerConfig = field(default_factory=IntQuantizerConfig)
    precision_policy: PrecisionPolicyConfig = field(
        default_factory=FixedPrecisionConfig
    )
    cache_policy: CachePolicyConfig = field(default_factory=DefaultCachePolicyConfig)
    store_codes: bool = False
    error_metric: bool = False
    quantize_prefill: bool = False


@KVQuantizationRegistry.register(KVQuantizationConfig)
class KVQuantization:
    def __init__(self, config: KVQuantizationConfig) -> None:
        self.config = config
        # Construction fails immediately for unimplemented components.
        self.key_transform = KVTransform.create_from_config(config.key_transform)
        self.value_transform = KVTransform.create_from_config(config.value_transform)
        self.key_quantizer = KVQuantizer.create_from_config(config.key_quantizer)
        self.value_quantizer = KVQuantizer.create_from_config(config.value_quantizer)
        self.precision_policy = cast(
            PrecisionPolicy, PrecisionPolicyRegistry.create(config.precision_policy)
        )
        self.cache_policy = cast(
            CachePolicy, CachePolicyRegistry.create(config.cache_policy)
        )
        self.precision_policy.validate_quantizers(
            config.key_quantizer, config.value_quantizer
        )
        self.cache_policy.validate_quantizers(
            config.key_quantizer, config.value_quantizer
        )
        # Optional quantizers for the sink and local regions; None keeps them in FP.
        self.region_quantizers: Dict[Tuple[str, str], KVQuantizer] = {}
        for kind in ("key", "value"):
            for region in ("sink", "local"):
                region_config = (
                    config.cache_policy.sink(kind)
                    if region == "sink"
                    else config.cache_policy.local(kind)
                )
                if region_config.quantizer is not None:
                    self.region_quantizers[
                        (kind, region)
                    ] = KVQuantizer.create_from_config(region_config.quantizer)
        self.error_metric = QuantErrorMetric() if config.error_metric else None
        self._owner = object()

    @classmethod
    def create_from_config(cls, config: KVQuantizationConfig) -> "KVQuantization":
        return cast(KVQuantization, KVQuantizationRegistry.create(config))

    def process(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        meta_data: Dict[str, Any],
        *,
        num_new_tokens: int,
        phase: str = "decode",
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare attention from old stored history plus exact current-call K/V.

        ``num_new_tokens`` is the query sequence length, never the difference
        between HF cache lengths. Zero imports/inspects completed history for
        offline codec experiments. Prefill is captured even when attention itself
        stays full precision; storage updates cannot alter this call's snapshot.
        """
        return self._process_kv_history(
            layer_idx,
            keys,
            values,
            meta_data,
            num_new_tokens=num_new_tokens,
            phase=phase,
            position_ids=position_ids,
            reconstruct_current=False,
        )

    def _process_kv_history(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        meta_data: Dict[str, Any],
        *,
        num_new_tokens: int,
        reconstruct_current: bool,
        phase: str = "decode",
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Import, compress and reconstruct the request's independent KV history.

        KIVI uses exact current-call tokens; reconstruct_current instead returns
        the stored reconstruction after compressing those arriving tokens too.
        """
        if (
            not isinstance(layer_idx, int)
            or isinstance(layer_idx, bool)
            or layer_idx < 0
        ):
            raise ValueError(
                "KV quantization requires a non-negative integer layer_idx"
            )
        if phase not in ("prefill", "decode"):
            raise ValueError("kv_quantization_phase must be 'prefill' or 'decode'")
        store = meta_data.get(CACHE_KEY)
        if store is None:
            cache_policy = self.config.cache_policy
            store = QuantizedKVStore(
                self._owner,
                store_codes=self.config.store_codes,
                sink_tokens={
                    "key": cache_policy.key_sink.tokens,
                    "value": cache_policy.value_sink.tokens,
                },
            )
            meta_data[CACHE_KEY] = store
            record_trace(meta_data, "QuantizedKVStore: create")
        if not isinstance(store, QuantizedKVStore) or store.owner is not self._owner:
            raise ValueError(
                "Metadata belongs to another quantization configuration; use a fresh request"
            )

        store.validate_input(layer_idx, keys, values)
        if (
            not isinstance(num_new_tokens, int)
            or isinstance(num_new_tokens, bool)
            or not 0 <= num_new_tokens <= keys.shape[-2]
        ):
            raise ValueError(
                "num_new_tokens must be an integer in [0, incoming KV length]"
            )
        if keys.shape[-2] == 0:
            raise ValueError("Cannot prepare attention from empty KV history")

        encode = self._encoder(layer_idx, meta_data, phase, position_ids)
        seen = store.seen_tokens(layer_idx)
        past_in_input = int(keys.shape[-2]) - num_new_tokens
        # A first full-history callback may arrive after native prefill. Seed that
        # completed context before taking the snapshot. Sliding adapters capture
        # prefill themselves; evicted context cannot be recovered from a suffix.
        if seen == 0 or num_new_tokens == 0:
            if past_in_input < seen:
                raise ValueError(
                    "KV history shrank; start a fresh request for rollback"
                )
            if past_in_input > seen:
                store.append(
                    layer_idx,
                    keys[..., seen:past_in_input, :],
                    values[..., seen:past_in_input, :],
                    encode,
                )
                self._compress(
                    store, layer_idx, meta_data, "before_attention", phase, position_ids
                )
        elif past_in_input > seen:
            raise ValueError(
                "Missing cached history; capture prefill or start a fresh request"
            )

        new_keys = keys[..., past_in_input:, :]
        new_values = values[..., past_in_input:, :]
        attention_keys, attention_values = keys, values
        if not reconstruct_current:
            if store.seen_tokens(layer_idx):
                previous_keys, previous_values = store.get(
                    layer_idx, self._decode_chunk
                )
            else:
                previous_keys, previous_values = keys[..., :0, :], values[..., :0, :]
            attention_keys = torch.cat([previous_keys, new_keys], dim=-2)
            attention_values = torch.cat([previous_values, new_values], dim=-2)
            # The snapshot owns its data before compression can change the store.
        imported = store.append(layer_idx, new_keys, new_values, encode)
        record_trace(
            meta_data,
            "QuantizedKVStore.append: unseen suffix",
            layer=layer_idx,
            tokens=imported,
        )

        self._compress(
            store,
            layer_idx,
            meta_data,
            "before_attention",
            phase,
            position_ids,
        )
        if reconstruct_current:
            attention_keys, attention_values = store.get(layer_idx, self._decode_chunk)
        record_trace(
            meta_data,
            (
                "KVQuantization.process: reconstructed history including new KV"
                if reconstruct_current
                else "KVQuantization.process: previous history + original new KV"
            ),
            layer=layer_idx,
            new_tokens=num_new_tokens,
        )
        if self.error_metric is not None:
            logger = MicroMetricLogger()
            kinds = [
                kind
                for kind in ("key", "value")
                if logger.is_metric_enabled(f"kv_quantization_{kind}_error")
            ]
            if kinds:
                state = store.layers[layer_idx]
                stored_keys, stored_values = store.get(layer_idx, self._decode_chunk)
                reference_length = int(keys.shape[-2])
                reference_start = state.seen_tokens - reference_length
                for kind, original, reconstructed in (
                    ("key", keys, stored_keys),
                    ("value", values, stored_values),
                ):
                    if kind in kinds:
                        self.error_metric.log(
                            kind,
                            original,
                            reconstructed[..., -reference_length:, :],
                            getattr(state, kind),
                            layer_idx,
                            start_token=reference_start,
                        )
            if logger.is_metric_enabled("kv_quantization_store_bytes"):
                logger.log(
                    "kv_quantization_store_bytes",
                    store.stats()["store_bytes"],
                    metadata={"layer_idx": layer_idx},
                )
        if phase == "prefill" and not self.config.quantize_prefill:
            return keys, values
        return attention_keys, attention_values

    def _quantizer_for(self, kind: str, region: str) -> Optional[KVQuantizer]:
        """Middle uses the main quantizer; sink/local use theirs, or None for FP."""
        if region == "middle":
            return self.key_quantizer if kind == "key" else self.value_quantizer
        return self.region_quantizers.get((kind, region))

    def _encoder(
        self,
        layer_idx: int,
        metadata: Dict[str, Any],
        phase: str,
        position_ids: Optional[torch.Tensor],
    ) -> Encoder:
        def encode(
            kind: str, region: str, tensor: torch.Tensor, start: int
        ) -> StoredChunk:
            return self._encode_chunk(
                kind, region, tensor, start, layer_idx, metadata, phase, position_ids
            )

        return encode

    def _encode_chunk(
        self,
        kind: str,
        region: str,
        tensor: torch.Tensor,
        start: int,
        layer_idx: int,
        metadata: Dict[str, Any],
        phase: str,
        position_ids: Optional[torch.Tensor],
    ) -> StoredChunk:
        token_count = int(tensor.shape[-2])
        quantizer = self._quantizer_for(kind, region)
        if quantizer is None:
            return StoredChunk(
                start,
                token_count,
                region=region,
                full_precision=tensor.detach().clone(),
            )
        context = AttentionContext(layer_idx, kind, start, phase, position_ids)
        transform = self.key_transform if kind == "key" else self.value_transform
        record_trace(
            metadata,
            "KVTransform.forward",
            layer=layer_idx,
            kind=kind,
            region=region,
            start=start,
        )
        transformed, saved = transform.forward(tensor, context)
        if region == "middle":
            positions = torch.arange(start, start + token_count, device=tensor.device)
            plan = self.precision_policy.bits_for(
                layer_idx,
                transformed,
                positions,
                tensor_kind=kind,
                default_bits=quantizer.config.bits,
            )
        else:
            plan = None  # Sink and local regions use their quantizer's own bits.
        record_trace(
            metadata,
            "PrecisionPolicy.bits_for -> KVQuantizer.quantize",
            layer=layer_idx,
            kind=kind,
            region=region,
            bits=quantizer.config.bits if plan is None else plan.bits,
        )
        chunk = StoredChunk(
            start, token_count, context, transform_state=saved, region=region
        )
        if plan is not None and plan.keep_tokens is not None:
            # The coordinator owns exact originals; the codec owns grouping and
            # excludes these positions from its numerical statistics. This is the
            # optional generic keep-token contract, not a paper-specific rule.
            keep = plan.keep_tokens.to(tensor.device)
            chunk.kept_index = keep.flatten().nonzero().flatten().to(torch.int32)
            chunk.kept_values = (
                tensor.detach()
                .reshape(-1, tensor.shape[-1])[chunk.kept_index.long()]
                .clone()
            )
        encoded = quantizer.quantize(transformed, plan, context=context)
        if self.config.store_codes:
            chunk.quantized = encoded
        else:
            chunk.dequantized = (
                restore_kept_tokens(
                    transform.inverse(quantizer.dequantize(encoded), context, saved),
                    chunk,
                )
                .detach()
                .clone()
            )
            # FAKE storage already contains the exact kept originals. Only packed
            # chunks need separate originals and indices for future decoding.
            chunk.kept_index = None
            chunk.kept_values = None
        return chunk

    def _decode_chunk(self, kind: str, chunk: StoredChunk) -> torch.Tensor:
        if chunk.full_precision is not None:
            return chunk.full_precision
        if chunk.dequantized is not None:
            return chunk.dequantized
        quantizer = self._quantizer_for(kind, chunk.region)
        transform = self.key_transform if kind == "key" else self.value_transform
        assert (
            chunk.quantized is not None
            and quantizer is not None
            and chunk.context is not None
        )
        return restore_kept_tokens(
            transform.inverse(
                quantizer.dequantize(chunk.quantized),
                chunk.context,
                chunk.transform_state,
            ),
            chunk,
        )

    def _compress(
        self,
        store: QuantizedKVStore,
        layer_idx: int,
        metadata: Dict[str, Any],
        timing: str,
        phase: str,
        position_ids: Optional[torch.Tensor],
    ) -> None:
        store.compress_ready(
            layer_idx,
            self.cache_policy,
            self._encoder(layer_idx, metadata, phase, position_ids),
            self._decode_chunk,
            phase=timing,
        )

    def after_attention(
        self,
        layer_idx: int,
        meta_data: Dict[str, Any],
        *,
        phase: str = "decode",
        position_ids: Optional[torch.Tensor] = None,
    ) -> None:
        store = meta_data.get(CACHE_KEY)
        if store is not None:
            self._compress(
                store,
                layer_idx,
                meta_data,
                "after_attention",
                phase,
                position_ids,
            )
            self.cache_policy.after_attention(store, layer_idx)

    def clear_request(self, meta_data: Dict[str, Any]) -> None:
        store = meta_data.pop(CACHE_KEY, None)
        if store is not None:
            store.clear()

    def calibrate(self, kv_samples: Any) -> None:
        """TODO: dispatch method-specific calibration and persist its artifacts."""
        raise NotImplementedError("Calibration recipes are scaffolded, not implemented")


def restore_kept_tokens(x: torch.Tensor, chunk: StoredChunk) -> torch.Tensor:
    """Put the original values of kept tokens back into decoded ``x``."""
    if chunk.kept_index is None or chunk.kept_values is None:
        return x
    restored = x.clone()
    restored.view(-1, x.shape[-1])[chunk.kept_index.long()] = chunk.kept_values.to(
        x.dtype
    )
    return restored


class PendingRecipe(KVQuantization):
    def __init__(self, config: KVQuantizationConfig) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} is a recipe scaffold. Implement and validate its "
            "numerics, calibration, storage and cache policy before selecting it. "
            "Use KVQuantizationConfig with IntQuantizerConfig for the working baseline."
        )
