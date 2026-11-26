"""ChannelBoost: give the most important key channels more bits (Kitty)."""

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from ....quantizers.base import KVQuantizerConfig, QuantGranularity
from ....registry import PrecisionPolicyRegistry
from ..base import PrecisionPlan, PrecisionPolicy, PrecisionPolicyConfig

KINDS = ("key", "value")
SELECTIONS = ("magnitude", "variance", "random")
COUNT_ROUNDINGS = ("round", "floor")
SCORE_DTYPES = ("fp32", "input")
FREEZES = ("none", "prefill")


@dataclass(frozen=True)
class ChannelBoostConfig(PrecisionPolicyConfig):
    """Per chunk and per head, the ``fraction`` of channels with the highest score
    get ``boosted_bits``; the others keep the quantizer's bits.

    Attributes:
        fraction: Share of channels boosted (Kitty: 12.5%; Kitty-Pro: 25%).
        boosted_bits: Bits of the boosted channels (2, 3, 4 or 8). The quantizer
            still checks that it can store the width (3 is for TurboQuant).
        kinds: Which tensors are boosted; Kitty boosts keys only.
        selection: How channels are scored over the chunk's tokens:
            ``"magnitude"`` mean |x| (Kitty's default), ``"variance"`` mean squared
            deviation from the channel mean, or ``"random"`` (Kitty's ablation;
            one random choice per head, shared by the batch, seeded by ``seed``).
        count_rounding: Boosted channels per head: ``"round"`` gives
            ``max(1, round(fraction * channels))``; ``"floor"`` gives Kitty's
            ``int(fraction * channels + 1e-6)`` (which may be 0: no boost).
        score_dtype: ``"fp32"`` scores in FP32; ``"input"`` scores in the chunk's
            own dtype, like the authors' simulation (ties then break the same way).
        seed: Seed of ``"random"`` selection; mixed with layer and chunk start.
        freeze: ``"none"`` scores every chunk again (Kitty). ``"prefill"`` scores
            the first chunk of a request once per (layer, key/value) and reuses
            that choice for every later chunk, so one-token decode chunks do not
            each pick (and store) their own channels. A chunk whose start is not
            after the previous chunk's start begins a new request.
    """

    fraction: float = 0.125
    boosted_bits: int = 4
    kinds: Tuple[str, ...] = ("key",)
    selection: str = "magnitude"
    count_rounding: str = "round"
    score_dtype: str = "fp32"
    seed: int = 0
    freeze: str = "none"


def supports_channel_bits(quantizer: KVQuantizerConfig) -> bool:
    """Whether a quantizer config declares that it can mix bits across channels."""
    return bool(getattr(type(quantizer), "supports_channel_bits", False))


def channel_bits_need_per_channel(quantizer: KVQuantizerConfig) -> bool:
    """Whether the quantizer can only mix channel bits with PER_CHANNEL groups.

    Grouped codecs (Int) need it so each group has one width; a codec that
    splits whole vectors by channel (TurboQuant) sets the class flag to False.
    """
    return bool(getattr(type(quantizer), "channel_bits_need_per_channel", True))


@PrecisionPolicyRegistry.register(ChannelBoostConfig)
class ChannelBoost(PrecisionPolicy):
    """Channels are scored over the chunk's tokens, separately for each head, so the
    boost follows each chunk. With a page-splitting cache policy (Kitty's), every
    page of ``group_size`` tokens chooses its own boosted channels."""

    config: ChannelBoostConfig

    def __init__(self, config: ChannelBoostConfig) -> None:
        super().__init__(config)
        if isinstance(config.fraction, bool) or not 0 < config.fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")
        if config.boosted_bits not in (2, 3, 4, 8):
            raise ValueError("boosted_bits must be 2, 3, 4 or 8")
        if any(kind not in KINDS for kind in config.kinds):
            raise ValueError(f"kinds must be drawn from {KINDS}")
        if config.selection not in SELECTIONS:
            raise ValueError(f"selection must be one of {SELECTIONS}")
        if config.count_rounding not in COUNT_ROUNDINGS:
            raise ValueError(f"count_rounding must be one of {COUNT_ROUNDINGS}")
        if config.score_dtype not in SCORE_DTYPES:
            raise ValueError(f"score_dtype must be one of {SCORE_DTYPES}")
        if type(config.seed) is not int or config.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if config.freeze not in FREEZES:
            raise ValueError(f"freeze must be one of {FREEZES}")
        # (layer, kind) -> (start of the last chunk seen, frozen channel_bits).
        self._frozen: Dict[Tuple[int, str], Tuple[int, torch.Tensor]] = {}

    def boosted_count(self, channels: int) -> int:
        if self.config.count_rounding == "floor":
            return max(0, min(int(channels * self.config.fraction + 1e-6), channels))
        return max(1, round(self.config.fraction * channels))

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        """Importance of each channel: ``[batch, heads, channels]``."""
        data = x if self.config.score_dtype == "input" else x.float()
        if self.config.selection == "magnitude":
            return data.abs().mean(dim=-2)
        diff = data - data.mean(dim=-2, keepdim=True)
        return diff.pow(2).mean(dim=-2)

    def _random_top(
        self, x: torch.Tensor, layer_idx: int, positions: torch.Tensor, count: int
    ) -> torch.Tensor:
        batch, heads, _, channels = x.shape
        start = int(positions[0]) if positions.numel() else 0
        seed = (self.config.seed * 1_000_003 + layer_idx) * 1_000_003 + start
        generator = torch.Generator().manual_seed(seed % (2**63))
        top = torch.stack(
            [torch.randperm(channels, generator=generator)[:count] for _ in range(heads)]
        )
        return top.to(x.device)[None].expand(batch, heads, count)

    def bits_for(
        self,
        layer_idx: int,
        x: torch.Tensor,
        positions: torch.Tensor,
        *,
        tensor_kind: str,
        default_bits: int,
    ) -> PrecisionPlan:
        if tensor_kind not in self.config.kinds:
            return PrecisionPlan(bits=default_bits)
        channels = x.shape[-1]
        boosted = self.boosted_count(channels)
        if boosted == 0:
            return PrecisionPlan(bits=default_bits)
        if self.config.freeze == "prefill":
            return self._frozen_plan(
                layer_idx, x, positions, tensor_kind, default_bits, boosted
            )
        return PrecisionPlan(
            bits=default_bits,
            channel_bits=self._choose(layer_idx, x, positions, default_bits, boosted),
        )

    def _choose(
        self,
        layer_idx: int,
        x: torch.Tensor,
        positions: torch.Tensor,
        default_bits: int,
        boosted: int,
    ) -> torch.Tensor:
        channels = x.shape[-1]
        shape = (x.shape[0], x.shape[1], channels)
        channel_bits = torch.full(
            shape, default_bits, dtype=torch.int64, device=x.device
        )
        if boosted >= channels:
            return channel_bits.fill_(self.config.boosted_bits)
        if self.config.selection == "random":
            top = self._random_top(x, layer_idx, positions, boosted)
        else:
            top = self.scores(x).topk(boosted, dim=-1).indices
        return channel_bits.scatter_(-1, top, self.config.boosted_bits)

    def _frozen_plan(
        self,
        layer_idx: int,
        x: torch.Tensor,
        positions: torch.Tensor,
        tensor_kind: str,
        default_bits: int,
        boosted: int,
    ) -> PrecisionPlan:
        """Reuse the request's first choice; the same tensor object is returned so
        codecs and the store can recognise (and count once) the shared choice."""
        key = (layer_idx, tensor_kind)
        start = int(positions[0]) if positions.numel() else 0
        saved = self._frozen.get(key)
        expected = (x.shape[0], x.shape[1], x.shape[-1])
        if (
            saved is None
            or start <= saved[0]
            or tuple(saved[1].shape) != expected
            or saved[1].device != x.device
        ):
            chosen = self._choose(layer_idx, x, positions, default_bits, boosted)
        else:
            chosen = saved[1]
        self._frozen[key] = (start, chosen)
        return PrecisionPlan(bits=default_bits, channel_bits=chosen)

    def reset(self) -> None:
        """Forget frozen choices (a new request also resets them automatically)."""
        self._frozen.clear()

    def validate_quantizers(
        self, key_quantizer: KVQuantizerConfig, value_quantizer: KVQuantizerConfig
    ) -> None:
        for kind, quantizer in zip(KINDS, (key_quantizer, value_quantizer)):
            if kind not in self.config.kinds:
                continue
            if not supports_channel_bits(quantizer) or (
                channel_bits_need_per_channel(quantizer)
                and QuantGranularity(quantizer.granularity)
                != QuantGranularity.PER_CHANNEL
            ):
                raise ValueError(
                    f"ChannelBoost on {kind}s needs a quantizer that mixes bits per "
                    "channel (IntQuantizer with PER_CHANNEL "
                    "granularity, so each group has one bit width, or TurboQuant)"
                )
