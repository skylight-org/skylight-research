# Inspired by https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py.
# Author: Haojun Xia (xhjustc@gmail.com)

from typing import Optional, Any
from dataclasses import dataclass
import argparse

import torch
from transformers.cache_utils import CacheConfig, DynamicCache

from .utils_quant import build_promote_mask, fake_quant_groupwise_lastdim


@dataclass
class KittyKVCacheConfig(CacheConfig):
    """
    Configuration class for Kitty KV cache settings.

    Attributes:
        nbits (`Optional[int]`, *optional*, defaults to 4):
            Number of bits, can be 2 or 4 for the `quanto` backend and one of [1, 2, 3, 4, 8] for the `HQQ` backend. Defaults to 2.
    """
    def __init__(
        self,
        sink_length: int = 32,
        buffer_length: int = 128,
        group_size: int = 128,
        kbits: int = 2,
        vbits: int = 2,
        promote_ratio: float = 0.1,
        promote_bit: int = 4,
        channel_selection: int = 1,               # -1: Unspecified, 0: Random, 1: Magnitude-based
        VCache_BitDecoding: bool = False,         # The behavior of Value Cache, set to True means BitDecoding, otherwise KIVI Style Value Cache
        PostQuant: bool = True,                   # Post Quantization is always enabled
    ):
        super().__init__("kitty_kv")
        self.sink_length = sink_length
        self.buffer_length = buffer_length
        self.group_size = group_size
        self.kbits = kbits
        self.vbits = vbits
        self.promote_ratio = promote_ratio
        self.promote_bit = promote_bit
        self.channel_selection = channel_selection
        self.VCache_BitDecoding = VCache_BitDecoding
        self.PostQuant = PostQuant
        #
        self.validate()

    def validate(self):
        """Validates if the arguments passed are correct"""
        incorrect_arg_msg = (
            "Some of the keys in `cache_config` are defined incorrectly. `{key}` should be {correct_value}` "
            "but found {found_value}"
        )
        if self.channel_selection not in [0,1]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="channel_selection",
                    correct_value="0 or 1",
                    found_value=self.channel_selection,
                ),
            )
        if self.buffer_length < 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="buffer_length",
                    correct_value="larger than 0",
                    found_value=self.buffer_length,
                ),
            )
        if self.sink_length < 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="sink_length",
                    correct_value="larger than 0",
                    found_value=self.sink_length,
                ),
            )
        if self.group_size <= 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="group_size",
                    correct_value="larger than 0",
                    found_value=self.group_size,
                ),
            )
        if self.kbits < 1 or self.kbits > 16:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="kbits",
                    correct_value="1 to 16",
                    found_value=self.kbits,
                ),
            )
        if self.vbits < 1 or self.vbits > 16:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="vbits",
                    correct_value="1 to 16",
                    found_value=self.vbits,
                ),
            )
        if self.promote_ratio < 0.0 or self.promote_ratio > 1.0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="promote_ratio",
                    correct_value="between 0.0 and 1.0",
                    found_value=self.promote_ratio,
                ),
            )
        if self.promote_ratio > 0 and self.promote_bit < self.kbits:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="promote_bit",
                    correct_value=f"promote_bit should be larger than kbits ({self.kbits})",
                    found_value=self.promote_bit,
                ),
            )
        if self.promote_bit <= 0 or self.promote_bit >= 16:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="promote_bit",
                    correct_value=f"between 1 and 15",
                    found_value=self.promote_bit,
                ),
            )
        if self.VCache_BitDecoding not in [False]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="VCache_BitDecoding",
                    correct_value="False",
                    found_value=self.VCache_BitDecoding,
                ),
            )
        if self.group_size > self.buffer_length or self.buffer_length % self.group_size != 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="group_size",
                    correct_value="a factor of buffer_length ({})".format(self.buffer_length),
                    found_value=self.group_size,
                ),
            )
        
class KittyKVCache(DynamicCache):
    """
    A quantizer cache that supports Kitty quantization.
    [batch_size, num_heads, seq_len, head_dim]
    """

    def __init__(self, cache_config: KittyKVCacheConfig) -> None:
        super().__init__()
        # Initialize Kitty-KV specific configurations
        self.sink_length = cache_config.sink_length
        self.buffer_length = cache_config.buffer_length
        self.group_size = cache_config.group_size
        self.kbits = cache_config.kbits
        self.vbits = cache_config.vbits
        self.promote_ratio = cache_config.promote_ratio
        self.promote_bit = cache_config.promote_bit
        self.channel_selection = cache_config.channel_selection
        self.VCache_BitDecoding = cache_config.VCache_BitDecoding
        self.PostQuant = cache_config.PostQuant
        self.cache_implementation = cache_config.cache_implementation
        #
        #self.query_cache: list[torch.Tensor] = []
        #self.query_score: list[torch.Tensor] = []

    # To Do: support prefill length smaller than sink_length
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        #query_states = cache_kwargs.get("query_states", None) if cache_kwargs is not None else None

        if len(self.key_cache) < layer_idx:
            raise ValueError("QuantizedCache does not support model usage where layers are skipped. Use DynamicCache.")
        ################################################## Prefill Phase ##################################################
        elif len(self.key_cache) == layer_idx:
            # Initialize the key and value caches for the layer
            self.key_cache.append(key_states.detach().clone())
            self.value_cache.append(value_states.detach().clone())
            current_key_cache = self.key_cache[layer_idx]
            current_value_cache = self.value_cache[layer_idx]
            current_cache_length = current_key_cache.shape[-2]

            if self.PostQuant:
                keys_to_return = current_key_cache.detach().clone()
                values_to_return = current_value_cache.detach().clone()

            assert current_cache_length > self.sink_length, "Kitty-KV: sequence length must be greater than sink_length, currently."
            # Need to quantize the middle part of the key and value caches
            if current_cache_length > self.sink_length + self.buffer_length:
                start_idx = self.sink_length
                num_tokens = current_cache_length - self.sink_length
                num_token_to_buffer = num_tokens % self.buffer_length
                num_token_to_quantize = num_tokens - num_token_to_buffer
                end_idx = start_idx + num_token_to_quantize
                # Quantize Key Cache
                for idx in range(start_idx, end_idx, self.buffer_length):
                    key_slice = current_key_cache[:, :, idx:idx+self.buffer_length, :].transpose(2, 3).contiguous()
                    promote_mask = build_promote_mask(key_slice, self.promote_ratio, self.channel_selection)
                    key_slice = fake_quant_groupwise_lastdim(key_slice, self.group_size, self.kbits, promote_mask, self.promote_bit).transpose(2, 3).contiguous()
                    current_key_cache[:, :, idx:idx+self.buffer_length, :] = key_slice
                # Quantize Value Cache
                if not self.VCache_BitDecoding:
                    num_token_to_quantize = num_tokens - self.buffer_length   # KIVI Style Value Cache
                    end_idx = start_idx + num_token_to_quantize
                value_slice = current_value_cache[:, :, start_idx:end_idx, :]
                value_slice = fake_quant_groupwise_lastdim(value_slice, self.group_size, self.vbits)
                current_value_cache[:, :, start_idx:end_idx, :] = value_slice
        ################################################## Decoding Phase ##################################################
        else:
            # update the key and value caches
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            current_key_cache = self.key_cache[layer_idx]
            current_value_cache = self.value_cache[layer_idx]
            current_cache_length = current_key_cache.shape[-2]

            if self.PostQuant:
                keys_to_return = current_key_cache.detach().clone()
                values_to_return = current_value_cache.detach().clone()

            # quantize
            num_tokens_kv_to_quantize = current_cache_length - self.sink_length - self.buffer_length
            if num_tokens_kv_to_quantize > 0 and (num_tokens_kv_to_quantize % self.buffer_length == 1):  # need to quantize
                # Quantize Key Cache
                key_slice = current_key_cache[:, :, -self.buffer_length-1:-1, :]
                promote_mask = build_promote_mask(key_slice.transpose(2, 3).contiguous(), self.promote_ratio, self.channel_selection)
                key_slice = fake_quant_groupwise_lastdim(key_slice.transpose(2, 3).contiguous(), self.group_size, self.kbits, promote_mask, self.promote_bit).transpose(2, 3).contiguous()
                current_key_cache[:, :, -self.buffer_length-1:-1, :] = key_slice
                # Quantize Value Cache (BitDecoding)
                if self.VCache_BitDecoding:
                    value_slice = current_value_cache[:, :, -self.buffer_length-1:-1, :]
                    value_slice = fake_quant_groupwise_lastdim(value_slice, self.group_size, self.vbits)
                    current_value_cache[:, :, -self.buffer_length-1:-1, :] = value_slice
            # Quantize Value Cache (KIVI Style Value Cache, quantizing a Token each Decoding Step)
            if not self.VCache_BitDecoding:
                if num_tokens_kv_to_quantize > 0:
                    value_slice = current_value_cache[:, :, -self.buffer_length-1:-self.buffer_length, :]
                    value_slice = fake_quant_groupwise_lastdim(value_slice, self.group_size, self.vbits)
                    current_value_cache[:, :, -self.buffer_length-1:-self.buffer_length, :] = value_slice
        ####################################################################################################################
        if self.PostQuant:
            return keys_to_return, values_to_return
        else:
            return current_key_cache, current_value_cache

def get_kvcache_kitty(args: argparse.Namespace) -> KittyKVCache:
    """
    Get the Kitty KVCache object.
    Returns:
        KittyKVCache: The KittyKVCache object.
    """
    #
    cache_config = KittyKVCacheConfig(
        sink_length         = args.sink_length,
        buffer_length       = args.buffer_length,
        group_size          = args.group_size,
        kbits               = args.kbits,
        vbits               = args.vbits,
        promote_ratio       = args.promote_ratio,
        promote_bit         = args.promote_bit,
        channel_selection   = args.channel_selection,
        VCache_BitDecoding  = False,  # Using KIVI Style V Cache
        PostQuant           = True,  # Post Quantization is always enabled for Kitty KV Cache
    )
    #
    return KittyKVCache(cache_config=cache_config)