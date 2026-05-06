#!/usr/bin/env python3
"""
OpenAI-compatible server for HuggingFace models (dense or sparse attention).

Usage:
    python scripts/start_server.py [options] MODEL_NAME PORT

    Multi-GPU (Accelerate): ``CUDA_VISIBLE_DEVICES=0,1,2,3`` then e.g.
    ``--device-map auto [--max-memory-per-gpu-gib 72] MODEL PORT``.

On CUDA, **requires** the ``flash_attn`` package when using
``attn_implementation="flash_attention_2"``. If ``SPARSE_CONFIG`` is set in this
file, the sparse prefill helper also prefers ``flash_attn_func``. Install in the
same Python you use to run this script (e.g. ``uv pip install flash-attn`` or the
repo extra ``.[flash_attn]``). For distributed runs, set
``SPARSE_ATTENTION_SERVER_PYTHON`` to that env's ``python``.

If a prefill call supplies a non-``None`` ``attention_mask`` (unusual for this
server), prefill falls back to SDPA for correctness.
"""

import argparse
import json
import os
import re
import sys
import types
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from sparse_attention_hub.metric_logging.logger import MicroMetricLogger


import torch
import torch.nn as nn
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

# Add project root to sys.path to allow importing from sparse_attention_hub
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from sparse_attention_hub.adapters import ModelAdapterHF
    from sparse_attention_hub.sparse_attention.research_attention import (
        ResearchAttentionConfig,
    )
    from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
        LocalMaskerConfig,
        OracleTopKConfig,
        SinkMaskerConfig,
    )
    from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
        AdaptiveSamplingMaskerConfig,
    )
except ImportError as e:
    print(f"Error: Could not import required modules. {e}")
    print(
        "Ensure you are running from the project root and have all dependencies installed."
    )
    sys.exit(1)


def _resolve_attention_implementation(device: str) -> str:
    """Choose Hugging Face ``attn_implementation`` for the given device.

    CUDA requires ``flash_attn`` so the model never uses SDPA for standard layers.
    CPU uses eager attention.

    Returns:
        ``"flash_attention_2"`` on CUDA, or ``"eager"`` on CPU.

    Raises:
        RuntimeError: If device is CUDA but ``flash_attn`` is not importable.

    Note:
        FlashAttention-2 reduces peak **activation** memory during attention
        and is usually faster. It does **not** remove the KV cache for long
        decoded sequences, so very long agent trajectories can still OOM.
    """
    if "cuda" not in device:
        return "eager"
    try:
        import flash_attn  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "start_server on CUDA requires the flash_attn package (FlashAttention-2). "
            "Install a wheel matching your CUDA/PyTorch (see "
            "https://github.com/Dao-AILab/flash-attention), e.g. "
            "`uv pip install flash-attn` from the repo root."
        ) from exc
    return "flash_attention_2"


def _dense_prefill_attention_bhld(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float,
    training: bool,
    module: nn.Module,
    sliding_window: Optional[Any] = None,
) -> torch.Tensor:
    """Run dense multi-token attention; return layout ``(B, L, H, D)`` like sparse path.

    On CUDA, prefers FlashAttention-2 ``flash_attn_func`` (GQA-friendly: fewer KV heads
    than Q heads). HuggingFace often passes a **materialized 4D SDPA mask** for causal
    decoder prefill; when ``module.is_causal``, batch size is 1, and query/key sequence
    lengths match, that mask is equivalent to ``flash_attn_func(..., causal=True)`` and
    is skipped. Otherwise (sliding window, cross-attn shapes, or ``batch>1`` with mask),
    falls back to PyTorch SDPA with ``repeat_kv``.

    Args:
        queries: Query tensor ``(B, H_q, L_q, D)``.
        keys: Key tensor ``(B, H_kv, L_k, D)``.
        values: Value tensor ``(B, H_kv, L_k, D)``.
        attention_mask: Optional SDPA-style mask (may be ignored for FA2; see above).
        scaling: Softmax scale (typically ``1 / sqrt(D)``).
        dropout: Dropout probability when ``training`` is True.
        training: Whether the owning module is in training mode.
        module: Attention module (for ``is_causal``).
        sliding_window: If set (sliding layers), forces SDPA.

    Returns:
        Attention output tensor of shape ``(B, L_q, H_q, D)``.
    """
    from sparse_attention_hub.sparse_attention.utils.kv_utils import (
        _get_num_key_value_groups,
        repeat_kv,
    )

    if not queries.is_cuda:
        num_key_value_groups_cpu: int = _get_num_key_value_groups(queries, keys)
        key_states_cpu: torch.Tensor = repeat_kv(keys, num_key_value_groups_cpu)
        value_states_cpu: torch.Tensor = repeat_kv(values, num_key_value_groups_cpu)
        out_cpu: torch.Tensor = F.scaled_dot_product_attention(
            queries,
            key_states_cpu,
            value_states_cpu,
            attn_mask=attention_mask,
            dropout_p=dropout if training else 0.0,
            is_causal=False,
        )
        return out_cpu.transpose(1, 2).contiguous()

    use_sliding: bool = sliding_window is not None
    flash_causal_ignore_4d_mask: bool = (
        not use_sliding
        and attention_mask is not None
        and bool(getattr(module, "is_causal", False))
        and queries.shape[0] == 1
        and queries.shape[2] == keys.shape[2]
    )
    needs_sdpa: bool = use_sliding or (
        attention_mask is not None and not flash_causal_ignore_4d_mask
    )

    if needs_sdpa:
        num_key_value_groups: int = _get_num_key_value_groups(queries, keys)
        key_states: torch.Tensor = repeat_kv(keys, num_key_value_groups)
        value_states: torch.Tensor = repeat_kv(values, num_key_value_groups)
        out_sdpa: torch.Tensor = F.scaled_dot_product_attention(
            queries,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=dropout if training else 0.0,
            is_causal=False,
        )
        return out_sdpa.transpose(1, 2).contiguous()

    from flash_attn import flash_attn_func

    causal_flag: bool = bool(getattr(module, "is_causal", False))
    q_bsqd: torch.Tensor = queries.transpose(1, 2).contiguous()
    k_bsqd: torch.Tensor = keys.transpose(1, 2).contiguous()
    v_bsqd: torch.Tensor = values.transpose(1, 2).contiguous()
    out_bsqd: torch.Tensor = flash_attn_func(
        q_bsqd,
        k_bsqd,
        v_bsqd,
        dropout_p=dropout if training else 0.0,
        softmax_scale=scaling,
        causal=causal_flag,
    )
    return out_bsqd.contiguous()


# ==============================================================================
# CONFIGURATION AREA
# Modify this section to change the model's attention behavior.
# ==============================================================================

# ``None``: fully dense HuggingFace attention. Default: sink + local window + OracleTopK
# at 2% key fraction per query row + vAttention adaptive sampling (Desai et al. 2025), matching
# ``research_attention/README.md`` Example 3 (offsets aligned with sink/local = 128).
SPARSE_CONFIG: Optional[ResearchAttentionConfig] = ResearchAttentionConfig(
    masker_configs=[
        SinkMaskerConfig(sink_size=128),
        LocalMaskerConfig(window_size=128),
        OracleTopKConfig(heavy_size=0.02, search_space={}),
        AdaptiveSamplingMaskerConfig(
            base_rate_sampling=0.1,
            epsilon=0.25,
            delta=0.25,
            init_offset=128,
            local_offset=128,
        ),
    ],
)
# ==============================================================================

# Global to track turns for warmup phase
REQUEST_COUNT = 0


def _attn_config(model: nn.Module) -> Any:
    """Return the config object that holds attention head dims (handles nested ``text_config``)."""

    cfg: Any = model.config
    return getattr(cfg, "text_config", cfg)


app = FastAPI(title="Sparse Attention Model Server")
model_adapter: Optional[ModelAdapterHF] = None

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ChatCompletionRequest(BaseModel):
    """OpenAI-style chat body; defaults match precise-coding preset (see distributed run)."""

    model_config = ConfigDict(extra="allow")
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 20
    repetition_penalty: Optional[float] = 1.0
    min_p: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    max_tokens: Optional[int] = 32768
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

def parse_tool_calls(text: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Parses tool calls from model output and converts them to OpenAI format.
    Handles XML tags, Qwen-style special tokens, and markdown-style JSON blocks.
    """
    tool_calls = []
    clean_text = text
    
    # 1. Handle XML-style tool calls (Qwen format)
    # Example: <tool_call><function=terminal><parameter=command>ls</parameter></function></tool_call>
    xml_pattern = r'<tool_call>(.*?)</tool_call>'
    xml_matches = list(re.finditer(xml_pattern, text, re.DOTALL))
    
    for match in xml_matches:
        content = match.group(1)
        func_match = re.search(r'<function=(.*?)>(.*?)</function>', content, re.DOTALL)
        if func_match:
            func_name = func_match.group(1).strip()
            params_content = func_match.group(2).strip()
            
            parameters = {}
            param_matches = re.findall(r'<parameter=(.*?)>(.*?)</parameter>', params_content, re.DOTALL)
            for p_name, p_value in param_matches:
                parameters[p_name.strip()] = p_value.strip()
            
            # Always include the tool call
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(parameters)
                }
            })
    
    # Remove XML tool calls from clean_text
    clean_text = re.sub(xml_pattern, '', clean_text, flags=re.DOTALL)
    
    # 2. Handle Markdown-style JSON tool calls
    # Example: ```json\n{"tool": "terminal", "parameters": {"command": "ls"}}\n```
    json_pattern = r'```json\s*(\{.*?\})\s*```'
    json_matches = list(re.finditer(json_pattern, clean_text, re.DOTALL))
    
    for match in json_matches:
        try:
            data = json.loads(match.group(1))
            # Check if it looks like a tool call
            if "tool" in data and "parameters" in data:
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {
                        "name": data["tool"],
                        "arguments": json.dumps(data["parameters"])
                    }
                })
                # Remove from clean_text
                clean_text = clean_text.replace(match.group(0), "")
        except:
            pass
            
    # 3. Handle Qwen-style special token blocks if they appear in text
    # Example: <|tool_call_start|>...
    if "<|tool_call_start|>" in clean_text:
        pass

    
    pass

    return tool_calls, clean_text.strip()

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    """
    global model_adapter, REQUEST_COUNT
    if model_adapter is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    # Increment turn counter
    REQUEST_COUNT += 1

    # Reset the dense logging flag for this new request
    if hasattr(model_adapter, 'sparse_attention') and model_adapter.sparse_attention:
        model_adapter.sparse_attention._logged_dense = False

    # print(f"Received request: {request.json()}") # Debugging
    
    # Construct messages list for chat template
    messages_dict = []
    for m in request.messages:
        msg = {"role": m.role}
        content = m.content
        if isinstance(content, list):
            # Extract text from complex content list (LiteLLM/OpenAI format)
            content_text = ""
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    content_text += item.get("text", "")
            content = content_text
        
        # Templates often require content to be a string, not None
        msg["content"] = content if content is not None else ""
            
        # Include tool-related fields if present
        if m.tool_call_id:
            msg["tool_call_id"] = m.tool_call_id
        if m.name:
            msg["name"] = m.name
        if m.tool_calls:
            processed_tool_calls = []
            for tc in m.tool_calls:
                new_tc = tc.copy()
                if "function" in new_tc and "arguments" in new_tc["function"]:
                    args = new_tc["function"]["arguments"]
                    if isinstance(args, str):
                        try:
                            new_tc["function"]["arguments"] = json.loads(args)
                        except json.JSONDecodeError:
                            pass # Keep as string if not valid JSON
                processed_tool_calls.append(new_tc)
            msg["tool_calls"] = processed_tool_calls
            
        messages_dict.append(msg)
    
    # Apply chat template if available
    # We pass the tools to the chat template if supported
    if request.tools and hasattr(model_adapter.tokenizer, "apply_chat_template"):
        try:
            prompt = model_adapter.tokenizer.apply_chat_template(
                messages_dict, 
                tools=request.tools,
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception:
            # Fallback if tools aren't supported by the specific chat template version
            prompt = model_adapter.tokenizer.apply_chat_template(
                messages_dict, tokenize=False, add_generation_prompt=True
            )
    elif hasattr(model_adapter.tokenizer, "apply_chat_template") and model_adapter.tokenizer.chat_template:
        prompt = model_adapter.tokenizer.apply_chat_template(
            messages_dict, tokenize=False, add_generation_prompt=True
        )
    else:
        # Fallback simple template
        prompt = ""
        for m in messages_dict:
            prompt += f"{m['role'].capitalize()}: {m['content']}\n"
        prompt += "Assistant:"

    # Tokenize input
    inputs = model_adapter.tokenizer(prompt, return_tensors="pt").to(model_adapter.model.device)
    # Same line shape as ``ModelAdapterHF.process_request`` so ``distributed_swebench_inference``
    # can parse ``server_gpu_*.log`` for live context size during SWE-bench runs.
    input_ids = inputs["input_ids"]
    print(f"Context tokens: {input_ids.shape}", flush=True)

    # Prepare sparse attention metadata required by the custom attention layers
    acfg: Any = _attn_config(model_adapter.model)
    sparse_meta_data: Dict[str, Any] = {
        "batch_size": inputs.input_ids.shape[0],
        "num_heads": acfg.num_attention_heads,
        "head_dim": acfg.hidden_size // acfg.num_attention_heads,
        "seq_len": inputs.input_ids.shape[1],
    }
    
    # Generation parameters
    gen_kwargs = {
        "max_new_tokens": request.max_tokens or 4096,
        "temperature": request.temperature if request.temperature > 0 else 0.001,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "repetition_penalty": request.repetition_penalty,
        "do_sample": request.temperature > 0,
        "pad_token_id": model_adapter.tokenizer.pad_token_id,
        "eos_token_id": model_adapter.tokenizer.eos_token_id,
    }
    
    # Generate response
    with torch.no_grad():
        if model_adapter.sparse_attention_config is not None:
            # Run within sparse mode context
            with model_adapter.enable_sparse_mode():
                outputs = model_adapter.model.generate(
                    **inputs,
                    **gen_kwargs,
                    sparse_meta_data=sparse_meta_data
                )
        else:
            # Standard dense execution
            outputs = model_adapter.model.generate(
                **inputs,
                **gen_kwargs,
            )
    
    # Decode newly generated tokens
    generated_text = model_adapter.tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False
    )
    
    # Log raw output for debugging
    with open("server_debug.log", "a") as f:
        f.write(f"\n--- NEW REQUEST ---\n")
        f.write(f"PROMPT: {prompt}\n")  # Full prompt, not truncated
        f.write(f"RAW OUTPUT: {generated_text}\n")
    
    # Clean up special tokens like <|endoftext|> or <|im_end|> if they appear
    # but keep them for tool call parsing if they are part of the protocol
    
    # Parse tool calls from the generated text
    tool_calls, clean_text = parse_tool_calls(generated_text)
    
    # OpenHands (and some other clients) can fail if BOTH content and tool_calls are present
    # in some versions, but usually content is allowed as "thinking" before tools.
    # However, if content is empty and no tool calls, it MUST be at least an empty string.
    
    # Strip any remaining special tokens from clean_text for the agent
    if model_adapter.tokenizer.eos_token:
        clean_text = clean_text.replace(model_adapter.tokenizer.eos_token, "")
    # Add other common stop tokens if needed
    for stop_token in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        clean_text = clean_text.replace(stop_token, "")
    
    clean_text = clean_text.strip()
    
    # Construct OpenAI-compatible response body
    choice = {
        "index": 0,
        "message": {
            "role": "assistant",
            "content": clean_text if clean_text else "", # Ensure it's at least empty string
        },
        "finish_reason": "tool_calls" if tool_calls else "stop",
    }
    
    if tool_calls:
        choice["message"]["tool_calls"] = tool_calls
        # If there are tool calls, OpenHands might expect the thinking part in content
        # or it might just follow the tool calls.
    
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    response_body = {
        "id": completion_id,
        "object": "chat.completion",
        "created": 123456789,
        "model": request.model,
        "choices": [choice],
        "usage": {
            "prompt_tokens": inputs.input_ids.shape[1],
            "completion_tokens": len(outputs[0]) - inputs.input_ids.shape[1],
            "total_tokens": len(outputs[0]),
        },
    }

    # Hybrid models (e.g. Qwen3.5) only run custom sparse attention on full-attention
    # layers, so logged events per request can stay below the logger's flush_every
    # threshold; flush here so ``server_metrics/micro_metrics.jsonl`` updates.
    MicroMetricLogger().flush()

    return response_body

@app.get("/v1/models")
async def list_models():
    """
    OpenAI-compatible models list endpoint.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": model_adapter.model_name if model_adapter else "unknown",
                "object": "model",
                "created": 123456789,
                "owned_by": "sparse-attention-hub",
            }
        ],
    }

def main():
    parser = argparse.ArgumentParser(
        description="Start OpenAI-compatible sparse attention server"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device label for the adapter (e.g. 'cuda:0')."
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="follow",
        choices=("follow", "auto"),
        help=(
            "How to pass HuggingFace ``device_map`` to ``from_pretrained``. "
            "``follow`` (default): use ``--device`` as ``device_map`` (single-GPU typical). "
            "``auto``: use Accelerate ``device_map='auto'`` across all GPUs visible in the "
            "process (set ``CUDA_VISIBLE_DEVICES`` to multiple indices for sharding)."
        ),
    )
    parser.add_argument(
        "--max-memory-per-gpu-gib",
        type=float,
        default=None,
        help=(
            "When ``--device-map auto``, optional per-visible-GPU cap for HF ``max_memory`` "
            "(e.g. 72.0 leaves headroom on 80GB cards). Indices are 0..N-1 in the visible set."
        ),
    )
    parser.add_argument(
        "--is-hybrid",
        action="store_true",
        help=(
            "Hybrid linear-attention models (e.g. Qwen3.5): set adapter ``hybrid=True`` for "
            "``process_request`` / library paths. HTTP ``/v1/chat/completions`` still uses "
            "``generate()``."
        ),
    )
    parser.add_argument("model", type=str, help="HuggingFace model name")
    parser.add_argument("remaining", nargs="*", help="Port (optionally preceded by 'and')")

    args = parser.parse_args()

    # Parse port from remaining arguments
    if not args.remaining:
        print("Error: Port number is required.")
        sys.exit(1)

    try:
        port = int(args.remaining[-1])
    except ValueError:
        print(f"Error: Invalid port number '{args.remaining[-1]}'.")
        sys.exit(1)

    global model_adapter

    # Use the configuration defined at the top of the file
    sparse_config = SPARSE_CONFIG

    if sparse_config:
        print(f"Running in SPARSE mode with config: {sparse_config}")
    else:
        print("Running in DENSE mode.")

    print(f"Loading model: {args.model}...")

    # Device and dtype setup
    if args.device:
        device: str = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if "cuda" in device else torch.float32
    attn_impl: str = _resolve_attention_implementation(device)
    print(f"Attention implementation: {attn_impl}")

    device_map_mode: str = str(args.device_map)
    if device_map_mode == "auto" and not torch.cuda.is_available():
        print("Error: --device-map auto requires CUDA.")
        sys.exit(1)

    # HuggingFace ``device_map`` / optional ``max_memory`` (Accelerate multi-GPU).
    adapter_device: str = device
    model_kwargs: Dict[str, Any] = {
        "dtype": dtype,
        "attn_implementation": attn_impl,
    }
    if device_map_mode == "auto":
        model_kwargs["device_map"] = "auto"
        if args.max_memory_per_gpu_gib is not None:
            n_vis: int = int(torch.cuda.device_count())
            cap_gib: float = float(args.max_memory_per_gpu_gib)
            max_mem: Dict[int, str] = {
                idx: f"{cap_gib}GiB" for idx in range(n_vis)
            }
            model_kwargs["max_memory"] = max_mem
            print(
                f"Using device_map=auto with max_memory per visible GPU (0..{n_vis - 1}): "
                f"{cap_gib} GiB",
                flush=True,
            )
        else:
            print("Using device_map=auto (no per-GPU max_memory cap).", flush=True)
        if not adapter_device.startswith("cuda"):
            adapter_device = "cuda:0"
    else:
        model_kwargs["device_map"] = device

    try:
        model_adapter = ModelAdapterHF(
            model_name=args.model,
            sparse_attention_config=sparse_config,
            model_kwargs=model_kwargs,
            device=adapter_device,
            hybrid=bool(args.is_hybrid),
        )

        # Patch _validate_model_kwargs to an empty function
        # This allows passing sparse_meta_data as a kwarg to generate()
        def _empty_validate_model_kwargs(self, model_kwargs: dict) -> None:
            """Empty function to bypass model kwargs validation."""
            pass

        model_adapter.model._validate_model_kwargs = types.MethodType(
            _empty_validate_model_kwargs, model_adapter.model
        )

        # Single-device forward wrapper only: multi-GPU ``device_map`` models must not
        # have all kwargs tensors forced onto one device (e.g. ``past_key_values`` shards).
        use_forward_patch: bool = model_kwargs.get("device_map") != "auto" and not isinstance(
            model_kwargs.get("device_map"), dict
        )
        if use_forward_patch:
            _orig_forward = model_adapter.model.forward
            model_dev: torch.device = next(model_adapter.model.parameters()).device

            def _wrapped_forward(*args: Any, **kwargs: Any) -> Any:
                new_args = tuple(
                    a.to(model_dev) if isinstance(a, torch.Tensor) else a for a in args
                )
                new_kwargs = {
                    k: v.to(model_dev) if isinstance(v, torch.Tensor) else v
                    for k, v in kwargs.items()
                }
                return _orig_forward(*new_args, **new_kwargs)

            model_adapter.model.forward = _wrapped_forward

        # Patch sparse attention to use dense FlashAttention-2 fallback for prefill / warmup
        if sparse_config:
            sparse_attn_instance = model_adapter.sparse_attention
            orig_custom_attn = sparse_attn_instance.custom_attention
            
            def patched_custom_attention(self, module, queries, keys, values, attention_mask, scaling, dropout, sparse_meta_data, **kwargs):
                # 1. Dense path: prefill (multiple query positions) or first two HTTP requests (warmup)
                is_prefill = queries.shape[2] > 1
                use_dense = is_prefill or (REQUEST_COUNT <= 2)

                if use_dense:
                    # Only print once per request, not per layer
                    if not hasattr(self, '_logged_dense') or not getattr(self, '_logged_dense', False):
                        phase = "PREFILL" if is_prefill else f"TURN {REQUEST_COUNT} (WARMUP)"
                        sliding_window_kw: Optional[Any] = kwargs.get("sliding_window")
                        use_sliding: bool = sliding_window_kw is not None
                        flash_causal_ignore_4d: bool = (
                            not use_sliding
                            and attention_mask is not None
                            and bool(getattr(module, "is_causal", False))
                            and queries.shape[0] == 1
                            and queries.shape[2] == keys.shape[2]
                        )
                        if attention_mask is None:
                            backend = "FlashAttention-2"
                        elif flash_causal_ignore_4d:
                            backend = (
                                "FlashAttention-2 (causal; ignoring materialized 4D mask, B=1)"
                            )
                        elif use_sliding:
                            backend = "SDPA (sliding_window)"
                        else:
                            backend = "SDPA (attention_mask)"
                        print(
                            f"[{phase}] Using {backend} dense path for {queries.shape[2]} tokens",
                            flush=True,
                        )
                        self._logged_dense = True
                        self._was_prefill = is_prefill

                    out_bhld: torch.Tensor = _dense_prefill_attention_bhld(
                        queries,
                        keys,
                        values,
                        attention_mask,
                        scaling,
                        dropout,
                        module.training,
                        module,
                        kwargs.get("sliding_window"),
                    )
                    return out_bhld, None

                # 2. Verify we are switching to Sparse for decoding (query length 1)
                if queries.shape[2] == 1 and getattr(self, '_was_prefill', False):
                    if kwargs.get('layer_idx') == 0:
                        print(f"[DECODING] Switching to Sparse (Vattn) path for generation...", flush=True)
                        self._was_prefill = False

                # Otherwise use the regular sparse attention path.
                # sparse_meta_data must be keyword-only: ResearchAttention.custom_attention
                # accepts (module..dropout) positionally and reads sparse_meta_data from kwargs.
                return orig_custom_attn(
                    module,
                    queries,
                    keys,
                    values,
                    attention_mask,
                    scaling,
                    dropout,
                    sparse_meta_data=sparse_meta_data,
                    **kwargs,
                )
            
            sparse_attn_instance.custom_attention = types.MethodType(patched_custom_attention, sparse_attn_instance)
            print(
                "Successfully patched sparse attention: dense prefill/warmup prefers "
                "FlashAttention-2; SDPA for sliding-window layers or masks FA2 cannot replace."
            )

        print("Successfully patched model forward and validation for sparse metadata handling.")
        print("Model loaded successfully.")

        if sparse_config is not None:
            metrics_env: Optional[str] = os.environ.get("SAH_METRICS_LOG_DIR")
            result_dir: Path = (
                Path(metrics_env)
                if metrics_env
                else Path("./server_metrics/")
            )
            result_dir.mkdir(parents=True, exist_ok=True)
            metric_logger: MicroMetricLogger = MicroMetricLogger()
            metric_logger.configure_logging(
                log_path=str(result_dir),
                enabled_metrics=[
                    "research_attention_density",
                    "research_attention_output_error",
                ],
            )
            metric_logger.flush()

    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    print(f"Starting server on 0.0.0.0:{port}...")
        
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
