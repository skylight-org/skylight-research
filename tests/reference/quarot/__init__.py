"""Pinned authors' numerical helpers, independent of Skylight's QuaRot codec."""

import ast
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import torch

NUMERICAL_NAMES = {
    "get_minq_maxq",
    "asym_quant",
    "asym_dequant",
    "asym_quant_dequant",
    "sym_quant",
    "sym_dequant",
    "sym_quant_dequant",
    "ActQuantizer",
}


@lru_cache(maxsize=1)
def official_functions():
    directory = Path(__file__).resolve().parent
    manifest = json.loads((directory / "SOURCE.json").read_text())
    for name, entry in manifest["files"].items():
        assert (
            hashlib.sha256((directory / name).read_bytes()).hexdigest()
            == entry["sha256"]
        )
    source = directory / "quant_utils.py.txt"
    definitions = [
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name in NUMERICAL_NAMES
    ]
    assert len(definitions) == len(NUMERICAL_NAMES)
    namespace = {
        "torch": torch,
        "utils": SimpleNamespace(cleanup_memory=lambda **kwargs: None),
    }
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace


def encode(
    x, *, mode="value_group", group_size=128, bits=4, symmetric=False, clip_ratio=0.95
):
    """Apply the actual upstream head or flattened v_proj invocation."""
    quantizer = official_functions()["ActQuantizer"]()
    batch, heads, tokens, channels = x.shape
    if mode == "key_head":
        source = x.reshape(-1, channels)
    else:
        source = x.transpose(1, 2).reshape(batch, tokens, heads * channels)
    quantizer.configure(
        bits,
        groupsize=-1 if mode == "key_head" else group_size,
        sym=symmetric,
        clip_ratio=clip_ratio,
    )
    quantizer.find_params(source)
    encoded = quantizer.quantize(source)

    def restore_layout(tensor):
        if mode == "key_head":
            return tensor.reshape_as(x)
        return (
            tensor.reshape(batch, tokens, heads, channels).transpose(1, 2).contiguous()
        )

    def compact(tensor):
        return (
            restore_layout(tensor)
            .reshape(batch, heads, tokens, channels // group_size, group_size)[..., :1]
            .clone()
        )

    return SimpleNamespace(
        logical_codes=restore_layout(encoded[0]),
        compact_scales=compact(encoded[1]),
        compact_zero_points=None if symmetric else compact(encoded[2]),
        reconstructed=restore_layout(quantizer(source)),
    )


def independent_unpack(record):
    """Decode the byte stream with Python integers, without production helpers."""
    stream = int.from_bytes(bytes(record.codes.cpu().tolist()), "little")
    count = record.metadata["code_count"]
    values = [
        (stream >> (index * record.bits)) & (2**record.bits - 1)
        for index in range(count)
    ]
    bias = record.metadata["code_bias"]
    return (
        torch.tensor(values, dtype=torch.int16, device=record.device) - bias
    ).reshape(record.shape)


@lru_cache(maxsize=16)
def normalized_matrix(dimension):
    """Construct Sylvester entries through bit parity, not the production builder."""
    return (
        torch.tensor(
            [
                [
                    1.0 if (row & col).bit_count() % 2 == 0 else -1.0
                    for col in range(dimension)
                ]
                for row in range(dimension)
            ],
            dtype=torch.float32,
        )
        / dimension**0.5
    )


def reconstruct(x, *, mode, group_size, bits=4, symmetric=False, clip_ratio=0.95):
    """Independent original-basis oracle for callback/cache-flow comparisons."""
    matrix = normalized_matrix(x.shape[-1]).to(x.device)
    # Store compression owns a contiguous concatenation of the newly moved chunk.
    # Match that layout before the reference GEMM; strided GEMMs can round
    # differently even for identical scalar inputs.
    rotated = (x.contiguous().float() @ matrix).to(x.dtype)
    record = encode(
        rotated,
        mode=mode,
        group_size=group_size,
        bits=bits,
        symmetric=symmetric,
        clip_ratio=clip_ratio,
    )
    return (record.reconstructed.float() @ matrix.T).to(x.dtype)
