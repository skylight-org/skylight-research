"""Run the authors' pinned pure-PyTorch KIVI helpers without CUDA dependencies."""

import ast
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import torch


class _TupleIndexing(ast.NodeTransformer):
    """Keep the authors' indexing semantics on PyTorch 2.9+, without changing math."""

    def visit_Subscript(self, node):
        self.generic_visit(node)
        if isinstance(node.slice, ast.Name) and node.slice.id in (
            "packed_indices",
            "unpacked_indices",
        ):
            node.slice = ast.Call(
                func=ast.Name(id="tuple", ctx=ast.Load()),
                args=[node.slice],
                keywords=[],
            )
        return node


@lru_cache(maxsize=1)
def official_functions():
    directory = Path(__file__).resolve().parent
    source = directory / "new_pack.py.txt"
    manifest = json.loads((directory / "SOURCE.json").read_text())
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == manifest["files"]["new_pack.py.txt"]["sha256"]
    )
    names = {
        "quant_and_pack_kcache",
        "quant_and_pack_vcache",
        "unpack_and_dequant_kcache",
        "unpack_and_dequant_vcache",
        "pack_tensor",
        "unpack_tensor",
    }
    definitions = [
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(definitions) == len(names)
    namespace = {"torch": torch}
    module = ast.fix_missing_locations(
        _TupleIndexing().visit(ast.Module(body=definitions, type_ignores=[]))
    )
    exec(
        compile(module, str(source), "exec"),
        namespace,
    )
    return namespace


def encode(x, kind, bits=2, group_size=32):
    """Return the official int32-packed tensor, scale, and minimum."""
    name = "quant_and_pack_kcache" if kind == "key" else "quant_and_pack_vcache"
    return official_functions()[name](x.contiguous(), group_size, bits)


def decode(record, kind, bits=2, group_size=32, *, dtype=None):
    """Use the official unpack arithmetic; allow an explicit framework output cast."""
    name = "unpack_and_dequant_kcache" if kind == "key" else "unpack_and_dequant_vcache"
    packed, scale, minimum = record
    result = official_functions()[name](packed, scale, minimum, group_size, bits)
    return result if dtype is None else result.to(dtype)


class OfficialKIVICache:
    """Small independent state machine following released llama_kivi.py ordering.

    Prefill uses original K/V. Decode attention consumes the old packed history
    and full residual plus the new token BEFORE buffers are compressed. GPU GEMV
    is represented by the official standalone reconstruction for a CPU oracle.
    """

    def __init__(self, bits=2, group_size=32, residual_length=128):
        self.bits = bits
        self.group_size = group_size
        self.residual_length = residual_length
        self.records = {"key": [], "value": []}
        self.residual = {}

    def prefill(self, keys, values):
        for kind, x in (("key", keys), ("value", values)):
            length = x.shape[-2]
            count = (
                length - length % self.residual_length
                if kind == "key"
                else max(0, length - self.residual_length)
            )
            if count:
                self.records[kind].append(
                    encode(x[..., :count, :], kind, self.bits, self.group_size)
                )
            self.residual[kind] = x[..., count:, :].clone()
        return keys, values

    def get(self, kind):
        residual = self.residual[kind]
        parts = [
            decode(record, kind, self.bits, self.group_size, dtype=residual.dtype)
            for record in self.records[kind]
        ]
        return torch.cat([*parts, residual], dim=-2)

    def step(self, keys, values):
        assert keys.shape[-2] == values.shape[-2] == 1
        prepared = tuple(
            torch.cat([self.get(kind), x], dim=-2)
            for kind, x in (("key", keys), ("value", values))
        )
        for kind, x in (("key", keys), ("value", values)):
            tail = torch.cat([self.residual[kind], x], dim=-2)
            count = 0
            if kind == "key" and tail.shape[-2] == self.residual_length:
                count = self.residual_length
            elif kind == "value" and tail.shape[-2] > self.residual_length:
                count = 1
            if count:
                self.records[kind].append(
                    encode(tail[..., :count, :], kind, self.bits, self.group_size)
                )
            self.residual[kind] = tail[..., count:, :].clone()
        return prepared
