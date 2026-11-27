"""Run the authors' pinned Kitty accuracy-simulation code (pure PyTorch)."""

import ast
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import torch

DIRECTORY = Path(__file__).resolve().parent


def _source(name: str) -> str:
    manifest = json.loads((DIRECTORY / "SOURCE.json").read_text())
    data = (DIRECTORY / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == manifest["files"][name]["sha256"]
    return data.decode()


def _definitions(name: str, wanted: set) -> ast.Module:
    nodes = [
        node
        for node in ast.parse(_source(name)).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted
    ]
    assert {node.name for node in nodes} == wanted
    return ast.Module(body=nodes, type_ignores=[])


class _CacheConfig:
    """Stand-in for transformers' removed CacheConfig: only stores the name."""

    def __init__(self, cache_implementation: str) -> None:
        self.cache_implementation = cache_implementation


class _DynamicCache:
    """Stand-in for the list-based DynamicCache the reference subclasses."""

    def __init__(self) -> None:
        self.key_cache: list = []
        self.value_cache: list = []


@lru_cache(maxsize=1)
def official_functions() -> dict:
    """``build_promote_mask``, ``fake_quant_groupwise_lastdim``, ``KittyKVCacheConfig``
    and ``KittyKVCache`` exactly as published."""
    from dataclasses import dataclass

    namespace: dict = {"torch": torch, "Optional": Optional, "Any": Any}
    utils = _definitions(
        "utils_quant.py.txt", {"build_promote_mask", "fake_quant_groupwise_lastdim"}
    )
    exec(compile(utils, "utils_quant.py", "exec"), namespace)
    namespace.update(
        dataclass=dataclass, CacheConfig=_CacheConfig, DynamicCache=_DynamicCache
    )
    cache = _definitions(
        "kitty_simulate.py.txt", {"KittyKVCacheConfig", "KittyKVCache"}
    )
    exec(compile(cache, "kitty_simulate.py", "exec"), namespace)
    return namespace
