"""Name -> reranker class registry, so a config file selects the implementation."""

from __future__ import annotations

import importlib
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from bgeopt.models.base import BaseReranker

_REGISTRY: dict[str, type[BaseReranker]] = {}

# Implementation key -> module that registers it. Imported lazily so that torch-heavy variants load only when used.
BUILTIN_IMPLEMENTATIONS = {
    "hf-cross-encoder": "bgeopt.models.hf_cross_encoder",
}


def register_reranker(key: str):
    def decorator(cls: type[BaseReranker]) -> type[BaseReranker]:
        if key in _REGISTRY and _REGISTRY[key] is not cls:
            raise ValueError(f"reranker implementation '{key}' is already registered by {_REGISTRY[key]}")
        _REGISTRY[key] = cls
        return cls

    return decorator


def available_rerankers() -> list[str]:
    return sorted(set(_REGISTRY) | set(BUILTIN_IMPLEMENTATIONS))


def build_reranker(config: dict[str, Any] | str | Path) -> BaseReranker:
    if not isinstance(config, dict):
        config = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    key = config.get("implementation")
    if key not in _REGISTRY and key in BUILTIN_IMPLEMENTATIONS:
        importlib.import_module(BUILTIN_IMPLEMENTATIONS[key])
    if key not in _REGISTRY:
        raise KeyError(f"unknown implementation '{key}', registered: {', '.join(available_rerankers())}")
    cls = _REGISTRY[key]
    allowed = {f.name for f in fields(cls.config_class)}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown config fields {sorted(unknown)}")
    return cls(cls.config_class(**config))
