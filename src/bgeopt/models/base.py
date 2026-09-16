"""Reranker interface. Every model variant (V0 baseline, compiled, custom Triton kernels) implements it,
so the evaluation pipeline stays identical and variants are swapped only via config.

The scoring of one query is split in two timed stages:
    prepare()  CPU: tokenization, truncation, sorting, batching, padding (whatever the variant needs)
    predict()  device: forward pass; must return host-side scores, i.e. the device is synchronized
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

import numpy as np


@dataclass
class RerankerConfig:
    name: str
    version: str
    implementation: str
    description: str = ""
    warmup_iterations: int = 3


@dataclass
class PreparedInput:
    """Output of prepare(). `payload` is variant-specific (padded batches, packed varlen tensors, ...)."""

    payload: Any
    num_pairs: int
    num_tokens: int  # real tokens, without padding
    num_padded_tokens: int  # tokens actually sent through the model (== num_tokens for unpadded variants)
    extra: dict[str, Any] = field(default_factory=dict)


class BaseReranker(ABC):
    config_class: ClassVar[type[RerankerConfig]] = RerankerConfig

    def __init__(self, config: RerankerConfig):
        self.config = config

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    def prepare(self, query: str, documents: Sequence[str]) -> PreparedInput:
        """Turn one query and its candidates into model inputs."""

    @abstractmethod
    def predict(self, prepared: PreparedInput) -> np.ndarray:
        """Relevance scores (float32, higher = more relevant) in the original document order."""

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        return self.predict(self.prepare(query, documents))

    def warmup(self) -> None:
        """Run representative inputs so kernel selection / compilation does not pollute measurements."""
        documents = [" ".join(["warmup"] * n) for n in (8, 64, 256, 512)] * 8
        for _ in range(self.config.warmup_iterations):
            self.score("warmup query", documents)

    def reset_peak_memory(self) -> None:
        """Reset the device peak-memory counter (no-op for variants without a device)."""

    def peak_memory_mb(self) -> float | None:
        return None

    def describe(self) -> dict[str, Any]:
        """Metadata stored with every evaluation run."""
        return {"class": f"{type(self).__module__}.{type(self).__qualname__}", "config": asdict(self.config)}
