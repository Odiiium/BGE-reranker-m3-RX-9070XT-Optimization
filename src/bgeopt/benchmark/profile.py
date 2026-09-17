"""Benchmark profile definitions (configs/benchmark.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SyntheticSpec:
    candidates: list[int]
    pair_tokens: list[int]
    query_tokens: int = 16
    warmup: int = 10
    target_seconds: float = 4.0
    min_reps: int = 30
    max_reps: int = 200


@dataclass(frozen=True)
class ReplaySpec:
    datasets: list[str] | None = None
    queries_per_dataset: int = 20
    repeats: int = 2
    warmup_queries: int = 3


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    sessions: int
    synthetic: SyntheticSpec
    replay: ReplaySpec


@dataclass(frozen=True)
class BenchmarkConfig:
    output_dir: str
    eval_config: str
    seed: int
    profile: BenchmarkProfile

    @classmethod
    def load(cls, path: str | Path, profile: str) -> BenchmarkConfig:
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if profile not in raw["profiles"]:
            raise KeyError(f"unknown benchmark profile '{profile}', available: {', '.join(raw['profiles'])}")
        spec = raw["profiles"][profile]
        return cls(
            output_dir=raw.get("output_dir", "results/latency"),
            eval_config=raw.get("eval_config", "configs/eval.yaml"),
            seed=int(raw.get("seed", 0)),
            profile=BenchmarkProfile(
                name=profile,
                sessions=int(spec["sessions"]),
                synthetic=SyntheticSpec(**spec["synthetic"]),
                replay=ReplaySpec(**spec["replay"]),
            ),
        )
