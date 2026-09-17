"""One benchmark session in the current process: synthetic grid + real-query replay.

Measurement rules:
- each cell gets its own warmup; the first warmup call is kept as `cold_ms` (per-shape kernel selection / compilation)
- measured calls run with Python GC disabled; `predict()` returns host-side scores, so device time is included
- cell and dataset order is shuffled per session so thermal drift does not correlate with a cell
"""

from __future__ import annotations

import gc
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from bgeopt.benchmark.profile import BenchmarkProfile
from bgeopt.data import DatasetLoader
from bgeopt.models import BaseReranker

SYNTHETIC_WORD = "data"  # one XLM-R token per repetition, so pair length is exactly controllable


@dataclass(frozen=True)
class ReplayQuery:
    dataset: str
    query_id: str
    query: str
    documents: list[str]


def select_replay_queries(loader: DatasetLoader, datasets: Sequence[str], per_dataset: int,
                          seed: int) -> list[ReplayQuery]:
    """Same seeded sample for every session and every model, so latency distributions are comparable."""
    selected = []
    for name in datasets:
        queries = loader.load(name).queries
        rng = np.random.default_rng(seed)
        picks = sorted(rng.choice(len(queries), size=min(per_dataset, len(queries)), replace=False))
        selected.extend(ReplayQuery(name, queries[i].query_id, queries[i].query, queries[i].doc_texts) for i in picks)
    return selected


def build_synthetic_input(reranker: BaseReranker, candidates: int, pair_tokens: int,
                          query_tokens: int) -> tuple[str, list[str]]:
    """Query + `candidates` identical documents whose pairs are exactly `pair_tokens` long after the variant's own
    prepare() (special tokens included), so there is no padding."""
    query = " ".join([SYNTHETIC_WORD] * query_tokens)
    words = max(pair_tokens - query_tokens - 4, 1)
    for _ in range(10):
        prepared = reranker.prepare(query, [" ".join([SYNTHETIC_WORD] * words)])
        if prepared.num_tokens == pair_tokens:
            return query, [" ".join([SYNTHETIC_WORD] * words)] * candidates
        words += pair_tokens - prepared.num_tokens
        if words < 1:
            break
    raise RuntimeError(f"cannot build a synthetic pair of {pair_tokens} tokens (query {query_tokens} tokens)")


def timed_call(reranker: BaseReranker, query: str, documents: Sequence[str]) -> dict[str, Any]:
    t0 = time.perf_counter()
    prepared = reranker.prepare(query, documents)
    t1 = time.perf_counter()
    reranker.predict(prepared)
    t2 = time.perf_counter()
    return {
        "prepare_ms": (t1 - t0) * 1e3, "predict_ms": (t2 - t1) * 1e3, "e2e_ms": (t2 - t0) * 1e3,
        "num_pairs": prepared.num_pairs, "num_tokens": prepared.num_tokens,
        "num_padded_tokens": prepared.num_padded_tokens,
    }


def run_session(reranker: BaseReranker, profile: BenchmarkProfile, replay: list[ReplayQuery], session: int,
                seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (samples: one row per measured call, cells: one row per synthetic cell / replay dataset)."""
    rng = np.random.default_rng(seed + 1000 * (session + 1))
    samples: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    syn = profile.synthetic

    grid = [(n, length) for n in syn.candidates for length in syn.pair_tokens]
    inputs = {length: build_synthetic_input(reranker, 1, length, syn.query_tokens) for length in syn.pair_tokens}
    for index in rng.permutation(len(grid)):
        candidates, length = grid[index]
        query, documents = inputs[length][0], inputs[length][1] * candidates
        cell = f"n{candidates}_L{length}"

        reranker.reset_peak_memory()
        warm = [timed_call(reranker, query, documents) for _ in range(syn.warmup)]
        steady = [w["e2e_ms"] for w in warm[1:]] or [warm[0]["e2e_ms"]]
        reps = int(min(max(math.ceil(syn.target_seconds * 1e3 / float(np.median(steady))), syn.min_reps), syn.max_reps))

        gc.collect()
        gc.disable()
        try:
            measured = [timed_call(reranker, query, documents) for _ in range(reps)]
        finally:
            gc.enable()
        samples.extend({"session": session, "workload": "synthetic", "cell": cell, "dataset": "", "query_id": "",
                        "candidates": candidates, "pair_tokens": length, "rep": rep, **m}
                       for rep, m in enumerate(measured))
        cells.append({"session": session, "workload": "synthetic", "cell": cell, "cold_ms": warm[0]["e2e_ms"],
                      "warmup": syn.warmup, "measured": reps, "peak_memory_mb": reranker.peak_memory_mb()})

    by_dataset: dict[str, list[ReplayQuery]] = {}
    for item in replay:
        by_dataset.setdefault(item.dataset, []).append(item)
    names = list(by_dataset)
    for index in rng.permutation(len(names)):
        name = names[index]
        queries = by_dataset[name]
        reranker.reset_peak_memory()
        warm = [timed_call(reranker, q.query, q.documents) for q in queries[: profile.replay.warmup_queries]]

        gc.collect()
        gc.disable()
        try:
            for rep in range(profile.replay.repeats):
                for q in queries:
                    m = timed_call(reranker, q.query, q.documents)
                    samples.append({"session": session, "workload": "replay", "cell": name, "dataset": name,
                                    "query_id": q.query_id, "candidates": m["num_pairs"], "pair_tokens": -1,
                                    "rep": rep, **m})
        finally:
            gc.enable()
        cells.append({"session": session, "workload": "replay", "cell": name,
                      "cold_ms": warm[0]["e2e_ms"] if warm else float("nan"), "warmup": len(warm),
                      "measured": profile.replay.repeats * len(queries), "peak_memory_mb": reranker.peak_memory_mb()})

    return pd.DataFrame(samples), pd.DataFrame(cells)
