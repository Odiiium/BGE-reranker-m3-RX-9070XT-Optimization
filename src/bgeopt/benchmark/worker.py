"""Benchmark worker: one session of one model in a fresh process (spawned by bgeopt.benchmark.cli).

    python -m bgeopt.benchmark.worker --model-config configs/models/v0_baseline.yaml --session 0 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from bgeopt.utils import ensure_rocm_wsl_env
from bgeopt.utils.paths import resolve_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--benchmark-config", default="configs/benchmark.yaml")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    ensure_rocm_wsl_env()
    logging.basicConfig(level=logging.WARNING)

    from bgeopt.benchmark.profile import BenchmarkConfig
    from bgeopt.benchmark.session import build_synthetic_input, run_session, select_replay_queries, timed_call
    from bgeopt.data import DatasetLoader
    from bgeopt.evaluation.runner import EvalConfig
    from bgeopt.models import build_reranker, load_model_config

    config = BenchmarkConfig.load(resolve_path(args.benchmark_config), args.profile)
    eval_config = EvalConfig.from_yaml(resolve_path(config.eval_config))
    datasets = config.profile.replay.datasets or eval_config.datasets
    loader = DatasetLoader(resolve_path(eval_config.datasets_config), resolve_path(eval_config.data_dir))
    missing = [name for name in datasets if not loader.is_prepared(name)]
    if missing:
        raise SystemExit(f"slices not prepared: {', '.join(missing)} — run `make data` first")
    replay = select_replay_queries(loader, datasets, config.profile.replay.queries_per_dataset, config.seed)

    load_started = time.perf_counter()
    reranker = build_reranker(load_model_config(resolve_path(args.model_config), args.override))
    load_s = time.perf_counter() - load_started

    query, documents = build_synthetic_input(reranker, 32, 256, config.profile.synthetic.query_tokens)
    first_call_ms = timed_call(reranker, query, documents)["e2e_ms"]
    warmup_started = time.perf_counter()
    reranker.warmup()
    warmup_s = time.perf_counter() - warmup_started

    samples, cells = run_session(reranker, config.profile, replay, args.session, config.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(out / "samples.parquet", index=False)
    cells.to_parquet(out / "cells.parquet", index=False)
    info = {"session": args.session, "pid": os.getpid(), "load_s": load_s, "first_call_ms": first_call_ms,
            "warmup_s": warmup_s, "total_s": time.perf_counter() - started, "model": reranker.describe()}
    (out / "session.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
