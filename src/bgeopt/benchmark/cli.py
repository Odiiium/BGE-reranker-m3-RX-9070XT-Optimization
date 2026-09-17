"""Latency benchmark orchestrator: spawns one worker process per (session, model), interleaving models.

    python scripts/benchmark.py --model-config configs/models/v0_baseline.yaml
    python scripts/benchmark.py --model-config A.yaml --model-config B.yaml --profile quick
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from bgeopt.utils import ensure_rocm_wsl_env
from bgeopt.utils.paths import REPO_ROOT, resolve_path

log = logging.getLogger("bgeopt.benchmark")


def main(argv: list[str] | None = None, default_model_config: str | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(description="Controlled latency benchmark (separate process per session).")
    parser.add_argument("--model-config", action="append", help="model YAML; repeat to benchmark several models")
    parser.add_argument("--override", action="append", default=[], help="key=value applied to every model config")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--benchmark-config", default="configs/benchmark.yaml")
    parser.add_argument("--tag", help="suffix for the results folder name")
    args = parser.parse_args(argv)
    model_configs = args.model_config or ([default_model_config] if default_model_config else None)
    if not model_configs:
        parser.error("--model-config is required")

    ensure_rocm_wsl_env()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")

    from bgeopt.benchmark.aggregate import write_benchmark_run
    from bgeopt.benchmark.profile import BenchmarkConfig
    from bgeopt.evaluation.runner import EvalConfig
    from bgeopt.models import load_model_config
    from bgeopt.utils import capture_environment

    config = BenchmarkConfig.load(resolve_path(args.benchmark_config), args.profile)
    replay_order = config.profile.replay.datasets or EvalConfig.from_yaml(resolve_path(config.eval_config)).datasets
    stamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    models = []
    loaded = [load_model_config(resolve_path(path), args.override) for path in model_configs]
    names = [c["name"] for c in loaded]
    for index, (path, model_config) in enumerate(zip(model_configs, loaded)):
        name = model_config["name"] + (f"-{index}" if names.count(model_config["name"]) > 1 else "")
        models.append({"config_path": path, "config": model_config,
                       "run_dir": resolve_path(config.output_dir) / f"{stamp}_{name}{suffix}"})

    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO_ROOT / "src"), os.environ.get("PYTHONPATH")]))}
    started = time.perf_counter()
    order = []
    for session in range(config.profile.sessions):
        rotation = session % len(models)
        for model in models[rotation:] + models[:rotation]:
            order.append((session, model["config"]["name"]))
            log.info("session %d/%d: %s", session + 1, config.profile.sessions, model["config"]["name"])
            command = [sys.executable, "-m", "bgeopt.benchmark.worker", "--model-config", str(model["config_path"]),
                       "--benchmark-config", args.benchmark_config, "--profile", args.profile,
                       "--session", str(session), "--out", str(model["run_dir"] / "sessions" / f"s{session}")]
            for item in args.override:
                command += ["--override", item]
            subprocess.run(command, check=True, cwd=REPO_ROOT, env=env)
    total_s = time.perf_counter() - started

    environment = capture_environment(REPO_ROOT)
    run_dirs = []
    for model in models:
        session_dirs = sorted((model["run_dir"] / "sessions").glob("s*"))
        sessions = [json.loads((d / "session.json").read_text(encoding="utf-8")) for d in session_dirs]
        samples = pd.concat([pd.read_parquet(d / "samples.parquet") for d in session_dirs], ignore_index=True)
        cells = pd.concat([pd.read_parquet(d / "cells.parquet") for d in session_dirs], ignore_index=True)
        run_info = {
            "run_dir": str(model["run_dir"]),
            "model": sessions[0]["model"],
            "overrides": args.override,
            "profile": {"name": config.profile.name, "sessions": config.profile.sessions,
                        "synthetic": config.profile.synthetic.__dict__, "replay": config.profile.replay.__dict__},
            "seed": config.seed,
            "session_order": order,
            "timing": {"total_s": total_s, "models_in_run": len(models)},
            "environment": environment,
        }
        write_benchmark_run(model["run_dir"], samples, cells, sessions, run_info, n_boot=2000, seed=config.seed,
                            replay_order=replay_order)
        print("\n" + (model["run_dir"] / "summary.md").read_text(encoding="utf-8"))
        print(f"results: {model['run_dir']}")
        run_dirs.append(model["run_dir"])
    return run_dirs
