"""Aggregates session samples into latency statistics and writes a benchmark run.

results/latency/<run>/
    run_info.json     model config + weights hash, profile, session order, environment, timing
    summary.md        cold start, synthetic grid pivots (p50 / p95 / pairs/s / session noise), replay table
    latency.csv       one row per synthetic cell / replay dataset
    samples.parquet   every measured call (all sessions)
    cells.parquet     per-session cell info: cold (first) call, reps, peak memory
    sessions/s<k>/    raw worker output
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bgeopt.evaluation.report import markdown_table
from bgeopt.utils.stats import median_ci


def aggregate_latency(samples: pd.DataFrame, cells: pd.DataFrame, n_boot: int = 2000, seed: int = 0,
                      replay_order: list[str] | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (workload, cell), group in samples.groupby(["workload", "cell"], sort=False):
        predict = group["predict_ms"].to_numpy()
        e2e = group["e2e_ms"].to_numpy()
        session_p50 = group.groupby("session")["predict_ms"].median()
        cell_info = cells[(cells["workload"] == workload) & (cells["cell"] == cell)]
        predict_s = predict.sum() / 1e3
        tokens, padded = group["num_tokens"].sum(), group["num_padded_tokens"].sum()
        ci = median_ci(predict, n_boot, rng)
        rows.append({
            "workload": workload,
            "cell": cell,
            "candidates": int(group["num_pairs"].median()),
            "pair_tokens": float(tokens / group["num_pairs"].sum()),
            "padding_ratio": float(1 - tokens / padded) if padded else 0.0,
            "samples": len(group),
            "sessions": int(group["session"].nunique()),
            "prepare_ms_p50": float(np.median(group["prepare_ms"])),
            "predict_ms_mean": float(predict.mean()),
            "predict_ms_std": float(predict.std(ddof=1)) if len(predict) > 1 else 0.0,
            "predict_ms_min": float(predict.min()),
            "predict_ms_p50": float(np.median(predict)),
            "predict_ms_p50_ci_low": ci[0],
            "predict_ms_p50_ci_high": ci[1],
            "predict_ms_p90": float(np.quantile(predict, 0.90)),
            "predict_ms_p95": float(np.quantile(predict, 0.95)),
            "predict_ms_p99": float(np.quantile(predict, 0.99)),
            "e2e_ms_p50": float(np.median(e2e)),
            "e2e_ms_p95": float(np.quantile(e2e, 0.95)),
            "e2e_ms_p99": float(np.quantile(e2e, 0.99)),
            "pairs_per_s": float(group["num_pairs"].sum() / predict_s) if predict_s else 0.0,
            "tokens_per_s": float(tokens / predict_s) if predict_s else 0.0,
            "session_p50_cv": float(session_p50.std(ddof=1) / session_p50.mean()) if len(session_p50) > 1 else float("nan"),
            "cold_ms_max": float(cell_info["cold_ms"].max()) if len(cell_info) else float("nan"),
            "peak_memory_mb": float(cell_info["peak_memory_mb"].max()) if len(cell_info) else float("nan"),
        })
    frame = pd.DataFrame(rows)
    synthetic = frame[frame["workload"] == "synthetic"].sort_values(["candidates", "pair_tokens"])
    replay = frame[frame["workload"] == "replay"]
    if replay_order:
        replay = replay.set_index("cell").loc[[c for c in replay_order if c in set(replay["cell"])]].reset_index()
        replay = replay[frame.columns]
    return pd.concat([synthetic, replay], ignore_index=True)


def _pivot(latency: pd.DataFrame, value: str, fmt: str) -> pd.DataFrame:
    synthetic = latency[latency["workload"] == "synthetic"].copy()
    synthetic["pair_tokens"] = synthetic["pair_tokens"].round().astype(int)
    table = synthetic.pivot(index="candidates", columns="pair_tokens", values=value)
    table = table.apply(lambda col: col.map(lambda v: fmt.format(v) if pd.notna(v) else "–"))
    table.columns = [f"L={c}" for c in table.columns]
    return table.reset_index().rename(columns={"candidates": "candidates \\ tokens"})


def write_benchmark_run(run_dir: Path, samples: pd.DataFrame, cells: pd.DataFrame, sessions: list[dict[str, Any]],
                        run_info: dict[str, Any], n_boot: int, seed: int,
                        replay_order: list[str] | None = None) -> pd.DataFrame:
    run_dir.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(run_dir / "samples.parquet", index=False)
    cells.to_parquet(run_dir / "cells.parquet", index=False)
    latency = aggregate_latency(samples, cells, n_boot=n_boot, seed=seed, replay_order=replay_order)
    latency.to_csv(run_dir / "latency.csv", index=False)
    run_info = {**run_info, "sessions": sessions}
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False, default=str),
                                           encoding="utf-8")

    model = run_info["model"]["config"]
    env = run_info["environment"]
    cold = pd.DataFrame([{"session": s["session"], "load_s": s["load_s"], "first_call_ms": s["first_call_ms"],
                          "warmup_s": s["warmup_s"], "session_min": s["total_s"] / 60} for s in sessions])
    replay = latency[latency["workload"] == "replay"].copy()
    replay_view = pd.DataFrame({
        "dataset": replay["cell"],
        "samples": replay["samples"],
        "tokens/pair": replay["pair_tokens"].round(0).astype(int),
        "padding": replay["padding_ratio"].map("{:.1%}".format),
        "prepare p50": replay["prepare_ms_p50"],
        "predict p50 [95% CI]": [f"{r.predict_ms_p50:.1f} [{r.predict_ms_p50_ci_low:.1f}, {r.predict_ms_p50_ci_high:.1f}]"
                                 for r in replay.itertuples()],
        "predict p95": replay["predict_ms_p95"],
        "predict p99": replay["predict_ms_p99"],
        "e2e p50": replay["e2e_ms_p50"],
        "e2e p95": replay["e2e_ms_p95"],
        "pairs/s": replay["pairs_per_s"].round(0).astype(int),
        "session CV": replay["session_p50_cv"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "–"),
    })
    lines = [
        f"# {model['name']} v{model['version']} — latency benchmark (profile `{run_info['profile']['name']}`)",
        "",
        f"- git: `{env['git'].get('describe')}` (commit `{env['git'].get('commit')}`)",
        f"- gpu: {env.get('gpu', {}).get('name')} ({env.get('gpu', {}).get('arch')}), torch {env['packages']['torch']}",
        f"- sessions: {len(sessions)} (separate processes), total {run_info['timing']['total_s'] / 60:.1f} min",
        "- all times in ms; `predict` = device stage incl. synchronization, `e2e` = prepare + predict",
        "",
        "## Cold start",
        markdown_table(cold, float_format="{:.1f}"),
        "",
        "## Synthetic grid: predict p50, ms (one query with N candidates of exactly L tokens, no padding)",
        markdown_table(_pivot(latency, "predict_ms_p50", "{:.1f}")),
        "",
        "## Synthetic grid: predict p95, ms",
        markdown_table(_pivot(latency, "predict_ms_p95", "{:.1f}")),
        "",
        "## Synthetic grid: throughput, pairs/s",
        markdown_table(_pivot(latency, "pairs_per_s", "{:.0f}")),
        "",
        "## Synthetic grid: session-to-session CV of p50 (measurement noise)",
        markdown_table(_pivot(latency, "session_p50_cv", "{:.1%}")),
        "",
        "## Synthetic grid: first (cold) call per shape, max over sessions, ms",
        markdown_table(_pivot(latency, "cold_ms_max", "{:.0f}")),
        "",
        "## Replay: real queries (100 candidates each, real lengths and padding)",
        markdown_table(replay_view, float_format="{:.1f}"),
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return latency
