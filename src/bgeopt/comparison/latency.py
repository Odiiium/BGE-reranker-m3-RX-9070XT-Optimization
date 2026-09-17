"""Compare two latency benchmark runs (results/latency/<run>): speedup per synthetic cell and replay dataset.

speedup = baseline / candidate (> 1 means the candidate is faster).

Sessions (separate worker processes) are the independent units. On this machine light cells (< ~60 ms) switch
between discrete per-process states (e.g. quora 44 ms vs 54 ms) while calls inside a session are stable, so a
bootstrap over individual calls is overconfident (A/A tests flagged 10-20% "changes"). The speedup and its 95% CI
therefore come from a Welch t-interval on log per-session p50 (ratio of geometric means). With a single session on
either side it falls back to a call-level bootstrap, flagged in `ci_method`.
A change is reported when the CI excludes 1 AND |effect| >= min_effect.
Runs from different profiles are not comparable (warmup length and sample counts shift light cells by 10-20%).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bgeopt.utils.stats import ratio_of_medians_ci, session_ratio_ci


@dataclass
class LatencyThresholds:
    bootstrap_samples: int = 2000
    seed: int = 0
    min_effect: float = 0.03


@dataclass
class LatencyRun:
    path: Path
    info: dict[str, Any]
    samples: pd.DataFrame
    latency: pd.DataFrame

    @classmethod
    def load(cls, path: str | Path) -> LatencyRun:
        path = Path(path)
        if not (path / "latency.csv").exists():
            raise ValueError(f"{path} is not a latency benchmark run (no latency.csv)")
        return cls(path=path, info=json.loads((path / "run_info.json").read_text(encoding="utf-8")),
                   samples=pd.read_parquet(path / "samples.parquet"), latency=pd.read_csv(path / "latency.csv"))

    @property
    def label(self) -> str:
        git = self.info.get("environment", {}).get("git", {})
        return (f"{self.info['model']['config']['name']} ({git.get('describe') or 'no git'}, "
                f"profile {self.info['profile']['name']}) — {self.path.name}")


@dataclass
class LatencyComparison:
    baseline: LatencyRun
    candidate: LatencyRun
    speedups: pd.DataFrame
    summary: dict[str, float]
    warnings: list[str] = field(default_factory=list)


def _geomean(values: pd.Series) -> float:
    values = values.dropna()
    return float(np.exp(np.log(values).mean())) if len(values) else float("nan")


def same_invocation(a: LatencyRun, b: LatencyRun) -> bool:
    return a.path.name.split("_")[0] == b.path.name.split("_")[0] and a.info.get("session_order") == b.info.get("session_order")


def compare_latency(baseline: LatencyRun, candidate: LatencyRun, thresholds: LatencyThresholds,
                    allow_profile_mismatch: bool = False) -> LatencyComparison:
    rng = np.random.default_rng(thresholds.seed)
    warnings = []
    profile_a, profile_b = baseline.info["profile"]["name"], candidate.info["profile"]["name"]
    if profile_a != profile_b:
        message = f"profiles differ ({profile_a} vs {profile_b}): light cells shift by 10-20% from warmup/sample counts"
        if not allow_profile_mismatch:
            raise ValueError(message + "; benchmark both versions with the same profile")
        warnings.append(message)
    if same_invocation(baseline, candidate):
        warnings.append("same benchmark invocation (interleaved sessions): drift between occasions is controlled")
    else:
        warnings.append("separate benchmark invocations: drift between occasions is only covered by the noise threshold; "
                        "for decisions benchmark both versions in one invocation")
    sha_a = baseline.info["model"].get("weights_sha256")
    sha_b = candidate.info["model"].get("weights_sha256")
    if sha_a != sha_b:
        warnings.append(f"model weights differ: {sha_a} vs {sha_b}")

    keys_a = set(map(tuple, baseline.latency[["workload", "cell"]].to_numpy()))
    keys_b = set(map(tuple, candidate.latency[["workload", "cell"]].to_numpy()))
    if keys_a != keys_b:
        warnings.append(f"cells present in only one run are skipped: {sorted(keys_a ^ keys_b)}")

    rows = []
    for row in baseline.latency.itertuples():
        key = (row.workload, row.cell)
        if key not in keys_b:
            continue
        a = baseline.samples[(baseline.samples["workload"] == row.workload) & (baseline.samples["cell"] == row.cell)]
        b = candidate.samples[(candidate.samples["workload"] == row.workload) & (candidate.samples["cell"] == row.cell)]
        if row.workload == "replay" and set(a["query_id"]) != set(b["query_id"]):
            warnings.append(f"replay {row.cell}: different query samples, speedup mixes query sets")
        other = candidate.latency[(candidate.latency["workload"] == row.workload) & (candidate.latency["cell"] == row.cell)].iloc[0]

        pred_a, pred_b = a["predict_ms"].to_numpy(), b["predict_ms"].to_numpy()
        e2e_a, e2e_b = a["e2e_ms"].to_numpy(), b["e2e_ms"].to_numpy()
        session_ci = session_ratio_ci(a.groupby("session")["predict_ms"].median().to_numpy(),
                                      b.groupby("session")["predict_ms"].median().to_numpy())
        e2e_session_ci = session_ratio_ci(a.groupby("session")["e2e_ms"].median().to_numpy(),
                                          b.groupby("session")["e2e_ms"].median().to_numpy())
        if session_ci is not None and e2e_session_ci is not None:
            speedup, *ci = session_ci
            e2e_speedup, *e2e_ci = e2e_session_ci
            ci_method = "sessions"
        else:
            speedup = float(np.median(pred_a) / np.median(pred_b))
            ci = ratio_of_medians_ci(pred_a, pred_b, thresholds.bootstrap_samples, rng)
            e2e_speedup = float(np.median(e2e_a) / np.median(e2e_b))
            e2e_ci = ratio_of_medians_ci(e2e_a, e2e_b, thresholds.bootstrap_samples, rng)
            ci_method = "calls"
        large_enough = abs(np.log(speedup)) >= np.log1p(thresholds.min_effect)
        rows.append({
            "workload": row.workload, "cell": row.cell, "candidates": row.candidates, "pair_tokens": row.pair_tokens,
            "baseline_predict_p50": float(np.median(pred_a)), "candidate_predict_p50": float(np.median(pred_b)),
            "speedup_p50": speedup, "speedup_p50_ci_low": ci[0], "speedup_p50_ci_high": ci[1],
            "speedup_p95": float(np.quantile(pred_a, 0.95) / np.quantile(pred_b, 0.95)),
            "speedup_p99": float(np.quantile(pred_a, 0.99) / np.quantile(pred_b, 0.99)),
            "e2e_speedup_p50": e2e_speedup,
            "e2e_speedup_p50_ci_low": e2e_ci[0], "e2e_speedup_p50_ci_high": e2e_ci[1],
            "ci_method": ci_method,
            "change": ("faster" if ci[0] > 1 and large_enough else "slower" if ci[1] < 1 and large_enough
                       else "no change"),
            "baseline_session_cv": row.session_p50_cv, "candidate_session_cv": other["session_p50_cv"],
            "baseline_peak_memory_mb": row.peak_memory_mb, "candidate_peak_memory_mb": other["peak_memory_mb"],
        })
    speedups = pd.DataFrame(rows)
    if (speedups["ci_method"] == "calls").any():
        warnings.append("some cells have < 2 sessions in a run: CIs fall back to call-level bootstrap and are "
                        "overconfident; use >= 2 sessions (quick) or 3 (default) for decisions")
    replay = speedups[speedups["workload"] == "replay"]
    synthetic = speedups[speedups["workload"] == "synthetic"]
    summary = {
        "replay_geomean_speedup_p50": _geomean(replay["speedup_p50"]),
        "replay_geomean_speedup_p95": _geomean(replay["speedup_p95"]),
        "replay_geomean_e2e_speedup_p50": _geomean(replay["e2e_speedup_p50"]),
        "synthetic_geomean_speedup_p50": _geomean(synthetic["speedup_p50"]),
        "cells_faster": int((speedups["change"] == "faster").sum()),
        "cells_slower": int((speedups["change"] == "slower").sum()),
        "cells_unchanged": int((speedups["change"] == "no change").sum()),
    }
    return LatencyComparison(baseline=baseline, candidate=candidate, speedups=speedups, summary=summary,
                             warnings=warnings)
