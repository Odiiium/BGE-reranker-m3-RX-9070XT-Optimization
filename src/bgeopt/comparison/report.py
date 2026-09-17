"""Writes comparison outputs to results/compare/<baseline>__vs__<candidate>/."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bgeopt.comparison.latency import LatencyComparison
from bgeopt.comparison.quality import QualityComparison
from bgeopt.evaluation.report import markdown_table

STATUS_MARK = {"equivalent": "≈", "better": "▲", "worse": "▼", "inconclusive": "?"}


def _header(title: str, baseline: str, candidate: str, warnings: list[str]) -> list[str]:
    lines = [f"# {title}", "", f"- baseline: {baseline}", f"- candidate: {candidate}"]
    lines += [f"- ⚠ {w}" for w in warnings]
    return lines + [""]


def write_quality_comparison(out_dir: Path, comparison: QualityComparison) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison.deltas.to_csv(out_dir / "quality_deltas.csv", index=False)
    comparison.parity.to_csv(out_dir / "parity.csv", index=False)
    comparison.coverage.to_csv(out_dir / "coverage.csv", index=False)
    t = comparison.thresholds
    (out_dir / "comparison.json").write_text(json.dumps({
        "type": "quality", "baseline": str(comparison.baseline.path), "candidate": str(comparison.candidate.path),
        "verdict": comparison.verdict, "thresholds": t.__dict__, "warnings": comparison.warnings,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    deltas = comparison.deltas
    macro = deltas[deltas["dataset"] == "MEAN"]
    macro_view = pd.DataFrame({
        "metric": macro["metric"],
        "baseline": macro["baseline"],
        "candidate": macro["candidate"],
        "delta": macro["delta"].map("{:+.4f}".format),
        "90% CI (equivalence)": [f"[{lo:+.4f}, {hi:+.4f}]" for lo, hi in zip(macro["ci90_low"], macro["ci90_high"])],
        "95% CI": [f"[{lo:+.4f}, {hi:+.4f}]" for lo, hi in zip(macro["ci95_low"], macro["ci95_high"])],
        "status": macro["status"],
    })
    per_dataset = deltas[deltas["dataset"] != "MEAN"].copy()
    per_dataset["cell"] = [f"{d:+.4f} {STATUS_MARK[s]}" for d, s in zip(per_dataset["delta"], per_dataset["status"])]
    pivot = per_dataset.pivot(index="dataset", columns="metric", values="cell")
    pivot = pivot.reindex(index=list(dict.fromkeys(per_dataset["dataset"])),
                          columns=list(dict.fromkeys(per_dataset["metric"]))).reset_index()

    ks = [c for c in comparison.parity.columns if c.startswith("overlap@")]
    parity_view = comparison.parity[["dataset", "queries", "pairs", "max_abs_dlogit", "mean_abs_dlogit",
                                     "identical_scores", "kendall_tau_median", "kendall_tau_p5", *ks,
                                     *[c for c in comparison.parity.columns if c.startswith("identical_top")]]].copy()
    parity_view["identical_scores"] = parity_view["identical_scores"].map("{:.1%}".format)

    lines = _header("Quality comparison", comparison.baseline.label, comparison.candidate.label, comparison.warnings)
    lines += [
        f"**Quality: {comparison.verdict['quality']}** — equivalent when the 90% paired bootstrap CI of Δ lies within "
        f"±{t.equivalence_margin} ({t.bootstrap_samples} resamples); otherwise ▼ worse / ▲ better when the 95% CI "
        "excludes 0, ? inconclusive. PASS = every macro metric equivalent or better; FAIL = a macro metric worse or a "
        "dataset worse by more than the margin. Per-dataset flags are not corrected for multiple comparisons.",
        "",
        f"**Numerical parity: {comparison.verdict['parity']}** — gates: median Kendall τ ≥ {t.min_kendall_tau_median}, "
        f"overlap@10 ≥ {t.min_overlap_at_10}",
        "",
        "## Macro mean over datasets",
        markdown_table(macro_view),
        "",
        "## Δ per dataset (candidate − baseline)",
        markdown_table(pivot),
        "",
        "## Numerical parity of raw scores",
        markdown_table(parity_view, float_format="{:.4g}"),
        "",
        "## Coverage",
        markdown_table(comparison.coverage),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return out_dir / "summary.md"


def write_latency_comparison(out_dir: Path, comparison: LatencyComparison) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison.speedups.to_csv(out_dir / "latency_speedups.csv", index=False)
    (out_dir / "comparison.json").write_text(json.dumps({
        "type": "latency", "baseline": str(comparison.baseline.path), "candidate": str(comparison.candidate.path),
        "summary": comparison.summary, "warnings": comparison.warnings,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    s = comparison.speedups
    s = s.assign(text=[f"{v:.2f}× [{lo:.2f}, {hi:.2f}]" for v, lo, hi in
                       zip(s["speedup_p50"], s["speedup_p50_ci_low"], s["speedup_p50_ci_high"])])
    synthetic = s[s["workload"] == "synthetic"].copy()
    synthetic["pair_tokens"] = synthetic["pair_tokens"].round().astype(int)
    grid = synthetic.pivot(index="candidates", columns="pair_tokens", values="text")
    grid.columns = [f"L={c}" for c in grid.columns]
    grid = grid.reset_index().rename(columns={"candidates": "candidates \\ tokens"})

    replay = s[s["workload"] == "replay"]
    replay_view = pd.DataFrame({
        "dataset": replay["cell"],
        "baseline p50": replay["baseline_predict_p50"],
        "candidate p50": replay["candidate_predict_p50"],
        "speedup p50 [95% CI]": replay["text"],
        "speedup p95": replay["speedup_p95"].map("{:.2f}×".format),
        "e2e speedup p50": replay["e2e_speedup_p50"].map("{:.2f}×".format),
        "change": replay["change"],
    })
    m = comparison.summary
    lines = _header("Latency comparison (speedup = baseline / candidate)", comparison.baseline.label,
                    comparison.candidate.label, comparison.warnings)
    lines += [
        f"- replay geometric-mean speedup: p50 **{m['replay_geomean_speedup_p50']:.2f}×**, "
        f"p95 {m['replay_geomean_speedup_p95']:.2f}×, end-to-end p50 {m['replay_geomean_e2e_speedup_p50']:.2f}×",
        f"- synthetic geometric-mean speedup p50: {m['synthetic_geomean_speedup_p50']:.2f}×",
        f"- cells: {m['cells_faster']} faster, {m['cells_slower']} slower, {m['cells_unchanged']} unchanged "
        "(changed = 95% CI of the p50 speedup excludes 1 and |effect| ≥ 3%; CI = Welch t-interval over per-session "
        "p50, sessions are the independent units)",
        "",
        "## Synthetic grid: predict p50 speedup [95% CI over sessions]",
        markdown_table(grid),
        "",
        "## Replay: real queries",
        markdown_table(replay_view, float_format="{:.1f}"),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return out_dir / "summary.md"
