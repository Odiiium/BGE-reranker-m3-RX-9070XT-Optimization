"""Writes evaluation outputs.

results/<run>/
    run_info.json            model config + weights hash, dataset fingerprints, environment, timing vs budget
    summary.md               human-readable tables (reranked, first stage, gain, efficiency)
    metrics.csv              long format: dataset, language, stage, metric, k, value, queries
    metrics_wide.csv         one row per (dataset, stage), one column per metric@k
    efficiency.csv           latency / throughput / memory per dataset
    per_query/<dataset>.parquet   per-query metrics of both stages + timings
    scores/<dataset>.parquet      raw scores per candidate (for parity checks between versions)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from bgeopt.evaluation.runner import DatasetEvaluation

STAGES = ("reranked", "first_stage")


def markdown_table(frame: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    def cell(value: Any) -> str:
        if isinstance(value, float):
            return float_format.format(value)
        return str(value)

    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    divider = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def write_dataset_outputs(run_dir: Path, result: DatasetEvaluation) -> None:
    name = result.dataset.name
    (run_dir / "per_query").mkdir(exist_ok=True)
    (run_dir / "scores").mkdir(exist_ok=True)
    per_query = (
        result.timings.set_index("query_id")
        .join(result.first_stage.per_query.add_prefix("first_stage/"))
        .join(result.reranked.per_query.add_prefix("reranked/"))
    )
    per_query.reset_index().to_parquet(run_dir / "per_query" / f"{name}.parquet", index=False)
    result.scores.to_parquet(run_dir / "scores" / f"{name}.parquet", index=False)


def _wide(results: list[DatasetEvaluation], stage: str) -> pd.DataFrame:
    rows = [{"dataset": r.dataset.name, **getattr(r, stage).mean} for r in results]
    frame = pd.DataFrame(rows)
    mean_row = {"dataset": "MEAN", **frame.drop(columns="dataset").mean().to_dict()}
    return pd.concat([frame, pd.DataFrame([mean_row])], ignore_index=True)


def write_run_summary(run_dir: Path, results: list[DatasetEvaluation], run_info: dict[str, Any]) -> None:
    long_rows = [
        {"dataset": r.dataset.name, "language": r.dataset.spec.language, "stage": stage,
         "metric": label.split("@")[0], "k": int(label.split("@")[1]), "value": value, "queries": r.dataset.num_queries}
        for r in results for stage in STAGES for label, value in getattr(r, stage).mean.items()
    ]
    pd.DataFrame(long_rows).to_csv(run_dir / "metrics.csv", index=False)

    wide = {stage: _wide(results, stage) for stage in STAGES}
    pd.concat([wide[s].assign(stage=s) for s in STAGES], ignore_index=True).to_csv(run_dir / "metrics_wide.csv",
                                                                                     index=False)
    gain = wide["reranked"].copy()
    metric_cols = [c for c in gain.columns if c != "dataset"]
    gain[metric_cols] = wide["reranked"][metric_cols] - wide["first_stage"][metric_cols]

    efficiency = pd.DataFrame([r.efficiency() for r in results])
    efficiency.to_csv(run_dir / "efficiency.csv", index=False)

    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False, default=str),
                                           encoding="utf-8")

    model = run_info["model"]["config"]
    env = run_info["environment"]
    timing = run_info["timing"]
    eff_view = efficiency[["dataset", "queries", "pairs", "padding_ratio", "prepare_ms_mean", "predict_ms_p50",
                           "predict_ms_p95", "pairs_per_s", "wall_s", "peak_memory_mb"]].copy()
    eff_view["padding_ratio"] = eff_view["padding_ratio"].map("{:.1%}".format)
    eff_view["pairs_per_s"] = eff_view["pairs_per_s"].round(0).astype(int)

    budget_note = "within budget" if timing["within_budget"] else "**OVER BUDGET**"
    lines = [
        f"# {model['name']} v{model['version']}",
        "",
        f"- git: `{env['git'].get('describe')}` (commit `{env['git'].get('commit')}`)",
        f"- gpu: {env.get('gpu', {}).get('name')} ({env.get('gpu', {}).get('arch')}), torch {env['packages']['torch']}, "
        f"hip {env.get('torch_hip')}",
        f"- model: `{model.get('model_id')}` @ `{model.get('revision')}`, weights sha256 "
        f"`{run_info['model'].get('weights_sha256')}`",
        f"- time: {timing['total_s'] / 60:.1f} min total (setup {timing['setup_s']:.0f}s, warmup "
        f"{timing['warmup_s']:.0f}s) / budget {timing['budget_s'] / 60:.0f} min — {budget_note}",
        "",
        "## Reranked",
        markdown_table(wide["reranked"]),
        "",
        "## First stage (candidates as retrieved)",
        markdown_table(wide["first_stage"]),
        "",
        "## Gain (reranked − first stage)",
        markdown_table(gain, float_format="{:+.4f}"),
        "",
        "## Efficiency (per query = all candidates of one query)",
        markdown_table(eff_view, float_format="{:.1f}"),
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
