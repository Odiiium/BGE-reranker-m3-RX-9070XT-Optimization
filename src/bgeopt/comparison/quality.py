"""Compare two quality runs (results/quality/<run>): metric equivalence and numerical parity of raw scores.

Two questions are answered separately:
1. Retrieval quality — per (dataset, metric@k), is the candidate equivalent to the baseline?
   Paired bootstrap over queries gives CIs of the mean difference; TOST equivalence = 90% CI within +-margin;
   a sign-flip randomization test gives the p-value of "no difference".
2. Numerical parity — on identical (query, candidate) pairs, how far did the scores / rankings drift?
   |delta logit|, per-query Kendall tau-b (tie-aware: fp16 logits are quantized) and top-k overlap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from bgeopt.metrics import rank_documents
from bgeopt.utils.stats import bootstrap_means, percentile_ci, sign_flip_p_value

STAGE_PREFIX = "reranked/"


@dataclass
class QualityThresholds:
    equivalence_margin: float = 0.005
    bootstrap_samples: int = 2000
    permutations: int = 5000
    seed: int = 0
    min_kendall_tau_median: float = 0.99
    min_overlap_at_10: float = 0.98


@dataclass
class QualityRun:
    path: Path
    info: dict[str, Any]
    datasets: list[str]

    @classmethod
    def load(cls, path: str | Path) -> QualityRun:
        path = Path(path)
        info = json.loads((path / "run_info.json").read_text(encoding="utf-8"))
        if "eval_config" not in info:
            raise ValueError(f"{path} is not a quality run (run_info.json has no eval_config)")
        return cls(path=path, info=info, datasets=list(info["datasets"]))

    @property
    def label(self) -> str:
        git = self.info.get("environment", {}).get("git", {})
        return f"{self.info['model']['config']['name']} ({git.get('describe') or 'no git'}) — {self.path.name}"

    def per_query(self, dataset: str) -> pd.DataFrame:
        return pd.read_parquet(self.path / "per_query" / f"{dataset}.parquet").set_index("query_id")

    def scores(self, dataset: str) -> pd.DataFrame:
        return pd.read_parquet(self.path / "scores" / f"{dataset}.parquet", columns=["query_id", "doc_id", "rerank_score"])


@dataclass
class QualityComparison:
    baseline: QualityRun
    candidate: QualityRun
    thresholds: QualityThresholds
    coverage: pd.DataFrame  # dataset, baseline_queries, candidate_queries, common_queries
    deltas: pd.DataFrame  # dataset (incl. MEAN), metric label, baseline, candidate, delta, CIs, p_value, status
    parity: pd.DataFrame  # dataset, drift of raw scores and rankings
    verdict: dict[str, str]
    warnings: list[str] = field(default_factory=list)


def _status(ci90: tuple[float, float], ci95: tuple[float, float], margin: float) -> str:
    """Equivalence is tested first: a statistically detectable but practically negligible shift (a few queries
    flipping in the same direction, typical for fp16 kernel swaps) is still equivalent."""
    if -margin <= ci90[0] and ci90[1] <= margin:
        return "equivalent"
    if ci95[1] < 0:
        return "worse"
    if ci95[0] > 0:
        return "better"
    return "inconclusive"


def _ranking_parity(merged: pd.DataFrame, ks: list[int]) -> dict[str, float]:
    """merged: query_id, doc_id, score_a, score_b for common pairs."""
    merged = merged.sort_values(["query_id", "doc_id"], kind="stable")
    boundaries = np.flatnonzero(merged["query_id"].to_numpy()[1:] != merged["query_id"].to_numpy()[:-1]) + 1
    doc_ids = np.split(merged["doc_id"].to_numpy(), boundaries)
    score_a = np.split(merged["score_a"].to_numpy(), boundaries)
    score_b = np.split(merged["score_b"].to_numpy(), boundaries)

    taus, overlaps, identical = [], {k: [] for k in ks}, {k: [] for k in ks}
    for docs, a, b in zip(doc_ids, score_a, score_b):
        if np.array_equal(a, b):
            taus.append(1.0)
        else:
            tau = kendalltau(a, b).statistic
            taus.append(1.0 if np.isnan(tau) else float(tau))
        ranked_a = rank_documents(dict(zip(docs, a)))  # same order (and tie-break) the metrics use
        ranked_b = rank_documents(dict(zip(docs, b)))
        for k in ks:
            top_a, top_b = ranked_a[:k], ranked_b[:k]
            overlaps[k].append(len(set(top_a) & set(top_b)) / max(min(k, len(docs)), 1))
            identical[k].append(float(top_a == top_b))

    taus_arr = np.asarray(taus)
    row = {"kendall_tau_median": float(np.median(taus_arr)), "kendall_tau_p5": float(np.quantile(taus_arr, 0.05)),
           "kendall_tau_min": float(taus_arr.min())}
    for k in ks:
        row[f"overlap@{k}"] = float(np.mean(overlaps[k]))
        row[f"identical_top{k}"] = float(np.mean(identical[k]))
    return row


def compare_quality(baseline: QualityRun, candidate: QualityRun, thresholds: QualityThresholds) -> QualityComparison:
    warnings: list[str] = []
    common = [d for d in baseline.datasets if d in candidate.datasets]
    if not common:
        raise ValueError("runs share no datasets")
    for name in common:
        fa = baseline.info["datasets"][name]["fingerprint"]
        fb = candidate.info["datasets"][name]["fingerprint"]
        if fa != fb:
            raise ValueError(f"{name}: slices differ (fingerprint {fa} vs {fb}); rerun one of the versions")
    sha_a = baseline.info["model"].get("weights_sha256")
    sha_b = candidate.info["model"].get("weights_sha256")
    if sha_a != sha_b:
        warnings.append(f"model weights differ: {sha_a} vs {sha_b}")
    skipped = sorted(set(baseline.datasets) ^ set(candidate.datasets))
    if skipped:
        warnings.append(f"datasets present in only one run are skipped: {', '.join(skipped)}")

    rng = np.random.default_rng(thresholds.seed)
    margin = thresholds.equivalence_margin
    coverage_rows, delta_rows, parity_rows = [], [], []
    boot_by_label: dict[str, list[np.ndarray]] = {}
    means_by_label: dict[str, list[tuple[float, float]]] = {}
    ks: list[int] = []

    for name in common:
        pq_a, pq_b = baseline.per_query(name), candidate.per_query(name)
        queries = pq_a.index.intersection(pq_b.index).sort_values()
        coverage_rows.append({"dataset": name, "baseline_queries": len(pq_a), "candidate_queries": len(pq_b),
                              "common_queries": len(queries)})
        if len(queries) < max(len(pq_a), len(pq_b)):
            warnings.append(f"{name}: compared on {len(queries)} common queries "
                            f"(baseline {len(pq_a)}, candidate {len(pq_b)})")
        if len(queries) == 0:
            continue

        labels = [c[len(STAGE_PREFIX):] for c in pq_a.columns if c.startswith(STAGE_PREFIX) and c in pq_b.columns]
        ks = sorted({int(label.split("@")[1]) for label in labels})
        indices = rng.integers(0, len(queries), size=(thresholds.bootstrap_samples, len(queries)))
        signs = rng.choice(np.array([-1.0, 1.0]), size=(thresholds.permutations, len(queries)))

        for label in labels:
            a = pq_a.loc[queries, STAGE_PREFIX + label].to_numpy(dtype=np.float64)
            b = pq_b.loc[queries, STAGE_PREFIX + label].to_numpy(dtype=np.float64)
            diffs = b - a
            boot = bootstrap_means(diffs, indices)
            boot_by_label.setdefault(label, []).append(boot)
            means_by_label.setdefault(label, []).append((float(a.mean()), float(b.mean())))
            ci95, ci90 = percentile_ci(boot, 0.95), percentile_ci(boot, 0.90)
            delta_rows.append({
                "dataset": name, "metric": label, "queries": len(queries), "baseline": a.mean(), "candidate": b.mean(),
                "delta": diffs.mean(), "ci95_low": ci95[0], "ci95_high": ci95[1], "ci90_low": ci90[0],
                "ci90_high": ci90[1], "p_value": sign_flip_p_value(diffs, signs),
                "changed_queries": int(np.count_nonzero(diffs)), "status": _status(ci90, ci95, margin),
            })

        merged = baseline.scores(name).merge(candidate.scores(name), on=["query_id", "doc_id"], suffixes=("_a", "_b"))
        merged = merged[merged["query_id"].isin(set(queries))].rename(
            columns={"rerank_score_a": "score_a", "rerank_score_b": "score_b"})
        abs_diff = (merged["score_b"] - merged["score_a"]).abs()
        parity_rows.append({
            "dataset": name, "queries": len(queries), "pairs": len(merged),
            "max_abs_dlogit": abs_diff.max(), "mean_abs_dlogit": abs_diff.mean(), "p99_abs_dlogit": abs_diff.quantile(0.99),
            "identical_scores": float((abs_diff == 0).mean()),
            **_ranking_parity(merged, ks),
        })

    # macro mean over datasets: bootstrap draws are independent per dataset, so they combine draw-by-draw
    for label, boots in boot_by_label.items():
        macro_boot = np.mean(np.stack(boots), axis=0)
        ci95, ci90 = percentile_ci(macro_boot, 0.95), percentile_ci(macro_boot, 0.90)
        base_mean = float(np.mean([m[0] for m in means_by_label[label]]))
        cand_mean = float(np.mean([m[1] for m in means_by_label[label]]))
        delta_rows.append({
            "dataset": "MEAN", "metric": label, "queries": int(sum(r["common_queries"] for r in coverage_rows)),
            "baseline": base_mean, "candidate": cand_mean, "delta": cand_mean - base_mean,
            "ci95_low": ci95[0], "ci95_high": ci95[1], "ci90_low": ci90[0], "ci90_high": ci90[1],
            "p_value": np.nan, "changed_queries": np.nan, "status": _status(ci90, ci95, margin),
        })

    deltas = pd.DataFrame(delta_rows)
    parity = pd.DataFrame(parity_rows)

    macro = deltas[deltas["dataset"] == "MEAN"]
    per_dataset = deltas[deltas["dataset"] != "MEAN"]
    harmful = per_dataset[(per_dataset["status"] == "worse") & (per_dataset["delta"] < -margin)]
    if (macro["status"] == "worse").any() or len(harmful):
        quality_verdict = "FAIL"
    elif macro["status"].isin(["equivalent", "better"]).all():
        quality_verdict = "PASS"
    else:
        quality_verdict = "INCONCLUSIVE"

    parity_ok = (parity["kendall_tau_median"] >= thresholds.min_kendall_tau_median).all()
    if "overlap@10" in parity:
        parity_ok = parity_ok and (parity["overlap@10"] >= thresholds.min_overlap_at_10).all()
    verdict = {"quality": quality_verdict, "parity": "PASS" if parity_ok else "FAIL"}

    return QualityComparison(baseline=baseline, candidate=candidate, thresholds=thresholds,
                             coverage=pd.DataFrame(coverage_rows), deltas=deltas, parity=parity,
                             verdict=verdict, warnings=warnings)
