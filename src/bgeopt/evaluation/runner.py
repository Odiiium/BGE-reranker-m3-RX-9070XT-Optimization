"""Runs a reranker over dataset slices: first-stage metrics, reranking, metrics at every cutoff, timings."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from bgeopt.data import DatasetLoader, RerankDataset
from bgeopt.evaluation.report import write_dataset_outputs, write_run_summary
from bgeopt.metrics import EvaluationResult, RetrievalEvaluator
from bgeopt.models import BaseReranker
from bgeopt.utils import capture_environment

log = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    datasets: list[str]
    metrics: list[str] = field(default_factory=lambda: ["ndcg", "mrr", "recall"])
    ks: list[int] = field(default_factory=lambda: [1, 5, 10])
    time_budget_minutes: float = 45.0
    datasets_config: str = "configs/datasets.yaml"
    data_dir: str = "data"
    output_dir: str = "results"

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalConfig:
        return cls(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))


@dataclass
class DatasetEvaluation:
    dataset: RerankDataset
    first_stage: EvaluationResult
    reranked: EvaluationResult
    timings: pd.DataFrame  # one row per query
    scores: pd.DataFrame  # one row per (query, candidate)
    wall_seconds: float
    peak_memory_mb: float | None

    def efficiency(self) -> dict[str, Any]:
        t = self.timings
        predict_s = t["predict_ms"].sum() / 1e3
        end_to_end = t["prepare_ms"] + t["predict_ms"]
        tokens, padded = int(t["num_tokens"].sum()), int(t["num_padded_tokens"].sum())
        return {
            "dataset": self.dataset.name,
            "queries": len(t),
            "pairs": int(t["num_pairs"].sum()),
            "tokens": tokens,
            "padding_ratio": 1 - tokens / padded if padded else 0.0,
            "prepare_ms_mean": t["prepare_ms"].mean(),
            "predict_ms_p50": t["predict_ms"].quantile(0.50),
            "predict_ms_p95": t["predict_ms"].quantile(0.95),
            "predict_ms_p99": t["predict_ms"].quantile(0.99),
            "e2e_ms_p50": end_to_end.quantile(0.50),
            "e2e_ms_p95": end_to_end.quantile(0.95),
            "pairs_per_s": t["num_pairs"].sum() / predict_s if predict_s else 0.0,
            "tokens_per_s": tokens / predict_s if predict_s else 0.0,
            "wall_s": self.wall_seconds,
            "peak_memory_mb": self.peak_memory_mb,
        }


class RerankingEvaluationRunner:
    def __init__(self, reranker: BaseReranker, loader: DatasetLoader, config: EvalConfig, repo_root: Path,
                 max_queries: int | None = None, run_tag: str | None = None):
        self.reranker = reranker
        self.loader = loader
        self.config = config
        self.repo_root = repo_root
        self.max_queries = max_queries
        self.run_tag = run_tag

    def evaluate_dataset(self, dataset: RerankDataset) -> DatasetEvaluation:
        evaluator = RetrievalEvaluator(self.config.metrics, self.config.ks, dataset.spec.relevance_threshold)
        first_stage = evaluator.evaluate(dataset.first_stage_run(), dataset.qrels)

        self.reranker.reset_peak_memory()
        started = time.perf_counter()
        run: dict[str, dict[str, float]] = {}
        timing_rows, score_rows = [], []
        for query in tqdm(dataset.queries, desc=dataset.name, unit="q", leave=False):
            t0 = time.perf_counter()
            prepared = self.reranker.prepare(query.query, query.doc_texts)
            t1 = time.perf_counter()
            scores = self.reranker.predict(prepared)
            t2 = time.perf_counter()

            if len(scores) != len(query.doc_ids) or not np.all(np.isfinite(scores)):
                raise RuntimeError(f"{dataset.name}/{query.query_id}: invalid scores from {self.reranker.name}")
            run[query.query_id] = dict(zip(query.doc_ids, scores.tolist()))
            timing_rows.append({
                "query_id": query.query_id, "prepare_ms": (t1 - t0) * 1e3, "predict_ms": (t2 - t1) * 1e3,
                "num_pairs": prepared.num_pairs, "num_tokens": prepared.num_tokens,
                "num_padded_tokens": prepared.num_padded_tokens,
            })
            score_rows.extend(
                (query.query_id, doc_id, rank, fs, float(s))
                for rank, (doc_id, fs, s) in enumerate(zip(query.doc_ids, query.first_stage_scores, scores), start=1)
            )
        wall = time.perf_counter() - started

        return DatasetEvaluation(
            dataset=dataset,
            first_stage=first_stage,
            reranked=evaluator.evaluate(run, dataset.qrels),
            timings=pd.DataFrame(timing_rows),
            scores=pd.DataFrame(score_rows, columns=["query_id", "doc_id", "first_stage_rank",
                                                     "first_stage_score", "rerank_score"]),
            wall_seconds=wall,
            peak_memory_mb=self.reranker.peak_memory_mb(),
        )

    def run(self, setup_seconds: float = 0.0) -> Path:
        suite_started = time.perf_counter() - setup_seconds  # model loading counts towards the budget
        budget_s = self.config.time_budget_minutes * 60

        datasets = [self.loader.load(name, max_queries=self.max_queries) for name in self.config.datasets]
        total_chars = sum(d.num_chars for d in datasets)
        log.info("suite: %d datasets, %d queries, %d pairs, budget %.0f min",
                 len(datasets), sum(d.num_queries for d in datasets), sum(d.num_pairs for d in datasets),
                 self.config.time_budget_minutes)

        stamp = time.strftime("%Y%m%d-%H%M%S")
        suffix = f"_{self.run_tag}" if self.run_tag else ""
        run_dir = Path(self.config.output_dir) / f"{stamp}_{self.reranker.name}{suffix}"
        run_dir.mkdir(parents=True, exist_ok=False)

        warmup_started = time.perf_counter()
        self.reranker.warmup()
        warmup_seconds = time.perf_counter() - warmup_started

        results: list[DatasetEvaluation] = []
        done_chars, eval_seconds = 0, 0.0
        for dataset in datasets:
            result = self.evaluate_dataset(dataset)
            write_dataset_outputs(run_dir, result)
            results.append(result)

            done_chars += dataset.num_chars
            eval_seconds += result.wall_seconds
            elapsed = time.perf_counter() - suite_started
            eta = elapsed + (total_chars - done_chars) * eval_seconds / max(done_chars, 1)
            log.info("%-14s nDCG@10 %.4f -> %.4f | %6.1fs, %5.0f pairs/s | elapsed %.1f min, projected %.1f / %.0f min",
                     dataset.name, result.first_stage.mean.get("ndcg@10", float("nan")),
                     result.reranked.mean.get("ndcg@10", float("nan")), result.wall_seconds,
                     result.efficiency()["pairs_per_s"], elapsed / 60, eta / 60, budget_s / 60)

        total_seconds = time.perf_counter() - suite_started
        run_info = {
            "run_dir": str(run_dir),
            "model": self.reranker.describe(),
            "eval_config": self.config.__dict__,
            "max_queries_override": self.max_queries,
            "datasets": {d.name: {"fingerprint": d.meta.get("fingerprint"), "sha256": d.meta.get("sha256"),
                                  "source_info": d.meta.get("source_info"), "queries": d.num_queries,
                                  "pairs": d.num_pairs} for d in datasets},
            "timing": {"setup_s": setup_seconds, "warmup_s": warmup_seconds, "total_s": total_seconds,
                       "budget_s": budget_s, "within_budget": total_seconds <= budget_s},
            "environment": capture_environment(self.repo_root),
        }
        write_run_summary(run_dir, results, run_info)
        if total_seconds > budget_s:
            log.warning("suite took %.1f min, over the %.0f min budget — reduce max_queries in configs/datasets.yaml",
                        total_seconds / 60, budget_s / 60)
        return run_dir
