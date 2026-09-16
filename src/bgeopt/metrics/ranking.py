"""Ranking metrics with trec_eval / pytrec_eval semantics (the convention used by BEIR and MTEB).

- Ranking: score descending, ties broken by doc_id descending (trec_eval's rule).
- nDCG@k: linear gain = graded relevance (negative grades -> 0); IDCG uses all judged documents of the query,
  including relevant ones the first stage never retrieved.
- MRR@k: 1 / rank of the first document with relevance >= threshold within top-k, else 0.
- Recall@k: relevant documents in top-k / all relevant judged documents (relevance >= threshold).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from bgeopt.data.schema import Qrels, Run


def rank_documents(scores: Mapping[str, float]) -> list[str]:
    ranked = sorted(scores.items(), key=lambda item: item[0], reverse=True)  # doc_id desc (tie-break)
    ranked.sort(key=lambda item: item[1], reverse=True)  # stable: score desc
    return [doc_id for doc_id, _ in ranked]


class RankingMetric(ABC):
    name: str

    def __init__(self, k: int, relevance_threshold: int = 1):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.relevance_threshold = relevance_threshold

    @property
    def label(self) -> str:
        return f"{self.name}@{self.k}"

    @abstractmethod
    def __call__(self, ranked_doc_ids: Sequence[str], judgments: Mapping[str, int]) -> float: ...

    def _num_relevant(self, judgments: Mapping[str, int]) -> int:
        return sum(1 for rel in judgments.values() if rel >= self.relevance_threshold)


class NDCG(RankingMetric):
    name = "ndcg"

    def __call__(self, ranked_doc_ids: Sequence[str], judgments: Mapping[str, int]) -> float:
        dcg = sum(max(judgments.get(doc_id, 0), 0) / math.log2(rank + 2)
                  for rank, doc_id in enumerate(ranked_doc_ids[: self.k]))
        ideal = sorted((rel for rel in judgments.values() if rel > 0), reverse=True)[: self.k]
        idcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal))
        return dcg / idcg if idcg > 0 else 0.0


class MRR(RankingMetric):
    name = "mrr"

    def __call__(self, ranked_doc_ids: Sequence[str], judgments: Mapping[str, int]) -> float:
        for rank, doc_id in enumerate(ranked_doc_ids[: self.k], start=1):
            if judgments.get(doc_id, 0) >= self.relevance_threshold:
                return 1.0 / rank
        return 0.0


class Recall(RankingMetric):
    name = "recall"

    def __call__(self, ranked_doc_ids: Sequence[str], judgments: Mapping[str, int]) -> float:
        total = self._num_relevant(judgments)
        if total == 0:
            return 0.0
        found = sum(1 for doc_id in ranked_doc_ids[: self.k] if judgments.get(doc_id, 0) >= self.relevance_threshold)
        return found / total


METRICS: dict[str, type[RankingMetric]] = {cls.name: cls for cls in (NDCG, MRR, Recall)}


def build_metrics(names: Sequence[str], ks: Sequence[int], relevance_threshold: int = 1) -> list[RankingMetric]:
    unknown = set(names) - set(METRICS)
    if unknown:
        raise ValueError(f"unknown metrics {sorted(unknown)}, available: {sorted(METRICS)}")
    return [METRICS[name](k, relevance_threshold) for name in names for k in sorted(set(ks))]


@dataclass
class EvaluationResult:
    per_query: pd.DataFrame  # index: query_id, columns: metric labels
    mean: dict[str, float]


class RetrievalEvaluator:
    """Evaluates a run against qrels with several metrics at several cutoffs (one pass per metric label)."""

    def __init__(self, metrics: Sequence[str] = ("ndcg", "mrr", "recall"), ks: Sequence[int] = (1, 5, 10),
                 relevance_threshold: int = 1):
        self.metrics = build_metrics(metrics, ks, relevance_threshold)

    @property
    def labels(self) -> list[str]:
        return [m.label for m in self.metrics]

    def evaluate(self, run: Run, qrels: Qrels) -> EvaluationResult:
        """Averages over all queries in qrels; a query missing from the run scores 0 on every metric."""
        rows = {}
        for query_id in sorted(qrels):
            ranked = rank_documents(run.get(query_id, {}))
            rows[query_id] = {m.label: m(ranked, qrels[query_id]) for m in self.metrics}
        per_query = pd.DataFrame.from_dict(rows, orient="index", columns=self.labels)
        per_query.index.name = "query_id"
        mean = {label: float(per_query[label].mean()) if len(per_query) else 0.0 for label in self.labels}
        return EvaluationResult(per_query=per_query, mean=mean)
