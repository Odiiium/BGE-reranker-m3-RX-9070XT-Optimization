"""Data structures shared by dataset sources, the loader and the evaluator."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

# qrels[query_id][doc_id] = graded relevance (<= 0 means not relevant)
Qrels = dict[str, dict[str, int]]
# run[query_id][doc_id] = score (higher = more relevant)
Run = dict[str, dict[str, float]]


@dataclass(frozen=True)
class DatasetSpec:
    """Slice definition from configs/datasets.yaml (defaults already merged)."""

    name: str
    source: str
    language: str
    top_k: int
    seed: int
    relevance_threshold: int
    max_queries: int | None = None
    params: dict[str, Any] = field(default_factory=dict)  # source-specific: hf_repo, revision, split, year, lang

    @classmethod
    def from_config(cls, name: str, raw: dict[str, Any], defaults: dict[str, Any]) -> DatasetSpec:
        merged = {**defaults, **raw}
        known = {"source", "language", "top_k", "seed", "relevance_threshold", "max_queries"}
        return cls(
            name=name,
            source=merged["source"],
            language=merged.get("language", "en"),
            top_k=int(merged["top_k"]),
            seed=int(merged["seed"]),
            relevance_threshold=int(merged["relevance_threshold"]),
            max_queries=merged.get("max_queries"),
            params={k: v for k, v in merged.items() if k not in known},
        )

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class QueryCandidates:
    """One query with its first-stage candidates in first-stage rank order."""

    query_id: str
    query: str
    doc_ids: list[str]
    doc_texts: list[str]
    first_stage_scores: list[float]

    def __len__(self) -> int:
        return len(self.doc_ids)


@dataclass
class RerankDataset:
    spec: DatasetSpec
    queries: list[QueryCandidates]
    qrels: Qrels
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def num_queries(self) -> int:
        return len(self.queries)

    @property
    def num_pairs(self) -> int:
        return sum(len(q) for q in self.queries)

    @property
    def num_chars(self) -> int:
        """Rough size proxy (query repeated per pair + documents), used for ETA before tokenization."""
        return sum(len(q.query) * len(q) + sum(len(t) for t in q.doc_texts) for q in self.queries)

    def first_stage_run(self) -> Run:
        return {q.query_id: dict(zip(q.doc_ids, q.first_stage_scores)) for q in self.queries}
