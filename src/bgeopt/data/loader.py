"""DatasetLoader: builds evaluation slices from raw sources, caches them on disk and loads them back.

Layout:
    data/raw/...                         downloaded source files
    data/processed/<slice>/candidates.parquet   query_id, query, doc_ids[], doc_texts[], first_stage_scores[]
    data/processed/<slice>/qrels.parquet        query_id, doc_id, relevance
    data/processed/<slice>/meta.json            spec, fingerprint, stats, content hash
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from bgeopt.data.schema import DatasetSpec, QueryCandidates, RerankDataset
from bgeopt.data.sources import SOURCES, SourceData

log = logging.getLogger(__name__)

BUILDER_VERSION = 1  # bump when slice construction logic changes, forces rebuild


def sample_query_ids(eligible: list[str], max_queries: int | None, seed: int) -> list[str]:
    """Deterministic sample: independent of input order, stable for a given (set, max_queries, seed)."""
    ordered = sorted(eligible)
    if max_queries is None or max_queries >= len(ordered):
        return ordered
    return sorted(random.Random(seed).sample(ordered, max_queries))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DatasetLoader:
    def __init__(self, config_path: str | Path = "configs/datasets.yaml", data_dir: str | Path = "data"):
        self.config_path = Path(config_path)
        self.data_dir = Path(data_dir)
        raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        defaults = raw.get("defaults", {})
        self.specs: dict[str, DatasetSpec] = {
            name: DatasetSpec.from_config(name, cfg, defaults) for name, cfg in raw["datasets"].items()
        }

    @property
    def names(self) -> list[str]:
        return list(self.specs)

    def spec(self, name: str) -> DatasetSpec:
        if name not in self.specs:
            raise KeyError(f"unknown dataset slice '{name}', available: {', '.join(self.specs)}")
        return self.specs[name]

    def slice_dir(self, name: str) -> Path:
        return self.data_dir / "processed" / name

    def _fingerprint(self, spec: DatasetSpec) -> str:
        return f"v{BUILDER_VERSION}-{spec.fingerprint()}"

    def is_prepared(self, name: str) -> bool:
        meta_path = self.slice_dir(name) / "meta.json"
        if not meta_path.exists():
            return False
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta.get("fingerprint") == self._fingerprint(self.spec(name))

    # ------------------------------------------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------------------------------------------

    def prepare(self, names: list[str] | None = None, force: bool = False) -> list[str]:
        """Build missing or outdated slices. Returns the names that were (re)built."""
        built = []
        for name in names or self.names:
            if not force and self.is_prepared(name):
                log.info("%s: up to date", name)
                continue
            self._build(self.spec(name))
            built.append(name)
        return built

    def _build(self, spec: DatasetSpec) -> None:
        if spec.source not in SOURCES:
            raise ValueError(f"{spec.name}: unknown source '{spec.source}', available: {', '.join(SOURCES)}")
        started = time.perf_counter()
        log.info("%s: building slice (source=%s, max_queries=%s, top_k=%d)", spec.name, spec.source,
                 spec.max_queries, spec.top_k)

        data: SourceData = SOURCES[spec.source](
            spec, self.data_dir / "raw", lambda eligible: sample_query_ids(eligible, spec.max_queries, spec.seed)
        )
        if not data.queries:
            raise RuntimeError(f"{spec.name}: slice is empty")

        out = self.slice_dir(spec.name)
        out.mkdir(parents=True, exist_ok=True)
        candidates_path, qrels_path = out / "candidates.parquet", out / "qrels.parquet"
        pd.DataFrame({
            "query_id": [q.query_id for q in data.queries],
            "query": [q.query for q in data.queries],
            "doc_ids": [q.doc_ids for q in data.queries],
            "doc_texts": [q.doc_texts for q in data.queries],
            "first_stage_scores": [q.first_stage_scores for q in data.queries],
        }).to_parquet(candidates_path, index=False)
        pd.DataFrame(
            [(qid, did, rel) for qid, judged in data.qrels.items() for did, rel in judged.items()],
            columns=["query_id", "doc_id", "relevance"],
        ).to_parquet(qrels_path, index=False)

        num_pairs = sum(len(q) for q in data.queries)
        meta = {
            "name": spec.name,
            "fingerprint": self._fingerprint(spec),
            "spec": {**spec.__dict__},
            "source_info": data.info,
            "stats": {
                "num_queries": len(data.queries),
                "num_pairs": num_pairs,
                "mean_candidates_per_query": num_pairs / len(data.queries),
                "mean_query_chars": sum(len(q.query) for q in data.queries) / len(data.queries),
                "mean_doc_chars": sum(len(t) for q in data.queries for t in q.doc_texts) / num_pairs,
                "mean_judged_per_query": sum(len(j) for j in data.qrels.values()) / len(data.qrels),
            },
            "sha256": {"candidates": _sha256(candidates_path), "qrels": _sha256(qrels_path)},
            "build_seconds": round(time.perf_counter() - started, 1),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("%s: %d queries, %d pairs (%.0fs)", spec.name, len(data.queries), num_pairs, meta["build_seconds"])

    # ------------------------------------------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------------------------------------------

    def load(self, name: str, max_queries: int | None = None) -> RerankDataset:
        """Load a prepared slice. `max_queries` takes a further prefix (for smoke tests), not a new sample."""
        spec = self.spec(name)
        if not self.is_prepared(name):
            raise FileNotFoundError(f"slice '{name}' is missing or outdated, run prepare() / `make data` first")
        folder = self.slice_dir(name)
        meta: dict[str, Any] = json.loads((folder / "meta.json").read_text(encoding="utf-8"))

        frame = pd.read_parquet(folder / "candidates.parquet")
        if max_queries is not None:
            frame = frame.head(max_queries)
        queries = [
            QueryCandidates(
                query_id=row.query_id,
                query=row.query,
                doc_ids=list(row.doc_ids),
                doc_texts=list(row.doc_texts),
                first_stage_scores=[float(s) for s in row.first_stage_scores],
            )
            for row in frame.itertuples(index=False)
        ]
        selected = {q.query_id for q in queries}
        qrels: dict[str, dict[str, int]] = {}
        for qid, did, rel in pd.read_parquet(folder / "qrels.parquet").itertuples(index=False):
            if qid in selected:
                qrels.setdefault(qid, {})[did] = int(rel)
        return RerankDataset(spec=spec, queries=queries, qrels=qrels, meta=meta)

    def summary(self, names: list[str] | None = None) -> pd.DataFrame:
        rows = []
        for name in names or self.names:
            spec = self.spec(name)
            row: dict[str, Any] = {"dataset": name, "language": spec.language, "source": spec.source,
                                   "prepared": self.is_prepared(name)}
            if row["prepared"]:
                stats = json.loads((self.slice_dir(name) / "meta.json").read_text(encoding="utf-8"))["stats"]
                row.update(queries=stats["num_queries"], pairs=stats["num_pairs"],
                           query_chars=round(stats["mean_query_chars"]), doc_chars=round(stats["mean_doc_chars"]))
            rows.append(row)
        return pd.DataFrame(rows)
