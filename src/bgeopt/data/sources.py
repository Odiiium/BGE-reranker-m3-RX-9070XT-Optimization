"""Dataset sources: download raw data and build first-stage candidate lists.

Every source returns `SourceData` for the *sampled* queries only, so expensive first-stage retrieval
(BM25) runs just for the queries that end up in the slice.
"""

from __future__ import annotations

import gzip
import json
import logging
import shutil
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import bm25s
import numpy as np
import pandas as pd
import Stemmer
from huggingface_hub import hf_hub_download

from bgeopt.data.schema import DatasetSpec, Qrels, QueryCandidates

log = logging.getLogger(__name__)

# Selects which of the eligible query ids go into the slice (sampling lives in the loader).
QuerySelector = Callable[[list[str]], list[str]]


@dataclass
class SourceData:
    queries: list[QueryCandidates]
    qrels: Qrels
    info: dict[str, Any] = field(default_factory=dict)


def eligible_query_ids(qrels: Qrels, relevance_threshold: int) -> list[str]:
    """Queries with at least one judgment >= threshold; others make Recall / MRR / nDCG undefined."""
    return sorted(qid for qid, judged in qrels.items() if any(rel >= relevance_threshold for rel in judged.values()))


# ----------------------------------------------------------------------------------------------------------
# BM25 (bm25s, Lucene variant k1=1.5 b=0.75, English stemming + stopwords)
# ----------------------------------------------------------------------------------------------------------


class BM25Index:
    def __init__(self, texts: list[str], language: str = "english"):
        self._stemmer = Stemmer.Stemmer(language)
        self._stopwords = "en" if language == "english" else None
        tokens = bm25s.tokenize(texts, stopwords=self._stopwords, stemmer=self._stemmer, show_progress=False)
        self._retriever = bm25s.BM25()
        self._retriever.index(tokens, show_progress=False)

    def _tokenize(self, queries: list[str]) -> Any:
        return bm25s.tokenize(queries, stopwords=self._stopwords, stemmer=self._stemmer, show_progress=False)

    def search(self, queries: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-k over the whole corpus: (doc_indices, scores), both [n_queries, k]."""
        return self._retriever.retrieve(self._tokenize(queries), k=k, show_progress=False, n_threads=0)

    def scores(self, query: str) -> np.ndarray:
        """Scores for every indexed document, [n_docs]."""
        tokenized = bm25s.tokenize([query], stopwords=self._stopwords, stemmer=self._stemmer,
                                   return_ids=False, show_progress=False)
        return self._retriever.get_scores(tokenized[0])


def _top_k_from_pool(scores: np.ndarray, pool: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    pool_scores = scores[pool]
    order = np.lexsort((pool, -pool_scores))[:k]  # score desc, index asc for ties
    return pool[order], pool_scores[order]


# ----------------------------------------------------------------------------------------------------------
# Download helpers
# ----------------------------------------------------------------------------------------------------------


def _download_url(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("downloading %s", url)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (bgeopt dataset downloader)"})  # NIST 403s urllib's default UA
    with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as out:
        shutil.copyfileobj(response, out)
    tmp.rename(dest)
    return dest


def _hf_file(spec: DatasetSpec, filename: str, raw_dir: Path) -> Path:
    repo = spec.params["hf_repo"]
    return Path(hf_hub_download(
        repo_id=repo, filename=filename, repo_type="dataset", revision=spec.params.get("revision"),
        local_dir=raw_dir / repo.replace("/", "__"),
    ))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _doc_text(title: str | None, text: str | None) -> str:
    return f"{title or ''} {text or ''}".strip()


# ----------------------------------------------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------------------------------------------


def drop_identical_ids(query_ids: list[str], doc_ids: list[str], indices: np.ndarray, scores: np.ndarray,
                       k: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Remove the query's own document from its hits (BEIR/MTEB `ignore_identical_ids`, e.g. ArguAna, Quora)."""
    kept_indices, kept_scores = [], []
    for row, qid in enumerate(query_ids):
        keep = np.array([doc_ids[i] != qid for i in indices[row]], dtype=bool)
        kept_indices.append(indices[row][keep][:k])
        kept_scores.append(scores[row][keep][:k])
    return kept_indices, kept_scores


def load_beir(spec: DatasetSpec, raw_dir: Path, select: QuerySelector) -> SourceData:
    """BEIR dataset in MTEB layout (corpus.jsonl, queries.jsonl, qrels/<split>.jsonl) + BM25 top-k."""
    split = spec.params.get("split", "test")
    exclude_identical = bool(spec.params.get("exclude_identical_ids", False))
    corpus_rows = _read_jsonl(_hf_file(spec, "corpus.jsonl", raw_dir))
    queries = {row["_id"]: row["text"] for row in _read_jsonl(_hf_file(spec, "queries.jsonl", raw_dir))}

    all_qrels: Qrels = defaultdict(dict)
    for row in _read_jsonl(_hf_file(spec, f"qrels/{split}.jsonl", raw_dir)):
        all_qrels[str(row["query-id"])][str(row["corpus-id"])] = int(float(row["score"]))

    selected = select([qid for qid in eligible_query_ids(all_qrels, spec.relevance_threshold) if qid in queries])

    doc_ids = [row["_id"] for row in corpus_rows]
    doc_texts = [_doc_text(row.get("title"), row.get("text")) for row in corpus_rows]
    log.info("%s: BM25 index over %d documents", spec.name, len(doc_ids))
    index = BM25Index(doc_texts)
    k = min(spec.top_k, len(doc_ids))
    indices, scores = index.search([queries[qid] for qid in selected], k=min(k + int(exclude_identical), len(doc_ids)))
    if exclude_identical:
        indices, scores = drop_identical_ids(selected, doc_ids, indices, scores, k)

    items = [
        QueryCandidates(
            query_id=qid,
            query=queries[qid],
            doc_ids=[doc_ids[i] for i in indices[row]],
            doc_texts=[doc_texts[i] for i in indices[row]],
            first_stage_scores=[float(s) for s in scores[row]],
        )
        for row, qid in enumerate(selected)
    ]
    return SourceData(
        queries=items,
        qrels={qid: dict(all_qrels[qid]) for qid in selected},
        info={"first_stage": "bm25s lucene k1=1.5 b=0.75, english stemmer, full corpus"
                             + (", query's own document excluded" if exclude_identical else ""),
              "corpus_size": len(doc_ids), "eligible_queries": len(eligible_query_ids(all_qrels, spec.relevance_threshold))},
    )


TREC_DL_URLS = {
    "top1000": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-passagetest{year}-top1000.tsv.gz",
    "qrels": "https://trec.nist.gov/data/deep/{year}qrels-pass.txt",
}


def load_trec_dl(spec: DatasetSpec, raw_dir: Path, select: QuerySelector) -> SourceData:
    """TREC DL passage track. The official top-1000 file has texts but no scores/order, so BM25 is computed
    over all passages of that file (IDF from the pooled passages) and each query's own pool is re-ranked."""
    year = int(spec.params["year"])
    base = raw_dir / "trec-dl"
    top1000 = _download_url(TREC_DL_URLS["top1000"].format(year=year), base / f"passagetest{year}-top1000.tsv.gz")
    qrels_path = _download_url(TREC_DL_URLS["qrels"].format(year=year), base / f"{year}qrels-pass.txt")

    all_qrels: Qrels = defaultdict(dict)
    with open(qrels_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                qid, _, pid, rel = line.split()
                all_qrels[qid][pid] = int(rel)

    queries: dict[str, str] = {}
    passage_index: dict[str, int] = {}
    passages: list[str] = []
    pools: dict[str, list[int]] = defaultdict(list)
    with gzip.open(top1000, "rt", encoding="utf-8") as f:
        for line in f:
            qid, pid, query, passage = line.rstrip("\n").split("\t", 3)
            queries[qid] = query
            if pid not in passage_index:
                passage_index[pid] = len(passages)
                passages.append(passage)
            pools[qid].append(passage_index[pid])

    pids = [None] * len(passage_index)
    for pid, i in passage_index.items():
        pids[i] = pid

    selected = select([qid for qid in eligible_query_ids(all_qrels, spec.relevance_threshold) if qid in pools])
    log.info("%s: BM25 index over %d pooled passages", spec.name, len(passages))
    index = BM25Index(passages)

    items = []
    for qid in selected:
        top, scores = _top_k_from_pool(index.scores(queries[qid]), np.asarray(pools[qid]), spec.top_k)
        items.append(QueryCandidates(
            query_id=qid,
            query=queries[qid],
            doc_ids=[pids[i] for i in top],
            doc_texts=[passages[i] for i in top],
            first_stage_scores=[float(s) for s in scores],
        ))
    return SourceData(
        queries=items,
        qrels={qid: dict(all_qrels[qid]) for qid in selected},
        info={"first_stage": "bm25s lucene over official top-1000 pool per query (IDF from pooled passages)",
              "pool_passages": len(passages), "eligible_queries": len(eligible_query_ids(all_qrels, spec.relevance_threshold))},
    )


def load_miracl_reranking(spec: DatasetSpec, raw_dir: Path, select: QuerySelector) -> SourceData:
    """MMTEB MIRACLReranking: ready-made candidate lists (first-stage order preserved, score = -rank)."""
    lang, split = spec.params["lang"], spec.params.get("split", "dev")

    def table(kind: str) -> pd.DataFrame:
        return pd.read_parquet(_hf_file(spec, f"{lang}-{kind}/{split}-00000-of-00001.parquet", raw_dir))

    corpus = table("corpus")
    texts = dict(zip(corpus["_id"], (_doc_text(t, x) for t, x in zip(corpus["title"], corpus["text"]))))
    queries_df = table("queries")
    queries = dict(zip(queries_df["_id"], queries_df["text"]))
    qrels_df = table("qrels")
    all_qrels: Qrels = defaultdict(dict)
    for qid, did, score in zip(qrels_df["query-id"], qrels_df["corpus-id"], qrels_df["score"]):
        all_qrels[qid][did] = int(score)
    ranked = dict(zip(table("top_ranked")["query-id"], table("top_ranked")["corpus-ids"]))

    selected = select([qid for qid in eligible_query_ids(all_qrels, spec.relevance_threshold) if qid in ranked])
    items = []
    for qid in selected:
        doc_ids = [str(d) for d in list(ranked[qid])[: spec.top_k]]
        items.append(QueryCandidates(
            query_id=qid,
            query=queries[qid],
            doc_ids=doc_ids,
            doc_texts=[texts[d] for d in doc_ids],
            first_stage_scores=[-float(rank) for rank in range(1, len(doc_ids) + 1)],
        ))
    return SourceData(
        queries=items,
        qrels={qid: dict(all_qrels[qid]) for qid in selected},
        info={"first_stage": "MMTEB MIRACLReranking top_ranked order (score = -rank)",
              "eligible_queries": len(eligible_query_ids(all_qrels, spec.relevance_threshold))},
    )


SOURCES: dict[str, Callable[[DatasetSpec, Path, QuerySelector], SourceData]] = {
    "beir": load_beir,
    "trec_dl": load_trec_dl,
    "miracl_reranking": load_miracl_reranking,
}
