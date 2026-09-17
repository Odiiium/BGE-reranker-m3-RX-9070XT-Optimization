"""A tiny fake dataset source and a CPU-only reranker, so loader and runner are testable without GPU or network."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pytest
import yaml

from bgeopt.data import QueryCandidates, sources
from bgeopt.data.sources import SourceData, eligible_query_ids
from bgeopt.models import BaseReranker, PreparedInput, RerankerConfig, register_reranker

FAKE_QUERIES = {f"q{i}": f"topic {i} question" for i in range(10)}
FAKE_QRELS = {f"q{i}": {f"q{i}-rel": 2, f"q{i}-weak": 1, f"q{i}-neg": 0} for i in range(9)}  # q9 has no judgments


def fake_source(spec, raw_dir, select):
    selected = select(eligible_query_ids(FAKE_QRELS, spec.relevance_threshold))
    items = []
    for qid in selected:
        doc_ids = [f"{qid}-neg", f"{qid}-weak", f"{qid}-rel", f"{qid}-noise"][: spec.top_k]
        texts = ["unrelated text", f"topic {qid[1:]} maybe", f"topic {qid[1:]} question answer", "noise"][: spec.top_k]
        items.append(QueryCandidates(qid, FAKE_QUERIES[qid], doc_ids, texts,
                                     [float(-r) for r in range(len(doc_ids))]))
    return SourceData(queries=items, qrels={qid: FAKE_QRELS[qid] for qid in selected}, info={"first_stage": "fake"})


@pytest.fixture
def fake_datasets_config(tmp_path, monkeypatch):
    monkeypatch.setitem(sources.SOURCES, "fake", fake_source)
    config = {
        "defaults": {"top_k": 100, "seed": 7, "relevance_threshold": 1, "max_queries": None},
        "datasets": {
            "fake-all": {"source": "fake", "language": "en"},
            "fake-small": {"source": "fake", "language": "en", "max_queries": 4, "top_k": 3},
        },
    }
    path = tmp_path / "datasets.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@dataclass
class OverlapConfig(RerankerConfig):
    batch_size: int = 2


@register_reranker("test-word-overlap")
class WordOverlapReranker(BaseReranker):
    """Scores by number of query words in the document."""

    config_class = OverlapConfig

    def prepare(self, query: str, documents: Sequence[str]) -> PreparedInput:
        words = set(query.split())
        counts = [len(words & set(doc.split())) for doc in documents]
        tokens = sum(len(doc.split()) for doc in documents)
        return PreparedInput(payload=counts, num_pairs=len(documents), num_tokens=tokens, num_padded_tokens=tokens)

    def predict(self, prepared: PreparedInput) -> np.ndarray:
        return np.asarray(prepared.payload, dtype=np.float32)


@register_reranker("test-reversed-overlap")
class ReversedOverlapReranker(WordOverlapReranker):
    """Deliberately broken variant: inverts the ranking."""

    def predict(self, prepared: PreparedInput) -> np.ndarray:
        return -super().predict(prepared)
