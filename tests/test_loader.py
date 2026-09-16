import numpy as np
import pytest
import yaml

from bgeopt.data import DatasetLoader, DatasetSpec, sample_query_ids
from bgeopt.data.sources import drop_identical_ids


def test_drop_identical_ids_removes_query_document_and_keeps_k():
    doc_ids = ["q1", "a", "b", "c"]
    indices = np.array([[0, 1, 2], [1, 2, 3]])
    scores = np.array([[9.0, 5.0, 4.0], [8.0, 7.0, 6.0]])
    kept_idx, kept_scores = drop_identical_ids(["q1", "q2"], doc_ids, indices, scores, k=2)
    assert [i.tolist() for i in kept_idx] == [[1, 2], [1, 2]]
    assert [s.tolist() for s in kept_scores] == [[5.0, 4.0], [8.0, 7.0]]


def test_sample_query_ids_is_deterministic_and_order_independent():
    ids = [f"q{i}" for i in range(50)]
    first = sample_query_ids(ids, 10, seed=1)
    assert first == sample_query_ids(list(reversed(ids)), 10, seed=1)
    assert len(first) == 10 and first == sorted(first)
    assert first != sample_query_ids(ids, 10, seed=2)
    assert sample_query_ids(ids, None, seed=1) == sorted(ids)
    assert sample_query_ids(ids, 500, seed=1) == sorted(ids)


def test_spec_merges_defaults_and_fingerprint_tracks_changes():
    defaults = {"top_k": 100, "seed": 42, "relevance_threshold": 1}
    spec = DatasetSpec.from_config("x", {"source": "beir", "hf_repo": "a/b", "max_queries": 5}, defaults)
    assert (spec.top_k, spec.max_queries, spec.params) == (100, 5, {"hf_repo": "a/b"})
    changed = DatasetSpec.from_config("x", {"source": "beir", "hf_repo": "a/b", "max_queries": 6}, defaults)
    assert spec.fingerprint() != changed.fingerprint()


def test_prepare_and_load_roundtrip_with_slicing(fake_datasets_config, tmp_path):
    loader = DatasetLoader(fake_datasets_config, tmp_path / "data")
    assert loader.prepare() == ["fake-all", "fake-small"]
    assert loader.prepare() == []  # up to date

    full = loader.load("fake-all")
    assert full.num_queries == 9  # q9 has no relevant judgments
    assert full.num_pairs == 9 * 4

    small = loader.load("fake-small")
    assert small.num_queries == 4 and all(len(q) == 3 for q in small.queries)
    assert set(small.qrels) == {q.query_id for q in small.queries}
    assert small.queries[0].doc_ids[:1] == [f"{small.queries[0].query_id}-neg"]

    assert loader.load("fake-all", max_queries=2).num_queries == 2
    summary = loader.summary()
    assert summary["pairs"].tolist() == [36, 12]


def test_changed_spec_marks_slice_outdated(fake_datasets_config, tmp_path):
    loader = DatasetLoader(fake_datasets_config, tmp_path / "data")
    loader.prepare(["fake-small"])
    config = yaml.safe_load(fake_datasets_config.read_text())
    config["datasets"]["fake-small"]["max_queries"] = 2
    fake_datasets_config.write_text(yaml.safe_dump(config))

    reloaded = DatasetLoader(fake_datasets_config, tmp_path / "data")
    assert not reloaded.is_prepared("fake-small")
    with pytest.raises(FileNotFoundError):
        reloaded.load("fake-small")
    assert reloaded.prepare(["fake-small"]) == ["fake-small"]
    assert reloaded.load("fake-small").num_queries == 2
