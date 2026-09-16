import json

import pandas as pd
import pytest

from bgeopt.data import DatasetLoader
from bgeopt.evaluation import EvalConfig, RerankingEvaluationRunner
from bgeopt.models import build_reranker


def test_registry_builds_variant_from_config_and_rejects_unknown_fields():
    reranker = build_reranker({"name": "overlap", "version": "0.0.1", "implementation": "test-word-overlap"})
    assert reranker.name == "overlap"
    with pytest.raises(ValueError):
        build_reranker({"name": "x", "version": "0", "implementation": "test-word-overlap", "bogus": 1})
    with pytest.raises(KeyError):
        build_reranker({"name": "x", "version": "0", "implementation": "does-not-exist"})


def test_runner_end_to_end_with_swappable_reranker(fake_datasets_config, tmp_path):
    loader = DatasetLoader(fake_datasets_config, tmp_path / "data")
    loader.prepare()
    config = EvalConfig(datasets=["fake-all", "fake-small"], ks=[1, 5, 10], output_dir=str(tmp_path / "results"),
                        time_budget_minutes=1)
    reranker = build_reranker({"name": "overlap", "version": "0.0.1", "implementation": "test-word-overlap",
                               "warmup_iterations": 1})

    run_dir = RerankingEvaluationRunner(reranker, loader, config, repo_root=tmp_path).run()

    metrics = pd.read_csv(run_dir / "metrics.csv")
    assert set(metrics["metric"]) == {"ndcg", "mrr", "recall"} and set(metrics["k"]) == {1, 5, 10}
    reranked = metrics[(metrics.dataset == "fake-all") & (metrics.stage == "reranked")].set_index(["metric", "k"])
    first = metrics[(metrics.dataset == "fake-all") & (metrics.stage == "first_stage")].set_index(["metric", "k"])
    # the overlap scorer puts the grade-2 document first; the first stage had it at rank 3
    assert reranked.loc[("mrr", 1), "value"] == 1.0
    assert first.loc[("mrr", 1), "value"] == 0.0
    assert reranked.loc[("recall", 10), "value"] == first.loc[("recall", 10), "value"] == 1.0

    efficiency = pd.read_csv(run_dir / "efficiency.csv")
    assert efficiency["pairs"].tolist() == [36, 12]
    scores = pd.read_parquet(run_dir / "scores" / "fake-small.parquet")
    assert len(scores) == 12
    per_query = pd.read_parquet(run_dir / "per_query" / "fake-all.parquet")
    assert {"prepare_ms", "reranked/ndcg@10", "first_stage/ndcg@10"} <= set(per_query.columns)

    info = json.loads((run_dir / "run_info.json").read_text())
    assert info["timing"]["within_budget"] is True
    assert info["datasets"]["fake-small"]["pairs"] == 12
    assert "## Reranked" in (run_dir / "summary.md").read_text()
