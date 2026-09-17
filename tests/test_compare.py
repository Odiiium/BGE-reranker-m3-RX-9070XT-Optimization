import json

import numpy as np
import pytest

from bgeopt.benchmark.aggregate import write_benchmark_run
from bgeopt.benchmark.profile import BenchmarkProfile, ReplaySpec, SyntheticSpec
from bgeopt.benchmark.session import build_synthetic_input, run_session, select_replay_queries
from bgeopt.comparison import (
    LatencyRun,
    LatencyThresholds,
    QualityRun,
    QualityThresholds,
    compare_latency,
    compare_quality,
)
from bgeopt.comparison.report import write_latency_comparison, write_quality_comparison
from bgeopt.data import DatasetLoader
from bgeopt.evaluation import EvalConfig, RerankingEvaluationRunner
from bgeopt.models import build_reranker, load_model_config


def _quality_run(loader, tmp_path, implementation, tag, max_queries=None):
    config = EvalConfig(datasets=["fake-all", "fake-small"], output_dir=str(tmp_path / "results"))
    reranker = build_reranker({"name": "overlap", "version": "0", "implementation": implementation,
                               "warmup_iterations": 1})
    return RerankingEvaluationRunner(reranker, loader, config, repo_root=tmp_path, max_queries=max_queries,
                                     run_tag=tag).run()


@pytest.fixture
def loader(fake_datasets_config, tmp_path):
    loader = DatasetLoader(fake_datasets_config, tmp_path / "data")
    loader.prepare()
    return loader


def test_identical_runs_are_equivalent_with_perfect_parity(loader, tmp_path):
    a = _quality_run(loader, tmp_path, "test-word-overlap", "a")
    b = _quality_run(loader, tmp_path, "test-word-overlap", "b")
    comparison = compare_quality(QualityRun.load(a), QualityRun.load(b), QualityThresholds(bootstrap_samples=200,
                                                                                            permutations=200))
    assert comparison.verdict == {"quality": "PASS", "parity": "PASS"}
    assert (comparison.deltas["delta"] == 0).all() and (comparison.deltas["status"] == "equivalent").all()
    assert (comparison.parity["max_abs_dlogit"] == 0).all()
    assert (comparison.parity["kendall_tau_median"] == 1).all() and (comparison.parity["overlap@10"] == 1).all()
    summary = write_quality_comparison(tmp_path / "cmp", comparison)
    assert "Quality: PASS" in summary.read_text()


def test_broken_candidate_fails_quality_and_parity(loader, tmp_path):
    a = _quality_run(loader, tmp_path, "test-word-overlap", "a")
    b = _quality_run(loader, tmp_path, "test-reversed-overlap", "b")
    comparison = compare_quality(QualityRun.load(a), QualityRun.load(b), QualityThresholds(bootstrap_samples=200,
                                                                                            permutations=500))
    assert comparison.verdict == {"quality": "FAIL", "parity": "FAIL"}
    mrr1 = comparison.deltas.query("dataset == 'fake-all' and metric == 'mrr@1'").iloc[0]
    assert mrr1["delta"] == -1.0 and mrr1["status"] == "worse" and mrr1["p_value"] < 0.05
    assert comparison.parity["kendall_tau_median"].max() < 0


def test_subset_candidate_is_compared_on_common_queries(loader, tmp_path):
    a = _quality_run(loader, tmp_path, "test-word-overlap", "a")
    b = _quality_run(loader, tmp_path, "test-word-overlap", "b", max_queries=3)
    comparison = compare_quality(QualityRun.load(a), QualityRun.load(b), QualityThresholds(bootstrap_samples=100,
                                                                                            permutations=100))
    assert comparison.coverage["common_queries"].tolist() == [3, 3]
    assert any("common queries" in w for w in comparison.warnings)


def test_mismatched_slices_are_rejected(loader, tmp_path):
    a = _quality_run(loader, tmp_path, "test-word-overlap", "a")
    b = _quality_run(loader, tmp_path, "test-word-overlap", "b")
    info = json.loads((b / "run_info.json").read_text())
    info["datasets"]["fake-all"]["fingerprint"] = "other"
    (b / "run_info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="slices differ"):
        compare_quality(QualityRun.load(a), QualityRun.load(b), QualityThresholds())


def test_load_model_config_applies_typed_overrides(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text("name: x\nversion: '0'\nimplementation: test-word-overlap\nbatch_size: 2\n")
    config = load_model_config(path, ["batch_size=8", "name=y", "warmup_iterations=0"])
    assert config["batch_size"] == 8 and config["name"] == "y" and config["warmup_iterations"] == 0
    with pytest.raises(ValueError):
        load_model_config(path, ["batch_size"])


PROFILE = BenchmarkProfile(
    name="test", sessions=2,
    synthetic=SyntheticSpec(candidates=[1, 4], pair_tokens=[16, 64], query_tokens=4, warmup=2,
                            target_seconds=0.001, min_reps=3, max_reps=5),
    replay=ReplaySpec(queries_per_dataset=3, repeats=2, warmup_queries=1),
)


def _latency_run(loader, tmp_path, name):
    reranker = build_reranker({"name": "overlap", "version": "0", "implementation": "test-word-overlap"})
    replay = select_replay_queries(loader, ["fake-all", "fake-small"], PROFILE.replay.queries_per_dataset, seed=5)
    parts = [run_session(reranker, PROFILE, replay, session, seed=5) for session in range(PROFILE.sessions)]
    import pandas as pd

    samples = pd.concat([p[0] for p in parts], ignore_index=True)
    cells = pd.concat([p[1] for p in parts], ignore_index=True)
    sessions = [{"session": s, "load_s": 0.0, "first_call_ms": 1.0, "warmup_s": 0.0, "total_s": 1.0}
                for s in range(PROFILE.sessions)]
    run_info = {"model": reranker.describe(), "profile": {"name": "test"}, "timing": {"total_s": 1.0},
                "environment": {"git": {}, "packages": {"torch": None}}}
    write_benchmark_run(tmp_path / name, samples, cells, sessions, run_info, n_boot=100, seed=0)
    return tmp_path / name


def test_synthetic_inputs_have_exact_pair_length():
    reranker = build_reranker({"name": "overlap", "version": "0", "implementation": "test-word-overlap"})
    query, documents = build_synthetic_input(reranker, 7, 50, 4)
    prepared = reranker.prepare(query, documents)
    assert prepared.num_pairs == 7 and prepared.num_tokens == 7 * 50


def test_benchmark_session_aggregation_and_self_comparison(loader, tmp_path):
    run = _latency_run(loader, tmp_path, "bench")
    latency = LatencyRun.load(run).latency
    synthetic = latency[latency["workload"] == "synthetic"]
    assert len(synthetic) == 4 and set(latency.loc[latency["workload"] == "replay", "cell"]) == {"fake-all", "fake-small"}
    assert (synthetic["sessions"] == 2).all() and (synthetic["padding_ratio"] == 0).all()
    assert np.allclose(synthetic["pair_tokens"], [16, 64, 16, 64])
    replay = latency[latency["workload"] == "replay"].set_index("cell")
    assert replay.loc["fake-all", "samples"] == 2 * 2 * 3  # sessions x repeats x queries
    assert "Synthetic grid" in (run / "summary.md").read_text()

    comparison = compare_latency(LatencyRun.load(run), LatencyRun.load(run), LatencyThresholds(bootstrap_samples=200))
    assert np.allclose(comparison.speedups["speedup_p50"], 1.0)
    assert (comparison.speedups["change"] == "no change").all()
    assert comparison.summary["replay_geomean_speedup_p50"] == pytest.approx(1.0)
    assert "Latency comparison" in write_latency_comparison(tmp_path / "cmp", comparison).read_text()
    assert (comparison.speedups["ci_method"] == "sessions").all()


def test_session_ratio_ci_uses_sessions_as_units():
    from bgeopt.utils.stats import session_ratio_ci

    # bimodal per-session state (44 vs 54 ms) in both runs: no significant difference
    ratio, low, high = session_ratio_ci(np.array([43.1, 54.2]), np.array([44.7, 54.8]))
    assert low < 1.0 < high
    # stable heavy cell, real 30% slowdown: detected
    ratio, low, high = session_ratio_ci(np.array([420.9, 423.2, 424.0]), np.array([600.1, 598.7, 603.3]))
    assert high < 1.0 and ratio == pytest.approx(0.70, abs=0.01)
    assert session_ratio_ci(np.array([1.0]), np.array([1.0, 1.1])) is None


def test_latency_runs_of_different_profiles_are_rejected(loader, tmp_path):
    run = _latency_run(loader, tmp_path, "bench")
    other = LatencyRun.load(run)
    other.info = {**other.info, "profile": {"name": "quick"}}
    with pytest.raises(ValueError, match="profiles differ"):
        compare_latency(LatencyRun.load(run), other, LatencyThresholds(bootstrap_samples=50))
    comparison = compare_latency(LatencyRun.load(run), other, LatencyThresholds(bootstrap_samples=50),
                                 allow_profile_mismatch=True)
    assert any("profiles differ" in w for w in comparison.warnings)
