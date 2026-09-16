"""Command-line entry point shared by scripts/evaluate.py and scripts/evaluate_baseline.py."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from bgeopt.utils import ensure_rocm_wsl_env

REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: list[str] | None = None, default_model_config: str | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Evaluate a reranker variant on the dataset suite.")
    parser.add_argument("--model-config", default=default_model_config, required=default_model_config is None,
                        help="YAML in configs/models/")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--datasets", nargs="+", help="override the dataset list of the eval config")
    parser.add_argument("--max-queries", type=int, help="take only the first N queries per slice (smoke test)")
    parser.add_argument("--tag", help="suffix for the results folder name")
    args = parser.parse_args(argv)

    ensure_rocm_wsl_env()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    log = logging.getLogger("bgeopt")

    # heavy imports after the ROCm env is fixed
    from bgeopt.data import DatasetLoader
    from bgeopt.evaluation.runner import EvalConfig, RerankingEvaluationRunner
    from bgeopt.models import build_reranker

    config = EvalConfig.from_yaml(_resolve(args.eval_config))
    if args.datasets:
        config.datasets = args.datasets
    config.data_dir = str(_resolve(config.data_dir))
    config.output_dir = str(_resolve(config.output_dir))

    loader = DatasetLoader(_resolve(config.datasets_config), config.data_dir)
    built = loader.prepare(config.datasets)
    if built:
        log.info("prepared slices: %s", ", ".join(built))

    started = time.perf_counter()
    reranker = build_reranker(_resolve(args.model_config))
    setup_seconds = time.perf_counter() - started
    log.info("loaded %s in %.1fs", reranker.name, setup_seconds)

    runner = RerankingEvaluationRunner(reranker, loader, config, REPO_ROOT, max_queries=args.max_queries,
                                       run_tag=args.tag)
    run_dir = runner.run(setup_seconds=setup_seconds)
    print("\n" + (run_dir / "summary.md").read_text(encoding="utf-8"))
    print(f"results: {run_dir}")
    return run_dir
