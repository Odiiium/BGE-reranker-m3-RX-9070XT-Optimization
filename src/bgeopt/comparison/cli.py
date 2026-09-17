"""Compare a candidate run with a baseline run of the same type (quality or latency).

    python scripts/compare.py results/quality/<baseline> results/quality/<candidate>
    python scripts/compare.py results/latency/<baseline> results/latency/<candidate>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from bgeopt.comparison.latency import LatencyRun, LatencyThresholds, compare_latency
from bgeopt.comparison.quality import QualityRun, QualityThresholds, compare_quality
from bgeopt.comparison.report import write_latency_comparison, write_quality_comparison
from bgeopt.utils.paths import resolve_path


def run_type(path: Path) -> str:
    if (path / "latency.csv").exists():
        return "latency"
    if (path / "metrics.csv").exists() and (path / "scores").is_dir():
        return "quality"
    raise ValueError(f"{path} is neither a quality run nor a latency benchmark run")


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--config", default="configs/compare.yaml")
    parser.add_argument("--out", help="output folder (default: results/compare/<baseline>__vs__<candidate>)")
    parser.add_argument("--allow-profile-mismatch", action="store_true",
                        help="compare latency runs of different benchmark profiles (not recommended)")
    args = parser.parse_args(argv)

    baseline, candidate = resolve_path(args.baseline), resolve_path(args.candidate)
    kind = run_type(baseline)
    if run_type(candidate) != kind:
        raise SystemExit("baseline and candidate must be runs of the same type")
    config = yaml.safe_load(resolve_path(args.config).read_text(encoding="utf-8"))
    out = resolve_path(args.out) if args.out else resolve_path("results/compare") / f"{baseline.name}__vs__{candidate.name}"

    if kind == "quality":
        thresholds = QualityThresholds(**config["quality"], **config["parity"])
        summary = write_quality_comparison(out, compare_quality(QualityRun.load(baseline), QualityRun.load(candidate),
                                                                thresholds))
    else:
        comparison = compare_latency(LatencyRun.load(baseline), LatencyRun.load(candidate),
                                     LatencyThresholds(**config["latency"]), args.allow_profile_mismatch)
        summary = write_latency_comparison(out, comparison)
    print(summary.read_text(encoding="utf-8"))
    print(f"results: {out}")
    return out
