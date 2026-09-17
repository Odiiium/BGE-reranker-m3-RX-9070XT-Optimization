"""Latency benchmark of the V0 baseline.

    python scripts/benchmark_baseline.py                   # default profile: 3 sessions (~12 min)
    python scripts/benchmark_baseline.py --profile quick   # 1 session (~2 min)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bgeopt.benchmark.cli import main  # noqa: E402

if __name__ == "__main__":
    main(default_model_config="configs/models/v0_baseline.yaml")
