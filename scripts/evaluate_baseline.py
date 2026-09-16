"""Evaluate the V0 baseline (transformers fp16 + SDPA) on the full dataset suite.

    python scripts/evaluate_baseline.py                       # full suite (<= 45 min)
    python scripts/evaluate_baseline.py --max-queries 5       # smoke test
    python scripts/evaluate_baseline.py --datasets scifact    # single slice
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bgeopt.evaluation.cli import main  # noqa: E402

if __name__ == "__main__":
    main(default_model_config="configs/models/v0_baseline.yaml")
