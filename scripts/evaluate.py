"""Evaluate any reranker variant: python scripts/evaluate.py --model-config configs/models/<variant>.yaml"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bgeopt.evaluation.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
