"""Latency benchmark of any variant (one process per session; repeat --model-config to interleave variants).

    python scripts/benchmark.py --model-config configs/models/v1_xxx.yaml --profile quick
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bgeopt.benchmark.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
