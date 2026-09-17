"""Compare a candidate run with a baseline run (quality runs or latency benchmark runs).

    python scripts/compare.py results/quality/<baseline> results/quality/<candidate>
    python scripts/compare.py results/latency/<baseline> results/latency/<candidate>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bgeopt.comparison.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
