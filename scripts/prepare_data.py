"""Download raw datasets and build evaluation slices (idempotent: up-to-date slices are skipped).

    python scripts/prepare_data.py                 # all slices from configs/datasets.yaml
    python scripts/prepare_data.py scifact --force # rebuild one slice
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bgeopt.data import DatasetLoader  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*", help="slice names (default: all)")
    parser.add_argument("--config", default=str(ROOT / "configs/datasets.yaml"))
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--force", action="store_true", help="rebuild even if up to date")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    loader = DatasetLoader(args.config, args.data_dir)
    loader.prepare(args.datasets or None, force=args.force)
    summary = loader.summary(args.datasets or None)
    print(summary.to_string(index=False))
    if "pairs" in summary:
        print(f"\ntotal: {int(summary['queries'].sum())} queries, {int(summary['pairs'].sum())} pairs")


if __name__ == "__main__":
    main()
