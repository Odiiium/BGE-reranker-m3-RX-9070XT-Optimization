from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path) -> Path:
    """Paths in configs are relative to the repository root."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path
