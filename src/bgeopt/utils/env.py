"""Environment capture stored with every run, so results of different versions stay comparable."""

from __future__ import annotations

import importlib
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

ENV_PREFIXES = ("HSA_", "HIP_", "ROCM", "ROCBLAS_", "MIOPEN_", "PYTORCH_", "TORCH_", "TRITON_",
                "FLASH_ATTENTION_", "HF_", "TOKENIZERS_", "OMP_")


def ensure_rocm_wsl_env() -> None:
    """Runtime defaults that must be set before torch / tokenizers initialize."""
    if os.path.exists("/dev/dxg"):  # ROCm under WSL2 only sees the GPU through /dev/dxg with this flag
        os.environ.setdefault("HSA_ENABLE_DXG_DETECTION", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")  # 4x faster prepare(); the pipeline never forks


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_info(repo_root: Path) -> dict[str, Any]:
    status = _git(repo_root, "status", "--porcelain")
    return {
        "commit": _git(repo_root, "rev-parse", "HEAD"),
        "branch": _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "describe": _git(repo_root, "describe", "--tags", "--always", "--dirty"),
        "dirty": bool(status) if status is not None else None,
    }


def _version(module: str) -> str | None:
    try:
        return getattr(importlib.import_module(module), "__version__", "unknown")
    except ImportError:
        return None


def capture_environment(repo_root: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "git": git_info(repo_root),
        "platform": platform.platform(),
        "wsl": "microsoft" in platform.release().lower(),
        "python": platform.python_version(),
        "packages": {m: _version(m) for m in ("torch", "triton", "transformers", "numpy", "pandas", "bm25s")},
        "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(ENV_PREFIXES) and "TOKEN" not in k},
    }
    try:
        import torch

        info["torch_hip"] = torch.version.hip
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = {"name": props.name, "arch": getattr(props, "gcnArchName", None),
                           "memory_mb": props.total_memory // 2**20}
    except ImportError:
        pass
    return info
