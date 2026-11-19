"""Seeding, device selection, and environment facts (hardware, versions, git) for run records."""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def seed_everything(seed: int | None) -> None:
    """Seed Python, NumPy, and torch (CPU and CUDA). None leaves everything unseeded."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(requested: str = "auto") -> str:
    """'auto' -> 'cuda' when available, else 'cpu'. Explicit values are validated."""
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device {requested!r} requested but CUDA is not available")
    return requested


def _ram_gb() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except (ValueError, OSError, AttributeError):
        return None


def hardware_info() -> dict[str, Any]:
    """Machine facts recorded with every run. Deliberately excludes hostname and user names."""
    import torch

    info: dict[str, Any] = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "gpu": None,
        "gpu_mem_gb": None,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["gpu_mem_gb"] = round(props.total_memory / 1024**3, 1)
    return info


def versions() -> dict[str, Any]:
    import torch

    from chess_rl import __version__

    return {
        "chess_rl": __version__,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "numpy": np.__version__,
    }


def git_info(repo: Path = REPO_ROOT) -> dict[str, Any]:
    """Commit SHA and whether tracked code had uncommitted changes.

    results/ is excluded: run records and Elo tables are written there by the runs themselves.
    """

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    sha = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)results")
    return {"sha": sha, "dirty": bool(status) if status is not None else None}
