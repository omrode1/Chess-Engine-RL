"""Compute accounting: one JSON line per run in results/runs.jsonl.

The M8 headline experiment compares how much compute different engines need, so every training and
evaluation run records what it cost and exactly what produced it:

    with track_run("train-sl", config, seed=config.seed) as run:
        model = train(...)
        run.outputs["final_model"] = path

The record is written even when the run fails or is interrupted (status "failed" / "interrupted").
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chess_rl.system import REPO_ROOT, git_info, hardware_info, versions

DEFAULT_RUNS_FILE = REPO_ROOT / "results" / "runs.jsonl"


def runs_file_path() -> Path:
    """results/runs.jsonl, or $CHESS_RL_RUNS_FILE (used by tests and scratch experiments)."""
    return Path(os.environ.get("CHESS_RL_RUNS_FILE", DEFAULT_RUNS_FILE))


@dataclasses.dataclass
class Run:
    kind: str
    config: dict[str, Any]
    seed: int | None
    device: str
    id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex[:12])
    outputs: dict[str, Any] = dataclasses.field(default_factory=dict)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@contextmanager
def track_run(
    kind: str,
    config: Any,
    *,
    seed: int | None = None,
    device: str = "cpu",
    runs_file: str | Path | None = None,
) -> Iterator[Run]:
    cfg = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else dict(config)
    run = Run(kind=kind, config=cfg, seed=seed, device=device)
    started_at, t0 = _now(), time.perf_counter()
    git = git_info()  # the code that runs, captured before the run writes anything
    status = "ok"
    try:
        yield run
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except BaseException:
        status = "failed"
        raise
    finally:
        wall = time.perf_counter() - t0
        record = {
            "id": run.id,
            "kind": kind,
            "status": status,
            "started_at": started_at,
            "finished_at": _now(),
            "wall_seconds": round(wall, 1),
            "gpu_hours": round(wall / 3600, 4) if device.startswith("cuda") else 0.0,
            "device": device,
            "seed": seed,
            "git": git,
            "hardware": hardware_info(),
            "versions": versions(),
            "config": cfg,
            "outputs": run.outputs,
        }
        path = Path(runs_file) if runs_file is not None else runs_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")


def read_runs(runs_file: str | Path | None = None) -> list[dict[str, Any]]:
    path = Path(runs_file) if runs_file is not None else runs_file_path()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
