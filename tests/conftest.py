import os
import random

import numpy as np
import pytest

if not os.environ.get("CHESS_RL_TEST_GPU"):  # tests run on CPU unless GPU tests are requested explicitly
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


@pytest.fixture(autouse=True)
def _runs_file(tmp_path, monkeypatch):
    """Never append test runs to the tracked results/runs.jsonl."""
    path = tmp_path / "runs.jsonl"
    monkeypatch.setenv("CHESS_RL_RUNS_FILE", str(path))
    monkeypatch.setenv("CHESS_RL_ELO_CSV", str(tmp_path / "elo.csv"))
    monkeypatch.setenv("CHESS_RL_TACTICS_CSV", str(tmp_path / "tactics.csv"))
    return path


@pytest.fixture(autouse=True)
def _seed():
    random.seed(0)
    np.random.seed(0)
