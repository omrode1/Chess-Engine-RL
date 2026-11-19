import os
import random

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # callbacks plot during training; never open a window in tests
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # keep tests on CPU and deterministic


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


@pytest.fixture
def env():
    from baseline.chess_environment import ChessEnvWithOpeningBook

    e = ChessEnvWithOpeningBook(use_opening_book=False)
    e.reset()
    return e


def play_uci(env, moves):
    """Play UCI moves through env.step, returning (rewards, done, info)."""
    import chess

    rewards, done, info = [], False, {}
    for uci in moves:
        action = env.legal_moves_list.index(chess.Move.from_uci(uci))
        _, reward, done, info = env.step(action)
        rewards.append(reward)
    return rewards, done, info
