import os
import random

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # callbacks plot during training; never open a window in tests
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # keep tests on CPU and deterministic


@pytest.fixture(autouse=True)
def _seed():
    random.seed(0)
    np.random.seed(0)


@pytest.fixture
def env():
    from src.chess_environment import ChessEnvWithOpeningBook

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
