import chess
import numpy as np
import pytest

from baseline.chess_environment import ChessEnvWithOpeningBook
from tests.conftest import play_uci

FOOLS_MATE = ["f2f3", "e7e5", "g2g4", "d8h4"]  # Black mates
SCHOLARS_MATE = ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]  # White mates


def test_reset_returns_board_tensor(env):
    obs = env.reset()
    assert obs.shape == (12, 8, 8)
    assert obs.dtype == np.float32
    assert obs.sum() == 32  # 32 pieces on the starting board
    assert env.observation_space.contains(obs)


def test_step_applies_indexed_legal_move(env):
    move = env.legal_moves_list[3]
    obs, reward, done, info = env.step(3)
    assert env.board.move_stack[-1] == move
    assert isinstance(reward, float) and not done
    assert info["move_count"] == 1
    assert obs.shape == (12, 8, 8)


def test_out_of_range_action_is_rejected_without_moving(env):
    fen = env.board.fen()
    _, reward, done, info = env.step(len(env.legal_moves_list))
    assert reward == -0.5 and not done and info["invalid_move"]
    assert env.board.fen() == fen


@pytest.mark.parametrize("moves", [FOOLS_MATE, SCHOLARS_MATE], ids=["black-mates", "white-mates"])
def test_checkmate_ends_episode(env, moves):
    _, done, info = play_uci(env, moves)
    assert done and info["is_checkmate"]


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_curriculum_levels_start_from_valid_positions(level):
    for _ in range(10):
        e = ChessEnvWithOpeningBook(use_opening_book=False, curriculum_level=level)
        e.reset()
        assert e.board.is_valid()
        assert e.legal_moves_list == list(e.board.legal_moves)


def test_custom_start_position():
    fen = "8/8/5k2/8/8/3Q4/5K2/7r w - - 0 1"
    e = ChessEnvWithOpeningBook(use_opening_book=False, start_position=fen)
    e.reset()
    assert e.board.fen() == fen


def test_env_with_opening_book_constructs():
    e = ChessEnvWithOpeningBook(use_opening_book=True)
    e.reset()
    assert e.get_hint_from_opening_book() in e.board.legal_moves


def test_playing_the_book_move_is_flagged(monkeypatch):
    e = ChessEnvWithOpeningBook(use_opening_book=True)
    e.reset()
    e2e4 = chess.Move.from_uci("e2e4")
    monkeypatch.setattr(e.opening_book, "get_move", lambda board: e2e4)
    _, _, _, info = e.step(e.legal_moves_list.index(e2e4))
    assert info["followed_opening_book"]


# --- Known reward bugs in the v0 DQN design (documented in #33). The redesign
# drops shaped rewards, so these stay xfail until the v0 pipeline is removed (#27).


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="v0 bug: checkmate reward depends on colour (+0.5 White vs ~+3 Black)"
)
def test_mating_move_reward_is_colour_symmetric():
    white, _, _ = play_uci(_fresh(), SCHOLARS_MATE)
    black, _, _ = play_uci(_fresh(), FOOLS_MATE)
    assert white[-1] == pytest.approx(black[-1], abs=1.0)
    assert white[-1] > 5


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="v0 bug: is_capture is evaluated after the move is pushed"
)
def test_quiet_move_is_not_rewarded_as_capture():
    e = _fresh()
    e.board.push(chess.Move.from_uci("e2e4"))
    assert not e.board.is_capture(chess.Move.from_uci("e2e4"))


def _fresh():
    e = ChessEnvWithOpeningBook(use_opening_book=False)
    e.reset()
    return e
