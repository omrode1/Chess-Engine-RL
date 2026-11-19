import random

import chess
import pytest

from baseline.chess_environment import ChessEnvWithOpeningBook
from chess_rl.positions import CURRICULUM, LEVEL_NAMES, fens


def test_levels_and_counts():
    assert set(CURRICULUM) == set(LEVEL_NAMES) == {0, 1, 2, 3}
    assert [len(CURRICULUM[lvl]) for lvl in (0, 1, 2, 3)] == [1, 4, 4, 4]
    assert fens(0) == [chess.STARTING_FEN]


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_all_positions_valid(level):
    for pos in CURRICULUM[level]:
        board = pos.board()
        assert board.is_valid(), (pos.label, board.status())
        assert not board.is_game_over()


def test_mate_labels_are_true():
    """Any position whose label claims a mate in 1 must have a mating move."""
    for positions in CURRICULUM.values():
        for pos in positions:
            label = pos.label.lower()
            if "mate in 1" in label and "no mate in 1" not in label:
                board = pos.board()
                assert any(board.is_checkmate() for _ in _after_each_move(board)), pos.label


def _after_each_move(board):
    for move in list(board.legal_moves):
        board.push(move)
        yield board
        board.pop()


def test_relabelled_position_really_has_no_mate_in_one():
    board = CURRICULUM[3][3].board()
    assert not any(board.is_checkmate() for _ in _after_each_move(board))


@pytest.mark.parametrize("level", [1, 2, 3])
def test_baseline_env_draws_the_same_positions_as_before(level):
    """Same seed -> same start positions as the 2025 inline lists (same order, same RNG use)."""
    random.seed(123)
    expected = [random.choice(fens(level)) for _ in range(20)]
    random.seed(123)
    env = ChessEnvWithOpeningBook(use_opening_book=False, curriculum_level=level)  # __init__ draws once
    got = [env.board.fen()]
    for _ in range(19):
        env.reset()
        got.append(env.board.fen())
    assert got == expected
