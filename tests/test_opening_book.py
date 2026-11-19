import random

import chess

from chess_rl.openings import OpeningBook


def test_book_has_expected_positions():
    assert len(OpeningBook().book) == 27


def test_every_book_line_is_legal():
    book = OpeningBook()
    rng = random.Random(0)
    for _ in range(300):
        random.seed(rng.random())
        board = chess.Board()
        while (move := book.get_move(board)) is not None:
            assert move in board.legal_moves, board.fen()
            board.push(move)


def test_unknown_position_returns_none():
    board = chess.Board("8/8/5k2/8/8/3Q4/5K2/7r w - - 0 1")
    assert OpeningBook().get_move(board) is None


def test_unweighted_selection_returns_book_move():
    book = OpeningBook(use_weighted_selection=False)
    first_moves = {m for m, _ in book.book[chess.Board().board_fen()]}
    assert book.get_move(chess.Board()) in first_moves
