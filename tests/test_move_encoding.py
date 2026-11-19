import time

import chess
import numpy as np
import pytest

from chess_rl.encoding import NUM_MOVES, index_to_move, legal_mask, legal_move_indices, move_to_index
from tests.test_board_encoding import random_positions

# Standard move-generator test positions (chessprogramming.org "Perft Results").
PERFT_FENS = [
    chess.STARTING_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # Kiwipete
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
]


def assert_round_trips(board: chess.Board) -> int:
    indices = set()
    for move in board.legal_moves:
        idx = move_to_index(move, board.turn)
        assert 0 <= idx < NUM_MOVES
        assert index_to_move(idx, board) == move, (board.fen(), move.uci(), idx)
        indices.add(idx)
    assert len(indices) == board.legal_moves.count(), f"index collision in {board.fen()}"
    return len(indices)


def _perft_positions(board: chess.Board, depth: int):
    yield board
    if depth == 0:
        return
    for move in list(board.legal_moves):
        board.push(move)
        yield from (b.copy() for b in _perft_positions(board, depth - 1))
        board.pop()


@pytest.mark.parametrize("fen", PERFT_FENS)
def test_round_trip_every_move_in_perft_trees(fen):
    """Every legal move within 2 plies of each perft position (and both colours) round-trips uniquely."""
    count = 0
    for b in _perft_positions(chess.Board(fen), depth=2):
        assert_round_trips(b)
        count += 1
    assert count > 20


def test_round_trip_random_positions():
    # CI runs 3k positions; a one-off run over 50k positions (1.32M moves, seed 11) also passed.
    moves = sum(assert_round_trips(b) for b in random_positions(3_000, seed=7, max_plies=200))
    assert moves > 50_000


def test_known_indices_and_colour_symmetry():
    start = chess.Board()
    e2e4 = move_to_index(chess.Move.from_uci("e2e4"), chess.WHITE)
    assert e2e4 == chess.E2 * 73 + 1  # from e2, direction N (0), distance 2
    after = chess.Board()
    after.push_uci("e2e4")
    assert move_to_index(chess.Move.from_uci("e7e5"), chess.BLACK) == e2e4  # mirrored: the same index
    assert move_to_index(chess.Move.from_uci("g1f3"), chess.WHITE) == chess.G1 * 73 + 56 + 7  # knight (+2, -1)
    assert index_to_move(e2e4, start) == chess.Move.from_uci("e2e4")


def test_promotions():
    board = chess.Board("1n5k/P7/8/8/8/8/8/K7 w - - 0 1")
    promos = {m.uci(): move_to_index(m, board.turn) for m in board.legal_moves if m.promotion}
    assert set(promos) == {f"a7{t}{p}" for t in ("a8", "b8") for p in "qrbn"}
    assert promos["a7a8q"] == chess.A7 * 73 + 0  # queen promotion = plain N move
    assert promos["a7a8n"] == chess.A7 * 73 + 64 + 1 * 3 + 0  # push, knight
    assert promos["a7b8r"] == chess.A7 * 73 + 64 + 2 * 3 + 2  # capture right, rook
    black = chess.Board("k7/8/8/8/8/8/p7/1N5K b - - 0 1")
    assert index_to_move(move_to_index(chess.Move.from_uci("a2b1n"), chess.BLACK), black).uci() == "a2b1n"
    assert index_to_move(move_to_index(chess.Move.from_uci("a2a1q"), chess.BLACK), black).uci() == "a2a1q"


def test_castling_is_king_two_square_move():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert move_to_index(chess.Move.from_uci("e1g1"), chess.WHITE) == chess.E1 * 73 + 2 * 7 + 1  # E, distance 2
    assert move_to_index(chess.Move.from_uci("e1c1"), chess.WHITE) == chess.E1 * 73 + 6 * 7 + 1  # W, distance 2
    board.push_uci("e1g1")
    assert index_to_move(move_to_index(chess.Move.from_uci("e8c8"), chess.BLACK), board).uci() == "e8c8"


def test_legal_mask():
    board = chess.Board(PERFT_FENS[1])
    mask = legal_mask(board)
    assert mask.shape == (NUM_MOVES,) and mask.dtype == bool
    assert mask.sum() == board.legal_moves.count() == 48
    assert {index_to_move(int(i), board) for i in np.flatnonzero(mask)} == set(board.legal_moves)
    np.testing.assert_array_equal(np.sort(legal_move_indices(board)), np.flatnonzero(mask))


def test_invalid_indices_rejected():
    with pytest.raises(ValueError):
        index_to_move(NUM_MOVES, chess.Board())
    with pytest.raises(ValueError):
        index_to_move(chess.H8 * 73 + 0, chess.Board())  # north from h8: off the board
    with pytest.raises(ValueError):
        move_to_index(chess.Move.from_uci("a1c4"), chess.WHITE)  # not a queen or knight move


def test_mask_speed():
    boards = random_positions(2000, seed=8)
    t0 = time.perf_counter()
    for b in boards:
        legal_mask(b)
    per_board_us = (time.perf_counter() - t0) / len(boards) * 1e6
    assert per_board_us < 1000, f"{per_board_us:.0f} us/board"
