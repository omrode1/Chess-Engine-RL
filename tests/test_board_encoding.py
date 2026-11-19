import random
import time

import chess
import numpy as np
import pytest

from chess_rl.encoding import NUM_PLANES, PLANE_NAMES, encode_board, encode_boards

P = {name: i for i, name in enumerate(PLANE_NAMES)}


def push(board: chess.Board, *ucis: str) -> chess.Board:
    for u in ucis:
        board.push_uci(u)
    return board


def random_positions(n: int, seed: int = 0, max_plies: int = 80) -> list[chess.Board]:
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b = chess.Board()
        for _ in range(rng.randrange(max_plies)):
            if b.is_game_over():
                break
            b.push(rng.choice(list(b.legal_moves)))
        out.append(b)
    return out


def test_start_position():
    x = encode_board(chess.Board())
    assert x.shape == (NUM_PLANES, 8, 8) and x.dtype == np.float32
    assert x[P["our_pawn"], 1].sum() == 8 and x[P["their_pawn"], 6].sum() == 8
    assert x[P["our_king"], 0, 4] == 1 and x[P["their_king"], 7, 4] == 1
    assert x[P["colour_white"]].all()
    for name in ("our_castle_kingside", "our_castle_queenside", "their_castle_kingside", "their_castle_queenside"):
        assert x[P[name]].all()
    assert not x[P["repetition_1"]].any() and not x[P["en_passant"]].any()
    assert x[:12].sum() == 32


def test_black_to_move_is_seen_from_blacks_side():
    x = encode_board(push(chess.Board(), "e2e4"))
    assert not x[P["colour_white"]].any()
    assert x[P["our_pawn"], 1].sum() == 8  # Black's pawns, oriented to rank 2
    assert x[P["their_pawn"], 6].sum() == 7 and x[P["their_pawn"], 4, 4] == 1  # e4 seen as e5
    assert x[P["our_king"], 0, 4] == 1  # Black's king on e8 appears on e1


def test_mirror_symmetry():
    """A position and its colour-flipped twin encode identically, apart from the colour plane."""
    for b in random_positions(300):
        b = chess.Board(b.fen())  # drop history: mirror() does not keep the move stack
        x, y = encode_board(b), encode_board(b.mirror())
        same = [i for i in range(NUM_PLANES) if i != P["colour_white"]]
        np.testing.assert_array_equal(x[same], y[same])
        assert x[P["colour_white"]].all() != y[P["colour_white"]].all()


def test_en_passant_only_when_capture_is_legal():
    assert not encode_board(push(chess.Board(), "e2e4"))[P["en_passant"]].any()
    x = encode_board(push(chess.Board(), "e2e4", "g8f6", "e4e5", "d7d5"))  # exd6 e.p. is legal
    assert x[P["en_passant"]].sum() == 1 and x[P["en_passant"], 5, 3] == 1  # d6
    y = encode_board(push(chess.Board(), "g1f3", "d7d5", "f3g1", "d5d4", "e2e4"))  # ...dxe3 e.p. for Black
    assert y[P["en_passant"]].sum() == 1 and y[P["en_passant"], 5, 4] == 1  # e3 seen from Black = e6


def test_repetition_planes():
    b = push(chess.Board(), "g1f3", "g8f6", "f3g1", "f6g8")
    x = encode_board(b)
    assert x[P["repetition_1"]].all() and not x[P["repetition_2"]].any()
    x = encode_board(push(b, "g1f3", "g8f6", "f3g1", "f6g8"))
    assert x[P["repetition_1"]].all() and x[P["repetition_2"]].all()


def test_castling_rights_follow_side_to_move():
    b = push(chess.Board(), "e2e4", "e7e5", "e1e2")  # White loses both rights; Black to move
    x = encode_board(b)
    assert x[P["our_castle_kingside"]].all() and x[P["our_castle_queenside"]].all()  # Black's
    assert not x[P["their_castle_kingside"]].any() and not x[P["their_castle_queenside"]].any()


def test_halfmove_clock_scaled_and_capped():
    b = chess.Board("8/8/4k3/8/8/4K3/8/7R w - - 37 80")
    assert encode_board(b)[P["halfmove_clock"]][0, 0] == pytest.approx(0.37)
    b = chess.Board("8/8/4k3/8/8/4K3/8/7R w - - 150 80")
    assert encode_board(b)[P["halfmove_clock"]][0, 0] == 1.0


def test_batch():
    boards = random_positions(5)
    batch = encode_boards(boards)
    assert batch.shape == (5, NUM_PLANES, 8, 8)
    np.testing.assert_array_equal(batch[3], encode_board(boards[3]))
    assert encode_boards([]).shape == (0, NUM_PLANES, 8, 8)


def test_encoding_speed():
    boards = random_positions(2000, seed=1)
    t0 = time.perf_counter()
    for b in boards:
        encode_board(b)
    per_board_ms = (time.perf_counter() - t0) / len(boards) * 1000
    assert per_board_ms < 1.0, f"{per_board_ms:.3f} ms/board"


def _reference_encode(board: chess.Board) -> np.ndarray:
    """Slow, obvious implementation: loop over the squares of the oriented board."""
    from chess_rl.encoding import oriented

    view = oriented(board)
    x = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    for sq, piece in view.piece_map().items():
        offset = 0 if piece.color == chess.WHITE else 6
        x[offset + piece.piece_type - 1, chess.square_rank(sq), chess.square_file(sq)] = 1
    if view.ep_square is not None and view.has_legal_en_passant():
        x[P["en_passant"], chess.square_rank(view.ep_square), chess.square_file(view.ep_square)] = 1
    x[P["repetition_1"]] = board.is_repetition(2)
    x[P["repetition_2"]] = board.is_repetition(3)
    x[P["colour_white"]] = board.turn == chess.WHITE
    x[P["our_castle_kingside"]] = view.has_kingside_castling_rights(chess.WHITE)
    x[P["our_castle_queenside"]] = view.has_queenside_castling_rights(chess.WHITE)
    x[P["their_castle_kingside"]] = view.has_kingside_castling_rights(chess.BLACK)
    x[P["their_castle_queenside"]] = view.has_queenside_castling_rights(chess.BLACK)
    x[P["halfmove_clock"]] = min(board.halfmove_clock, 100) / 100
    return x


def test_matches_reference_implementation():
    for b in random_positions(500, seed=3, max_plies=200):
        np.testing.assert_array_equal(encode_board(b), _reference_encode(b))
