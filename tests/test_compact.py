import chess
import numpy as np
import torch

from chess_rl.encoding import encode_board
from chess_rl.encoding.compact import compact_board, planes_from_compact, planes_from_compact_np
from tests.test_board_encoding import random_positions


def _pack(boards):
    parts = [compact_board(b) for b in boards]
    bb = np.stack([p[0] for p in parts])
    flags = np.array([p[1] for p in parts], dtype=np.uint8)
    half = np.array([p[2] for p in parts], dtype=np.uint8)
    return bb, flags, half


def test_numpy_decode_matches_encode_board():
    boards = random_positions(600, seed=21, max_plies=200)
    boards += [chess.Board("8/8/4k3/8/8/4K3/8/7R w - - 150 80")]  # halfmove cap
    b = chess.Board()
    for u in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"):  # repetitions
        b.push_uci(u)
    boards.append(b)
    bb, flags, half = _pack(boards)
    expected = np.stack([encode_board(x) for x in boards])
    np.testing.assert_array_equal(planes_from_compact_np(bb, flags, half), expected)


def test_torch_decode_matches_numpy_including_bit_63():
    boards = random_positions(200, seed=22)
    boards.append(chess.Board("7k/8/8/8/8/8/8/K6R b - - 0 1"))  # rook on h1 -> after flip bit 63 (h8)
    bb, flags, half = _pack(boards)
    t = planes_from_compact(torch.from_numpy(bb.view(np.int64)), torch.from_numpy(flags), torch.from_numpy(half))
    np.testing.assert_array_equal(t.numpy(), planes_from_compact_np(bb, flags, half))


def test_compact_size():
    bb, flags, half = compact_board(chess.Board())
    assert bb.dtype == np.uint64 and bb.shape == (13,) and 0 <= flags < 128 and half == 0
