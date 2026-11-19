"""Fixed move encoding: AlphaZero's 8x8x73 = 4672 policy indices, relative to the side to move.

    index = from_square * 73 + move_type

Squares are oriented like the board planes (`chess_rl.encoding.board`): when Black is to move,
ranks are flipped, so a move always "goes up the board" for the side to move. As a result, 1. e4 and
1... e5 share an index.

Move types (73 per from-square):
     0-55  queen-like moves: direction * 7 + (distance - 1), with directions
           N, NE, E, SE, S, SW, W, NW (in oriented coordinates). This covers king moves, castling
           (the king moves two squares E/W), pawn pushes and captures, and promotions to a queen.
    56-63  knight moves, in the order of _KNIGHT_DELTAS
    64-72  underpromotions: 64 + direction * 3 + piece, with direction capture-left / push / capture-right
           and piece knight / bishop / rook

Many of the 4672 indices can never be legal (for example, moving north from rank 8). They are kept so
that the policy head has a fixed shape. Use `legal_mask` to restrict outputs to legal moves.
Only standard chess is supported, not Chess960 (castling is encoded as the king's two-square move).
"""

from __future__ import annotations

import chess
import numpy as np

NUM_MOVE_TYPES = 73
NUM_MOVES = 64 * NUM_MOVE_TYPES  # 4672

_QUEEN_DIRS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))  # (d_rank, d_file)
_KNIGHT_DELTAS = ((2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1))
_UNDERPROMO_FILE_DELTAS = (-1, 0, 1)
_UNDERPROMO_PIECES = (chess.KNIGHT, chess.BISHOP, chess.ROOK)

_QUEEN_DIR_INDEX = {d: i for i, d in enumerate(_QUEEN_DIRS)}
_KNIGHT_INDEX = {d: i for i, d in enumerate(_KNIGHT_DELTAS)}


def _orient(square: chess.Square, turn: chess.Color) -> chess.Square:
    return square if turn == chess.WHITE else chess.square_mirror(square)


def _move_type(d_rank: int, d_file: int, promotion: chess.PieceType | None) -> int:
    if promotion is not None and promotion != chess.QUEEN:
        if d_rank != 1 or d_file not in _UNDERPROMO_FILE_DELTAS:
            raise ValueError("underpromotion must move one rank forward")
        return 64 + _UNDERPROMO_FILE_DELTAS.index(d_file) * 3 + _UNDERPROMO_PIECES.index(promotion)
    if (d_rank, d_file) in _KNIGHT_INDEX:
        return 56 + _KNIGHT_INDEX[(d_rank, d_file)]
    distance = max(abs(d_rank), abs(d_file))
    if distance == 0 or (d_rank != 0 and d_file != 0 and abs(d_rank) != abs(d_file)):
        raise ValueError(f"not a queen or knight move: d_rank={d_rank}, d_file={d_file}")
    direction = _QUEEN_DIR_INDEX[(int(np.sign(d_rank)), int(np.sign(d_file)))]
    return direction * 7 + (distance - 1)


def _build_decode_tables() -> tuple[np.ndarray, np.ndarray]:
    """For every index: oriented destination square (-1 if off the board) and underpromotion piece (0 if none)."""
    to_sq = np.full(NUM_MOVES, -1, dtype=np.int16)
    promo = np.zeros(NUM_MOVES, dtype=np.int8)
    for frm in chess.SQUARES:
        r, f = chess.square_rank(frm), chess.square_file(frm)
        targets: list[tuple[int, int, int]] = []  # (move_type, d_rank, d_file)
        for di, (dr, df) in enumerate(_QUEEN_DIRS):
            targets += [(di * 7 + dist - 1, dr * dist, df * dist) for dist in range(1, 8)]
        targets += [(56 + ki, dr, df) for ki, (dr, df) in enumerate(_KNIGHT_DELTAS)]
        for ui, df in enumerate(_UNDERPROMO_FILE_DELTAS):
            for pi, piece in enumerate(_UNDERPROMO_PIECES):
                t = 64 + ui * 3 + pi
                targets.append((t, 1, df))
                promo[frm * NUM_MOVE_TYPES + t] = piece
        for t, dr, df in targets:
            if 0 <= r + dr < 8 and 0 <= f + df < 8:
                to_sq[frm * NUM_MOVE_TYPES + t] = chess.square(f + df, r + dr)
    return to_sq, promo


_TO_SQUARE, _UNDERPROMOTION = _build_decode_tables()


def _build_encode_table() -> np.ndarray:
    """(from_oriented * 64 + to_oriented) -> index for every queen-like / knight move (-1 if none)."""
    table = np.full(64 * 64, -1, dtype=np.int32)
    for index in range(NUM_MOVES):
        if index % NUM_MOVE_TYPES < 64 and _TO_SQUARE[index] >= 0:
            table[(index // NUM_MOVE_TYPES) * 64 + _TO_SQUARE[index]] = index
    return table


_ENCODE = _build_encode_table()


def move_to_index(move: chess.Move, turn: chess.Color) -> int:
    """Policy index of `move` played by the side `turn`."""
    frm, to = _orient(move.from_square, turn), _orient(move.to_square, turn)
    if move.promotion is not None and move.promotion != chess.QUEEN:
        d_rank = chess.square_rank(to) - chess.square_rank(frm)
        d_file = chess.square_file(to) - chess.square_file(frm)
        return frm * NUM_MOVE_TYPES + _move_type(d_rank, d_file, move.promotion)
    index = int(_ENCODE[frm * 64 + to])
    if index < 0:
        raise ValueError(f"not a queen or knight move: {move.uci()}")
    return index


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """The move that `index` denotes in `board` (side to move = board.turn).

    A pawn reaching the last rank with a queen-like index is a queen promotion. The result is not
    guaranteed to be legal; check with `move in board.legal_moves` or use `legal_mask`.
    """
    if not 0 <= index < NUM_MOVES:
        raise ValueError(f"index {index} out of range 0..{NUM_MOVES - 1}")
    to_oriented = int(_TO_SQUARE[index])
    if to_oriented < 0:
        raise ValueError(f"index {index} points off the board")
    turn = board.turn
    frm = _orient(index // NUM_MOVE_TYPES, turn)
    to = _orient(to_oriented, turn)
    promotion = int(_UNDERPROMOTION[index]) or None
    if promotion is None and chess.square_rank(to_oriented) == 7 and board.piece_type_at(frm) == chess.PAWN:
        promotion = chess.QUEEN
    return chess.Move(frm, to, promotion=promotion)


def legal_move_indices(board: chess.Board) -> np.ndarray:
    """Policy indices of all legal moves, in python-chess generation order (int64)."""
    turn = board.turn
    return np.fromiter((move_to_index(m, turn) for m in board.legal_moves), dtype=np.int64)


def legal_mask(board: chess.Board) -> np.ndarray:
    """Boolean mask of shape (4672,): True where the index is a legal move."""
    mask = np.zeros(NUM_MOVES, dtype=bool)
    mask[legal_move_indices(board)] = True
    return mask
