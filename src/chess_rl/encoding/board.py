"""Board -> input planes, always from the perspective of the side to move.

If Black is to move, the position is mirrored vertically and the colours are swapped
(`chess.Board.mirror()`), so the network always plays "up the board" as the side to move.
The same position seen from either side produces identical planes, except for the colour plane.
Move indices (`chess_rl.encoding.moves`) use the same orientation.

Planes (index: meaning), each 8x8, row = rank 1..8 and column = file a..h after orientation:

     0-5   our pawns, knights, bishops, rooks, queens, king
     6-11  their pawns, knights, bishops, rooks, queens, king
    12     position has occurred before (repetition count >= 2)
    13     position has occurred twice before (repetition count >= 3)
    14     colour: 1.0 if the side to move is White
    15-16  our castling rights: kingside, queenside
    17-18  their castling rights: kingside, queenside
    19     en-passant target square (1.0 on that square)
    20     halfmove clock / 100 (fifty-move rule progress)
"""

from __future__ import annotations

from collections.abc import Iterable

import chess
import numpy as np

NUM_PLANES = 21
PLANE_NAMES: tuple[str, ...] = (
    *(f"our_{n}" for n in ("pawn", "knight", "bishop", "rook", "queen", "king")),
    *(f"their_{n}" for n in ("pawn", "knight", "bishop", "rook", "queen", "king")),
    "repetition_1",
    "repetition_2",
    "colour_white",
    "our_castle_kingside",
    "our_castle_queenside",
    "their_castle_kingside",
    "their_castle_queenside",
    "en_passant",
    "halfmove_clock",
)
assert len(PLANE_NAMES) == NUM_PLANES

_PIECE_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)


def oriented(board: chess.Board) -> chess.Board:
    """The board as seen by the side to move: unchanged for White, mirrored with colours swapped for Black."""
    return board if board.turn == chess.WHITE else board.mirror()


def _bitboards_to_planes(bitboards: list[int]) -> np.ndarray:
    """N 64-bit bitboards -> (N, 8, 8) uint8 planes (bit i = square i, a1 = 0)."""
    raw = np.array(bitboards, dtype="<u8").view(np.uint8)
    return np.unpackbits(raw, bitorder="little").reshape(len(bitboards), 8, 8)


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode one position as a float32 array of shape (NUM_PLANES, 8, 8)."""
    us, them = board.turn, not board.turn
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    masks = [board.pieces_mask(pt, us) for pt in _PIECE_TYPES]
    masks += [board.pieces_mask(pt, them) for pt in _PIECE_TYPES]
    has_ep = board.ep_square is not None and board.has_legal_en_passant()
    masks.append(chess.BB_SQUARES[board.ep_square] if has_ep else 0)
    if us == chess.BLACK:  # orient: flip ranks so the side to move plays up the board (same as board.mirror())
        masks = [chess.flip_vertical(m) for m in masks]
    bits = _bitboards_to_planes(masks)
    planes[0:12] = bits[0:12]
    planes[19] = bits[12]

    if board.is_repetition(2):
        planes[12] = 1.0
        if board.is_repetition(3):
            planes[13] = 1.0
    planes[14] = 1.0 if us == chess.WHITE else 0.0
    planes[15] = board.has_kingside_castling_rights(us)
    planes[16] = board.has_queenside_castling_rights(us)
    planes[17] = board.has_kingside_castling_rights(them)
    planes[18] = board.has_queenside_castling_rights(them)
    planes[20] = min(board.halfmove_clock, 100) / 100.0
    return planes


def encode_boards(boards: Iterable[chess.Board]) -> np.ndarray:
    """Encode a batch of positions: (N, NUM_PLANES, 8, 8) float32."""
    encoded = [encode_board(b) for b in boards]
    if not encoded:
        return np.zeros((0, NUM_PLANES, 8, 8), dtype=np.float32)
    return np.stack(encoded)
