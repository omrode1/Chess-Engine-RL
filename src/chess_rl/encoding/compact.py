"""Compact storage form of the board planes: 13 bitboards + 1 flag byte + halfmove clock (106 bytes).

Datasets store this instead of float planes (5376 bytes/position). `planes_from_compact` (torch, runs
on the GPU during training) and `planes_from_compact_np` rebuild exactly `encode_board(board)`, which
tests check on random positions.

    bitboards uint64[13]: our P N B R Q K, their P N B R Q K, en-passant square (already oriented)
    flags     uint8:      bit0 repetition>=2, bit1 repetition>=3, bit2 white to move,
                          bit3 our O-O, bit4 our O-O-O, bit5 their O-O, bit6 their O-O-O
    halfmove  uint8:      min(halfmove clock, 100)
"""

from __future__ import annotations

import chess
import numpy as np

from chess_rl.encoding.board import _PIECE_TYPES, NUM_PLANES

NUM_BITBOARDS = 13


def compact_board(board: chess.Board) -> tuple[np.ndarray, int, int]:
    """(bitboards uint64[13], flags, halfmove) for one position."""
    us, them = board.turn, not board.turn
    masks = [board.pieces_mask(pt, us) for pt in _PIECE_TYPES]
    masks += [board.pieces_mask(pt, them) for pt in _PIECE_TYPES]
    has_ep = board.ep_square is not None and board.has_legal_en_passant()
    masks.append(chess.BB_SQUARES[board.ep_square] if has_ep else 0)
    if us == chess.BLACK:
        masks = [chess.flip_vertical(m) for m in masks]
    flags = 0
    if board.is_repetition(2):
        flags |= 1
        if board.is_repetition(3):
            flags |= 2
    if us == chess.WHITE:
        flags |= 4
    flags |= board.has_kingside_castling_rights(us) << 3
    flags |= board.has_queenside_castling_rights(us) << 4
    flags |= board.has_kingside_castling_rights(them) << 5
    flags |= board.has_queenside_castling_rights(them) << 6
    return np.array(masks, dtype=np.uint64), flags, min(board.halfmove_clock, 100)


def planes_from_compact_np(bitboards: np.ndarray, flags: np.ndarray, halfmove: np.ndarray) -> np.ndarray:
    """(N,13) uint64, (N,) uint8, (N,) uint8 -> (N, 21, 8, 8) float32."""
    n = len(bitboards)
    raw = np.ascontiguousarray(bitboards, dtype="<u8").view(np.uint8).reshape(n, NUM_BITBOARDS, 8)
    bits = np.unpackbits(raw, axis=-1, bitorder="little").reshape(n, NUM_BITBOARDS, 8, 8)
    planes = np.zeros((n, NUM_PLANES, 8, 8), dtype=np.float32)
    planes[:, 0:12] = bits[:, 0:12]
    planes[:, 19] = bits[:, 12]
    flags = np.asarray(flags, dtype=np.uint8)
    for plane, bit in ((12, 0), (13, 1), (14, 2), (15, 3), (16, 4), (17, 5), (18, 6)):
        planes[:, plane] = ((flags >> bit) & 1).astype(np.float32)[:, None, None]
    planes[:, 20] = (np.asarray(halfmove, dtype=np.float32) / 100.0)[:, None, None]
    return planes


def planes_from_compact(bitboards, flags, halfmove):
    """Torch version for the training loop: (N,13) int64 (uint64 bits), (N,) uint8, (N,) uint8 -> (N,21,8,8)."""
    import torch

    n = bitboards.shape[0]
    shifts = torch.arange(64, device=bitboards.device, dtype=torch.int64)
    bits = ((bitboards.unsqueeze(-1) >> shifts) & 1).to(torch.float32).view(n, NUM_BITBOARDS, 8, 8)
    planes = torch.zeros((n, NUM_PLANES, 8, 8), dtype=torch.float32, device=bitboards.device)
    planes[:, 0:12] = bits[:, 0:12]
    planes[:, 19] = bits[:, 12]
    flags = flags.to(torch.int64)
    for plane, bit in ((12, 0), (13, 1), (14, 2), (15, 3), (16, 4), (17, 5), (18, 6)):
        planes[:, plane] = ((flags >> bit) & 1).to(torch.float32).view(n, 1, 1)
    planes[:, 20] = (halfmove.to(torch.float32) / 100.0).view(n, 1, 1)
    return planes
