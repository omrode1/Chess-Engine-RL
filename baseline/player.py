"""The frozen v0 DQN model as a chess_rl Player (so later engines can be measured against it)."""

from __future__ import annotations

import chess
import numpy as np

from chess_rl.eval.players import Clock

_PIECE_INDEX = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}


def v0_tensor(board: chess.Board) -> np.ndarray:
    """v0's 12x8x8 observation (absolute colours, no side to move), as in baseline.chess_environment."""
    x = np.zeros((12, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        offset = 6 if piece.color == chess.BLACK else 0
        row, col = divmod(square, 8)
        x[_PIECE_INDEX[piece.piece_type] + offset, row, col] = 1
    return x


class V0Player:
    """Plays like v0 did: argmax Q over 218 outputs, wrapped onto python-chess's legal-move list."""

    def __init__(self, path: str, name: str = "v0"):
        from stable_baselines3 import DQN

        self.name = name
        self.model = DQN.load(path, device="cpu")

    def new_game(self) -> None:
        pass

    def choose(self, board: chess.Board, clock: Clock | None = None) -> chess.Move:
        legal = list(board.legal_moves)
        action, _ = self.model.predict(v0_tensor(board), deterministic=True)
        return legal[int(action) % len(legal)]

    def close(self) -> None:
        pass
