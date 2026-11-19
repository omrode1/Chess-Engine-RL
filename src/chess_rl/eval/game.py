"""Play one game between two Players."""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import chess.pgn

from chess_rl.eval.players import Player


@dataclass
class GameResult:
    result: str  # "1-0", "0-1", "1/2-1/2"
    # python-chess Termination name in lower case (checkmate, stalemate, threefold_repetition, ...),
    # or "max_plies" / "illegal_move"
    termination: str
    board: chess.Board
    white: str
    black: str
    headers: dict[str, str] = field(default_factory=dict)

    def score_for(self, color: chess.Color) -> float:
        white_score = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[self.result]
        return white_score if color == chess.WHITE else 1.0 - white_score

    def pgn(self) -> chess.pgn.Game:
        game = chess.pgn.Game.from_board(self.board)
        game.headers.update(
            {"White": self.white, "Black": self.black, "Result": self.result, "Termination": self.termination}
        )
        game.headers.update(self.headers)
        return game


def play_game(
    white: Player,
    black: Player,
    start: chess.Board | None = None,
    max_plies: int = 400,
    headers: dict[str, str] | None = None,
) -> GameResult:
    """Play until the game ends (draws are claimed automatically) or max_plies, which is adjudicated a draw.

    An illegal move loses immediately. `start` may carry an opening already played (its move stack
    is kept in the PGN).
    """
    board = start.copy() if start is not None else chess.Board()
    for p in (white, black):
        p.new_game()
    plies = 0
    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            return GameResult(
                outcome.result(), outcome.termination.name.lower(), board, white.name, black.name, headers or {}
            )
        if plies >= max_plies:
            return GameResult("1/2-1/2", "max_plies", board, white.name, black.name, headers or {})
        mover = white if board.turn == chess.WHITE else black
        move = mover.choose(board.copy())
        if move not in board.legal_moves:
            result = "0-1" if board.turn == chess.WHITE else "1-0"
            return GameResult(result, "illegal_move", board, white.name, black.name, headers or {})
        board.push(move)
        plies += 1
