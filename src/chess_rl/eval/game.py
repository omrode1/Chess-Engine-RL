"""Play one game between two Players."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import chess
import chess.pgn

from chess_rl.eval.players import Clock, Player


@dataclass(frozen=True)
class TimeControl:
    """Base time and increment per move, in seconds."""

    base: float
    inc: float = 0.0

    @classmethod
    def parse(cls, text: str) -> TimeControl:
        """'10+5' = 10 minutes + 5 seconds per move (chess.com notation); '0.5+0.1' works too."""
        minutes, _, inc = text.partition("+")
        tc = cls(float(minutes) * 60, float(inc or 0))
        if tc.base <= 0 or tc.inc < 0:
            raise ValueError(f"bad time control {text!r}")
        return tc

    def pgn(self) -> str:
        """PGN TimeControl tag: seconds+increment."""
        return f"{self.base:g}+{self.inc:g}"


@dataclass
class GameResult:
    result: str  # "1-0", "0-1", "1/2-1/2"
    # python-chess Termination name in lower case (checkmate, stalemate, threefold_repetition, ...),
    # or "max_plies" / "illegal_move" / "time_forfeit"
    termination: str
    board: chess.Board
    white: str
    black: str
    headers: dict[str, str] = field(default_factory=dict)
    clocks: list[float] = field(default_factory=list)  # mover's remaining time after each move (clocked games)

    def score_for(self, color: chess.Color) -> float:
        white_score = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[self.result]
        return white_score if color == chess.WHITE else 1.0 - white_score

    def pgn(self) -> chess.pgn.Game:
        game = chess.pgn.Game.from_board(self.board)
        if self.clocks:
            skip = len(self.board.move_stack) - len(self.clocks)  # opening moves played before the clock
            for i, node in enumerate(game.mainline()):
                if i >= skip:
                    node.set_clock(max(self.clocks[i - skip], 0.0))
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
    time_control: TimeControl | None = None,
) -> GameResult:
    """Play until the game ends (draws are claimed automatically) or max_plies, which is adjudicated a draw.

    An illegal move loses immediately. `start` may carry an opening already played (its move stack
    is kept in the PGN). With a time control, each player's thinking time is measured on the wall clock
    and passed to it as a Clock; running out loses ("time_forfeit"), or draws if the opponent has no
    mating material.
    """
    board = start.copy() if start is not None else chess.Board()
    headers = dict(headers or {})
    for p in (white, black):
        p.new_game()
    remaining = {chess.WHITE: 0.0, chess.BLACK: 0.0}
    clocks: list[float] = []
    if time_control is not None:
        remaining = {chess.WHITE: time_control.base, chess.BLACK: time_control.base}
        headers["TimeControl"] = time_control.pgn()

    def result(res: str, termination: str) -> GameResult:
        return GameResult(res, termination, board, white.name, black.name, headers, clocks)

    plies = 0
    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            return result(outcome.result(), outcome.termination.name.lower())
        if plies >= max_plies:
            return result("1/2-1/2", "max_plies")
        mover = white if board.turn == chess.WHITE else black
        clock = None
        if time_control is not None:
            clock = Clock(remaining[chess.WHITE], remaining[chess.BLACK], time_control.inc, time_control.inc)
        started = time.perf_counter()
        move = mover.choose(board.copy(), clock)
        if time_control is not None:
            remaining[board.turn] -= time.perf_counter() - started
            if remaining[board.turn] < 0:
                if board.has_insufficient_material(not board.turn):
                    return result("1/2-1/2", "time_forfeit")
                return result("0-1" if board.turn == chess.WHITE else "1-0", "time_forfeit")
            remaining[board.turn] += time_control.inc
            clocks.append(remaining[board.turn])
        if move not in board.legal_moves:
            return result("0-1" if board.turn == chess.WHITE else "1-0", "illegal_move")
        board.push(move)
        plies += 1
