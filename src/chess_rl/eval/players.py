"""Players: anything that picks a move for a position. Every engine (ours, baselines, Stockfish) is a Player.

player.new_game()
move = player.choose(board)   # board is not modified; the move must be legal
player.close()
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import chess
import chess.engine

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


@dataclass(frozen=True)
class Clock:
    """Remaining time and increments in seconds (optional; most players ignore it)."""

    white: float
    black: float
    white_inc: float = 0.0
    black_inc: float = 0.0


@runtime_checkable
class Player(Protocol):
    name: str

    def new_game(self) -> None: ...

    def choose(self, board: chess.Board, clock: Clock | None = None) -> chess.Move: ...

    def close(self) -> None: ...


class RandomPlayer:
    """Uniformly random legal move."""

    def __init__(self, seed: int | None = None, name: str = "random"):
        self.name = name
        self._rng = random.Random(seed)

    def new_game(self) -> None:
        pass

    def choose(self, board: chess.Board, clock: Clock | None = None) -> chess.Move:
        return self._rng.choice(list(board.legal_moves))

    def close(self) -> None:
        pass


def material(board: chess.Board, color: chess.Color) -> int:
    """Material of `color` minus material of the opponent."""
    total = 0
    for piece_type, value in PIECE_VALUES.items():
        total += value * (len(board.pieces(piece_type, color)) - len(board.pieces(piece_type, not color)))
    return total


class MaterialGreedyPlayer:
    """One-ply greedy: mate in one if available, else the move with the best material after it.

    Ties are broken randomly. It does not look at the reply, so it happily walks into recaptures.
    """

    def __init__(self, seed: int | None = None, name: str = "greedy"):
        self.name = name
        self._rng = random.Random(seed)

    def new_game(self) -> None:
        pass

    def choose(self, board: chess.Board, clock: Clock | None = None) -> chess.Move:
        me = board.turn
        best_score, best = None, []
        for move in board.legal_moves:
            board.push(move)
            score = 10_000 if board.is_checkmate() else material(board, me)
            board.pop()
            if best_score is None or score > best_score:
                best_score, best = score, [move]
            elif score == best_score:
                best.append(move)
        return self._rng.choice(best)

    def close(self) -> None:
        pass


class StockfishPlayer:
    """Stockfish over UCI, optionally strength-limited with UCI_Elo (1320–3190) or Skill Level (0–20).

    The search budget is fixed per move (movetime seconds and/or nodes), so results don't depend on the
    clock. One thread and a small hash by default, so several games can run in parallel on the laptop.
    """

    def __init__(
        self,
        elo: int | None = None,
        skill: int | None = None,
        movetime: float | None = 0.1,
        nodes: int | None = None,
        threads: int = 1,
        hash_mb: int = 16,
        path: str | Path | None = None,
        name: str | None = None,
    ):
        from chess_rl.eval.stockfish import find_stockfish

        if movetime is None and nodes is None:
            raise ValueError("give movetime and/or nodes")
        self.limit = chess.engine.Limit(time=movetime, nodes=nodes)
        self.name = name or (
            "stockfish" + (f"-elo{elo}" if elo else "") + (f"-skill{skill}" if skill is not None else "")
        )
        self.path = Path(path) if path else find_stockfish()
        self._engine = chess.engine.SimpleEngine.popen_uci(str(self.path))
        options: dict[str, object] = {"Threads": threads, "Hash": hash_mb}
        if elo is not None:
            options.update({"UCI_LimitStrength": True, "UCI_Elo": elo})
        if skill is not None:
            options["Skill Level"] = skill
        self._engine.configure(options)

    def new_game(self) -> None:
        self._engine.protocol.send_line("ucinewgame")

    def choose(self, board: chess.Board, clock: Clock | None = None) -> chess.Move:
        result = self._engine.play(board, self.limit)
        if result.move is None:
            raise RuntimeError(f"{self.name} returned no move for {board.fen()}")
        return result.move

    def close(self) -> None:
        self._engine.quit()

    def __enter__(self) -> StockfishPlayer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
