"""Tactics test suites: does a player find the winning move?

Suites are EPD files in `chess_rl/eval/suites/` (provenance and licenses in NOTICE.md):

    wac   Win At Chess, 300 positions with published solutions (bm), one with an avoid move (am)

The level-3 "tactical" curriculum positions are deliberately not a suite: Stockfish 19 at depth 24 finds
no tactic in any of them (see chess_rl.positions).

A position counts as solved when the player's move is one of the `bm` moves (or, if a position only
lists `am`, any move except those). The tracked "hangs pieces" problem (#1) shows up as low scores here.

    uv run chess-rl tactics run --player greedy --suite wac
"""

from __future__ import annotations

import csv
import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

import chess

from chess_rl.eval.players import Player
from chess_rl.system import REPO_ROOT, git_info

SUITES = ("wac",)
WAC_SHA256 = "9bfb7b42cf5bbc7f3d60ff767ca3be460e0ae4d9110ef9a71ddf0bd7071d4d46"
TACTICS_CSV = REPO_ROOT / "results" / "tactics.csv"


@dataclass(frozen=True)
class Puzzle:
    id: str
    board: chess.Board
    best: frozenset[chess.Move] = frozenset()
    avoid: frozenset[chess.Move] = frozenset()

    def is_solved(self, move: chess.Move) -> bool:
        if self.best:
            return move in self.best and move not in self.avoid
        return move not in self.avoid


def suite_path(name: str) -> Path:
    if name not in SUITES:
        raise ValueError(f"unknown suite {name!r}; known: {SUITES}")
    return Path(str(resources.files("chess_rl.eval") / "suites" / f"{name}.epd"))


def parse_epd(text: str) -> list[Puzzle]:
    puzzles = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        board, ops = chess.Board.from_epd(line)
        best = frozenset(ops.get("bm", []))
        avoid = frozenset(ops.get("am", []))
        if not best and not avoid:
            raise ValueError(f"line {n}: no bm/am")
        puzzles.append(Puzzle(str(ops.get("id", f"#{n}")), board, best, avoid))
    return puzzles


def load_suite(name: str) -> list[Puzzle]:
    return parse_epd(suite_path(name).read_text())


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class TacticsResult:
    player: str
    suite: str
    solved: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (id, move played)

    @property
    def total(self) -> int:
        return len(self.solved) + len(self.failed)

    @property
    def score(self) -> float:
        return len(self.solved) / self.total if self.total else 0.0

    def __str__(self) -> str:
        return f"{self.player} on {self.suite}: {len(self.solved)}/{self.total} solved ({100 * self.score:.1f}%)"


def run_suite(
    player: Player,
    suite: str,
    puzzles: list[Puzzle] | None = None,
    progress: Callable[[int, int, int], None] | None = None,
) -> TacticsResult:
    """Ask the player for one move per position. progress(done, total, solved_so_far)."""
    puzzles = puzzles if puzzles is not None else load_suite(suite)
    result = TacticsResult(player.name, suite)
    for i, puzzle in enumerate(puzzles, 1):
        player.new_game()
        move = player.choose(puzzle.board.copy())
        if puzzle.is_solved(move):
            result.solved.append(puzzle.id)
        else:
            result.failed.append((puzzle.id, move.uci()))
        if progress:
            progress(i, len(puzzles), len(result.solved))
    return result


def tactics_csv_path() -> Path:
    return Path(os.environ.get("CHESS_RL_TACTICS_CSV", TACTICS_CSV))


def append_tactics_row(result: TacticsResult, csv_path: Path | None = None) -> None:
    path = csv_path or tactics_csv_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["date", "player", "suite", "solved", "total", "score", "git_sha"])
        writer.writerow(
            [
                datetime.now(UTC).date().isoformat(),
                result.player,
                result.suite,
                len(result.solved),
                result.total,
                round(result.score, 4),
                (git_info()["sha"] or "")[:12],
            ]
        )
