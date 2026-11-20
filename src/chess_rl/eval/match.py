"""Matches between two players: paired openings, parallel games, PGN output, Elo estimate.

Games come in pairs. Each pair starts from the same opening (a seeded random walk through the opening
book) and the players swap colours, which cancels most of the opening's bias. Players are built from
spec strings inside worker processes, so engines like Stockfish are never pickled.
"""

from __future__ import annotations

import csv
import io
import logging
import multiprocessing as mp
import multiprocessing.util
import os
import random
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import chess

from chess_rl.eval.elo import EloEstimate, estimate
from chess_rl.eval.game import TimeControl, play_game
from chess_rl.eval.registry import display_name, make_player
from chess_rl.openings import OpeningBook
from chess_rl.system import REPO_ROOT, git_info

log = logging.getLogger(__name__)

ELO_CSV = REPO_ROOT / "results" / "elo.csv"


def elo_csv_path() -> Path:
    """results/elo.csv, or $CHESS_RL_ELO_CSV (tests and scratch experiments)."""
    return Path(os.environ.get("CHESS_RL_ELO_CSV", ELO_CSV))


ELO_COLUMNS = [
    "date",
    "player_a",
    "player_b",
    "games",
    "wins",
    "draws",
    "losses",
    "score",
    "elo",
    "elo_low",
    "elo_high",
    "los",
    "opening_plies",
    "max_plies",
    "seed",
    "git_sha",
    "time_control",
]


def book_opening(rng: random.Random, max_plies: int, book: OpeningBook | None = None) -> list[chess.Move]:
    """Random walk through the opening book (weighted like OpeningBook.get_move), up to max_plies."""
    book = book or OpeningBook()
    board, moves = chess.Board(), []
    while len(moves) < max_plies:
        entries = book.book.get(board.board_fen())
        if not entries:
            break
        move = rng.choices([m for m, _ in entries], weights=[w for _, w in entries])[0]
        board.push(move)
        moves.append(move)
    return moves


@dataclass(frozen=True)
class GameTask:
    index: int
    a_is_white: bool
    opening: tuple[str, ...]  # UCI moves
    seed: int


@dataclass(frozen=True)
class GameRecord:
    index: int
    a_is_white: bool
    score_a: float
    result: str
    termination: str
    plies: int
    pgn: str


# Per-worker player cache: building Stockfish for every game would waste ~0.3 s each time.
_WORKER: dict[str, object] = {}


def _init_worker(spec_a: str, spec_b: str, max_plies: int, time_control: TimeControl | None = None) -> None:
    _WORKER.update(spec_a=spec_a, spec_b=spec_b, max_plies=max_plies, time_control=time_control, players={})
    # Close engines when the worker exits. python-chess runs each engine on a non-daemon thread, and an
    # unclosed engine blocks the worker's interpreter shutdown forever (this deadlocked matches vs
    # Stockfish). multiprocessing runs these finalizers before it joins threads.
    multiprocessing.util.Finalize(None, _close_players, exitpriority=10)


def _close_players() -> None:
    players = _WORKER.get("players") or {}
    assert isinstance(players, dict)
    for player in players.values():
        try:
            player.close()
        except Exception:  # best effort during shutdown
            log.exception("closing %s failed", getattr(player, "name", player))
    players.clear()


def _player(key: str, seed: int):
    players = _WORKER["players"]
    assert isinstance(players, dict)
    if key not in players:
        spec = _WORKER[f"spec_{key}"]
        assert isinstance(spec, str)
        players[key] = make_player(spec, seed=seed, name=display_name(spec))
    return players[key]


def _play(task: GameTask) -> GameRecord:
    a, b = _player("a", task.seed), _player("b", task.seed + 1)
    for player, s in ((a, task.seed), (b, task.seed + 1)):
        if hasattr(player, "reseed"):
            player.reseed(s)
    start = chess.Board()
    for uci in task.opening:
        start.push_uci(uci)
    white, black = (a, b) if task.a_is_white else (b, a)
    max_plies = _WORKER["max_plies"]
    assert isinstance(max_plies, int)
    game = play_game(
        white,
        black,
        start=start,
        max_plies=max_plies,
        headers={"Round": str(task.index + 1)},
        time_control=_WORKER.get("time_control"),  # type: ignore[arg-type]
    )
    score_a = game.score_for(chess.WHITE if task.a_is_white else chess.BLACK)
    out = io.StringIO()
    print(game.pgn(), file=out, end="\n\n")
    return GameRecord(
        task.index, task.a_is_white, score_a, game.result, game.termination, game.board.ply(), out.getvalue()
    )


def _tasks(games: int, opening_plies: int, seed: int) -> Iterator[GameTask]:
    rng, book = random.Random(seed), OpeningBook()
    for pair in range((games + 1) // 2):
        opening = tuple(m.uci() for m in book_opening(rng, opening_plies, book))
        for k in range(2):
            index = 2 * pair + k
            if index < games:
                yield GameTask(index, a_is_white=(k == 0), opening=opening, seed=seed * 100_003 + index)


@dataclass
class MatchResult:
    player_a: str
    player_b: str
    estimate: EloEstimate
    records: list[GameRecord]
    pgn_path: Path | None

    def terminations(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.records:
            out[r.termination] = out.get(r.termination, 0) + 1
        return out


def run_match(
    spec_a: str,
    spec_b: str,
    games: int = 100,
    workers: int = 4,
    opening_plies: int = 8,
    max_plies: int = 400,
    seed: int = 0,
    pgn_path: str | Path | None = None,
    stall_timeout: float = 300.0,
    progress: Callable[[GameRecord, int, int], None] | None = None,
    time_control: TimeControl | None = None,
) -> MatchResult:
    """Play `games` games between two player specs.

    stall_timeout: abort with TimeoutError if no game finishes for this many seconds (a hung engine or
    worker must fail loudly, not wait forever). progress(record, done, total) is called after each game.
    time_control: play with real clocks (players get a Clock; flagging loses). Without it, players use
    their own fixed budgets (Stockfish always does: its movetime/nodes, so its strength stays calibrated).
    With a clock, the stall timeout is raised to cover the longest possible game.
    """
    if time_control is not None:
        stall_timeout = max(stall_timeout, 2 * (time_control.base + time_control.inc * max_plies / 2) + 60)
    tasks = list(_tasks(games, opening_plies, seed))
    if workers <= 1:
        _WORKER.update(spec_a=spec_a, spec_b=spec_b, max_plies=max_plies, time_control=time_control, players={})
        records = []
        try:
            for t in tasks:
                records.append(_play(t))
                if progress:
                    progress(records[-1], len(records), len(tasks))
        finally:
            _close_players()
    else:
        records = _run_parallel(tasks, spec_a, spec_b, max_plies, workers, stall_timeout, progress, time_control)
    records.sort(key=lambda r: r.index)
    wins = sum(r.score_a == 1.0 for r in records)
    draws = sum(r.score_a == 0.5 for r in records)
    losses = sum(r.score_a == 0.0 for r in records)
    path = Path(pgn_path) if pgn_path else None
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(r.pgn for r in records))
    return MatchResult(display_name(spec_a), display_name(spec_b), estimate(wins, draws, losses), records, path)


def _run_parallel(
    tasks: list[GameTask],
    spec_a: str,
    spec_b: str,
    max_plies: int,
    workers: int,
    stall_timeout: float,
    progress: Callable[[GameRecord, int, int], None] | None,
    time_control: TimeControl | None = None,
) -> list[GameRecord]:
    pool = ProcessPoolExecutor(
        workers,
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
        initargs=(spec_a, spec_b, max_plies, time_control),
    )
    processes = []
    records: list[GameRecord] = []
    completed = False
    try:
        pending = {pool.submit(_play, t) for t in tasks}
        processes = list(pool._processes.values())  # private, but the only handle for killing stuck workers
        while pending:
            done, pending = wait(pending, timeout=stall_timeout, return_when=FIRST_COMPLETED)
            if not done:
                raise TimeoutError(
                    f"no game finished in {stall_timeout:.0f}s ({len(records)}/{len(tasks)} done): "
                    "a player or worker is stuck"
                )
            for future in done:
                records.append(future.result())
                if progress:
                    progress(records[-1], len(records), len(tasks))
        completed = True
    finally:
        if not completed:  # stall or error: kill workers first so shutdown never waits on a stuck one
            for proc in processes:
                proc.terminate()
        pool.shutdown(wait=False, cancel_futures=True)
        # The executor's manager thread joins the workers (they close their engines on exit). Wait for
        # it with a bound instead of joining the processes ourselves, which races with that thread.
        manager = getattr(pool, "_executor_manager_thread", None)
        if manager is not None:
            manager.join(timeout=30)
            if manager.is_alive():
                log.warning("match workers did not exit within 30s; terminating them")
                for proc in processes:
                    proc.terminate()
    return records


def append_elo_row(
    result: MatchResult,
    opening_plies: int,
    max_plies: int,
    seed: int,
    csv_path: Path | None = None,
    time_control: TimeControl | None = None,
) -> None:
    csv_path = csv_path or elo_csv_path()
    est = result.estimate
    row = {
        "date": datetime.now(UTC).date().isoformat(),
        "player_a": result.player_a,
        "player_b": result.player_b,
        "games": est.games,
        "wins": est.wins,
        "draws": est.draws,
        "losses": est.losses,
        "score": round(est.score, 4),
        "elo": round(est.elo, 1),
        "elo_low": round(est.elo_low, 1),
        "elo_high": round(est.elo_high, 1),
        "los": round(est.los, 4) if est.los is not None else "",
        "opening_plies": opening_plies,
        "max_plies": max_plies,
        "seed": seed,
        "git_sha": (git_info()["sha"] or "")[:12],
        "time_control": time_control.pgn() if time_control else "",
    }
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    if not new:
        with csv_path.open(newline="") as f:
            header = next(csv.reader(f), [])
        if header != ELO_COLUMNS:
            raise ValueError(f"{csv_path} has columns {header}, expected {ELO_COLUMNS}")
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ELO_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)
