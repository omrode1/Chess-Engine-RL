import csv

import chess
import pytest

from chess_rl.cli import main
from chess_rl.eval import RandomPlayer
from chess_rl.eval.tactics import (
    WAC_SHA256,
    Puzzle,
    file_sha256,
    load_suite,
    parse_epd,
    run_suite,
    suite_path,
    tactics_csv_path,
)
from chess_rl.runlog import read_runs
from tests.test_players import needs_stockfish


def test_wac_file_is_the_pinned_one():
    assert file_sha256(suite_path("wac")) == WAC_SHA256


def test_wac_parses_completely():
    puzzles = load_suite("wac")
    assert len(puzzles) == 300
    assert [p.id for p in puzzles] == [f"WAC.{i:03d}" for i in range(1, 301)]
    for p in puzzles:
        assert p.board.is_valid() and p.best
        assert all(m in p.board.legal_moves for m in p.best | p.avoid)
    assert sum(bool(p.avoid) for p in puzzles) == 1


def test_mate_suites_are_exact():
    """bm must be exactly the set of mating moves (mate1) or of first moves forcing mate in two (mate2)."""
    from scripts.make_mate_suites import forces_mate_in_2, mates_in_1

    mate1, mate2 = load_suite("mate1"), load_suite("mate2")
    assert len(mate1) == len(mate2) == 150
    assert len({p.id for p in mate1 + mate2}) == 300
    for p in mate1:
        assert p.best == frozenset(mates_in_1(p.board))
    for p in mate2[:30]:  # brute force is slow-ish; the generator already checked all of them
        assert not mates_in_1(p.board)
        assert p.best == frozenset(m for m in p.board.legal_moves if forces_mate_in_2(p.board, m))


def test_solution_rules():
    board = chess.Board()
    e4, d4, a3 = (chess.Move.from_uci(u) for u in ("e2e4", "d2d4", "a2a3"))
    assert Puzzle("x", board, best=frozenset({e4, d4})).is_solved(d4)
    assert not Puzzle("x", board, best=frozenset({e4})).is_solved(a3)
    avoid_only = Puzzle("y", board, avoid=frozenset({a3}))
    assert avoid_only.is_solved(e4) and not avoid_only.is_solved(a3)
    with pytest.raises(ValueError, match="no bm/am"):
        parse_epd('8/8/4k3/8/8/4K3/8/7R w - - id "x";\n')


class _Oracle:
    """Plays a listed solution: run_suite must then score 100%."""

    name = "oracle"

    def __init__(self, puzzles):
        self.answers = {p.board.epd(): next(iter(sorted(p.best, key=lambda m: m.uci()))) for p in puzzles}

    def new_game(self):
        pass

    def choose(self, board, clock=None):
        return self.answers[board.epd()]

    def close(self):
        pass


def test_oracle_scores_full_and_random_scores_low():
    puzzles = load_suite("wac")
    assert run_suite(_Oracle(puzzles), "wac", puzzles).score == 1.0
    random_result = run_suite(RandomPlayer(seed=0), "wac", puzzles)
    assert random_result.total == 300 and random_result.score < 0.15
    assert len(random_result.failed) == 300 - len(random_result.solved)


@needs_stockfish
def test_stockfish_agrees_with_the_suite():
    """Full-strength Stockfish should solve nearly every WAC position: checks parsing and scoring."""
    from chess_rl.eval import StockfishPlayer

    puzzles = load_suite("wac")[:40]
    with StockfishPlayer(movetime=0.1) as sf:
        assert run_suite(sf, "wac", puzzles).score >= 0.9


def test_cli_records_run_and_csv(capsys):
    assert main(["tactics", "run", "--player", "greedy", "--suite", "wac"]) == 0
    assert "greedy on wac:" in capsys.readouterr().out
    (rec,) = read_runs()
    assert rec["kind"] == "tactics" and rec["outputs"]["total"] == 300
    (row,) = list(csv.DictReader(tactics_csv_path().open()))
    assert row["player"] == "greedy" and row["total"] == "300"
