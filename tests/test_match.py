import csv
import math
import random

import chess
import numpy as np
import pytest

from chess_rl.cli import main
from chess_rl.eval.elo import elo_to_score, estimate, score_to_elo
from chess_rl.eval.match import _tasks, append_elo_row, book_opening, elo_csv_path, run_match
from chess_rl.eval.registry import display_name, make_player, parse_spec
from chess_rl.openings import OpeningBook
from chess_rl.runlog import read_runs

# --- Elo ---------------------------------------------------------------------


def test_elo_curve():
    assert score_to_elo(0.5) == 0
    assert score_to_elo(0.75) == pytest.approx(190.85, abs=0.01)
    assert elo_to_score(score_to_elo(0.3)) == pytest.approx(0.3)
    assert score_to_elo(1.0) == math.inf and score_to_elo(0.0) == -math.inf


def test_estimate_interval_and_los():
    est = estimate(60, 20, 20)
    assert est.games == 100 and est.score == pytest.approx(0.7)
    assert est.elo_low < est.elo < est.elo_high
    assert est.los > 0.99
    mirrored = estimate(20, 20, 60)
    assert mirrored.elo == pytest.approx(-est.elo) and mirrored.los == pytest.approx(1 - est.los)
    assert estimate(0, 10, 0).los is None and estimate(0, 10, 0).elo == 0
    with pytest.raises(ValueError):
        estimate(0, 0, 0)


def test_interval_shrinks_with_more_games():
    small, big = estimate(6, 2, 2), estimate(600, 200, 200)
    assert (big.elo_high - big.elo_low) < (small.elo_high - small.elo_low) / 5


# --- openings and pairing ----------------------------------------------------


def test_book_openings_are_legal_and_seeded():
    book = OpeningBook()
    a = book_opening(random.Random(3), 8, book)
    assert a == book_opening(random.Random(3), 8, book)
    board = chess.Board()
    for move in a:
        assert move in board.legal_moves
        board.push(move)
    assert 1 <= len(a) <= 8


def test_games_come_in_colour_swapped_pairs():
    tasks = list(_tasks(games=7, opening_plies=6, seed=1))
    assert [t.index for t in tasks] == list(range(7))
    for first, second in zip(tasks[0::2], tasks[1::2], strict=False):
        assert first.opening == second.opening and first.a_is_white and not second.a_is_white
    assert tasks[-1].a_is_white  # odd game out


# --- registry ----------------------------------------------------------------


def test_parse_and_build_specs():
    assert parse_spec("stockfish:elo=1500, movetime=0.1") == ("stockfish", {"elo": "1500", "movetime": "0.1"})
    assert make_player("random").name == "random" and make_player("greedy", name="g").name == "g"
    assert display_name("v0:path=runs/v0.1.0/models/chess_tactical_final") == "v0:path=chess_tactical_final"
    for bad in ("nope", "random:x=1", "stockfish:color=red", "stockfish:elo"):
        with pytest.raises(ValueError):
            make_player(bad)


# --- matches -----------------------------------------------------------------


def test_match_totals_and_determinism(tmp_path):
    kwargs = {"games": 6, "opening_plies": 4, "max_plies": 60, "seed": 5}
    serial = run_match("random", "greedy", workers=1, pgn_path=tmp_path / "s.pgn", **kwargs)
    parallel = run_match("random", "greedy", workers=2, pgn_path=tmp_path / "p.pgn", **kwargs)
    assert serial.estimate.games == 6
    assert [r.result for r in serial.records] == [r.result for r in parallel.records]
    assert (tmp_path / "s.pgn").read_text() == (tmp_path / "p.pgn").read_text()
    assert (tmp_path / "s.pgn").read_text().count("[Event") == 6


def test_greedy_beats_random():
    result = run_match("greedy", "random", games=20, workers=2, opening_plies=4, max_plies=200, seed=0)
    assert result.estimate.score > 0.6, str(result.estimate)


def test_elo_csv_is_appended_with_one_header(tmp_path):
    result = run_match("random", "random", games=2, workers=1, max_plies=10)
    append_elo_row(result, 8, 10, 0)
    append_elo_row(result, 8, 10, 0)
    rows = list(csv.DictReader(elo_csv_path().open()))
    assert len(rows) == 2 and rows[0]["player_a"] == "random" and rows[0]["games"] == "2"


def test_cli_match_records_run_and_elo(tmp_path, capsys):
    code = main(
        ["match", "greedy", "random", "--games", "4", "--workers", "1", "--max-plies", "40"]
        + ["--pgn", str(tmp_path / "g.pgn")]
    )
    assert code == 0
    assert "greedy vs random" in capsys.readouterr().out
    (rec,) = read_runs()
    assert rec["kind"] == "match" and rec["outputs"]["games"] == 4
    assert len(list(csv.DictReader(elo_csv_path().open()))) == 1


# --- v0 as a player ----------------------------------------------------------


def test_v0_player_matches_v0_observation_and_plays_legal(tmp_path):
    from baseline.chess_environment import ChessEnvWithOpeningBook
    from baseline.player import v0_tensor
    from tests.test_board_encoding import random_positions
    from tests.test_training_smoke import _model

    env = ChessEnvWithOpeningBook(use_opening_book=False)
    for board in random_positions(50, seed=4):
        env.board = board
        np.testing.assert_array_equal(v0_tensor(board), env.board_to_tensor())

    _model().save(tmp_path / "v0")
    player = make_player(f"v0:path={tmp_path / 'v0'}")
    board = chess.Board()
    assert player.choose(board) in board.legal_moves


# --- robustness (regression: matches vs Stockfish deadlocked at worker exit) --


@pytest.mark.timeout(120)
def test_parallel_match_with_stockfish_finishes(caplog):
    import time

    from tests.test_players import HAVE_STOCKFISH

    if not HAVE_STOCKFISH:
        pytest.skip("Stockfish not installed")
    seen = []
    t0 = time.monotonic()
    result = run_match(
        "random",
        "stockfish:elo=1320,movetime=0.02",
        games=4,
        workers=2,
        max_plies=200,
        stall_timeout=60,
        progress=lambda rec, done, total: seen.append((done, total)),
    )
    assert result.estimate.games == 4 and result.estimate.losses >= 3  # Stockfish wins
    assert sorted(seen) == [(i, 4) for i in range(1, 5)]
    assert time.monotonic() - t0 < 60
    assert "did not exit" not in caplog.text  # workers must shut down cleanly, not be killed


class _Stuck:
    name = "stuck"

    def new_game(self):
        pass

    def choose(self, board, clock=None):
        import time

        time.sleep(3600)

    def close(self):
        pass


@pytest.mark.timeout(60)
def test_stalled_match_fails_loudly(monkeypatch):
    """A worker that never finishes must raise TimeoutError and be terminated, not waited on forever."""
    import multiprocessing
    import time

    from chess_rl.eval import registry
    from chess_rl.eval.match import _run_parallel, _tasks

    monkeypatch.setitem(registry._BUILDERS, "stuck", lambda p, seed, name: _Stuck())
    fork = multiprocessing.get_context("fork")  # fork so the worker inherits the patched registry
    monkeypatch.setattr("chess_rl.eval.match.mp.get_context", lambda method: fork)
    t0 = time.monotonic()
    with pytest.raises(TimeoutError, match="no game finished"):
        _run_parallel(list(_tasks(2, 0, 0)), "stuck", "random", 50, workers=1, stall_timeout=3, progress=None)
    assert time.monotonic() - t0 < 30
