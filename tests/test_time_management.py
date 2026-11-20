import time

import chess
import pytest

from chess_rl.eval.game import TimeControl, play_game
from chess_rl.eval.players import Clock, RandomPlayer
from chess_rl.search.timeman import TimeManager


class SlowPlayer(RandomPlayer):
    def __init__(self, delay: float, **kw):
        super().__init__(**kw)
        self.delay = delay

    def choose(self, board, clock=None):
        time.sleep(self.delay)
        return super().choose(board, clock)


def test_parse_time_control():
    assert TimeControl.parse("10+5") == TimeControl(600, 5)
    assert TimeControl.parse("3") == TimeControl(180, 0)
    assert TimeControl.parse("0.5+0.1").pgn() == "30+0.1"
    with pytest.raises(ValueError):
        TimeControl.parse("0+1")


def test_slow_player_loses_on_time():
    game = play_game(SlowPlayer(0.05, seed=1), RandomPlayer(seed=2), time_control=TimeControl(0.2, 0.0))
    assert game.termination == "time_forfeit" and game.result == "0-1"
    assert game.headers["TimeControl"] == "0.2+0"
    assert "%clk" in str(game.pgn())


def test_flag_against_bare_king_is_a_draw():
    start = chess.Board("4k3/8/8/8/8/8/8/4K2Q b - - 0 1")  # Black (bare king) is slow and flags
    game = play_game(RandomPlayer(seed=1), SlowPlayer(0.05, seed=2), start=start, time_control=TimeControl(0.1))
    assert game.termination == "time_forfeit" and game.result == "1-0"
    start = chess.Board("4k3/8/8/8/8/8/8/4K2Q w - - 0 1")  # White flags; Black can't mate: draw
    game = play_game(SlowPlayer(0.05, seed=1), RandomPlayer(seed=2), start=start, time_control=TimeControl(0.1))
    assert game.termination == "time_forfeit" and game.result == "1/2-1/2"


def test_clock_is_passed_and_decreases():
    seen = []

    class Recorder(RandomPlayer):
        def choose(self, board, clock=None):
            seen.append(clock)
            return super().choose(board, clock)

    play_game(Recorder(seed=1), Recorder(seed=2), max_plies=10, time_control=TimeControl(60, 1))
    assert all(isinstance(c, Clock) and c.white_inc == 1 for c in seen)
    assert seen[0].white == 60 and 60 < seen[2].white <= 61  # spent a little, gained the increment


@pytest.mark.parametrize("inc", [0.1, 1.0, 5.0])
@pytest.mark.parametrize("base", [1.0, 60.0, 180.0, 600.0])
def test_budget_never_flags_with_increment(base, inc):
    """Spend the full hard limit plus the overhead every move for 200 moves: the clock must never run out."""
    tm = TimeManager(move_overhead=0.05)
    board = chess.Board()
    remaining = base
    for move_number in range(1, 201):
        board.fullmove_number = move_number
        soft, hard = tm.budget(board, Clock(remaining, remaining, inc, inc))
        assert 0 < soft <= hard < remaining
        remaining = remaining - hard - 0.05 + inc
        assert remaining > 0, f"flagged at move {move_number}"


@pytest.mark.parametrize("base", [60.0, 180.0, 600.0])
def test_sudden_death_lasts_long_games(base):
    """No increment: spending the soft budget (+ overhead) leaves time after 100 moves."""
    tm = TimeManager(move_overhead=0.05)
    board = chess.Board()
    remaining = base
    for move_number in range(1, 101):
        board.fullmove_number = move_number
        soft, hard = tm.budget(board, Clock(remaining, remaining, 0, 0))
        assert soft <= hard < remaining
        remaining -= soft + 0.05
    assert remaining > 0


def test_budget_shape():
    tm = TimeManager()
    soft, hard = tm.budget(chess.Board(), Clock(600, 600, 5, 5))
    assert 10 < soft < 25 and soft < hard <= 60  # 10+5: ~17 s on move one
    soft_fast, _ = tm.budget(chess.Board(), Clock(60, 60, 0, 0))
    assert 0.5 < soft_fast < 3


def test_mcts_player_respects_the_clock(tmp_path):
    from chess_rl.model.network import ChessNet, NetConfig, save_checkpoint
    from chess_rl.search.player import MCTSPlayer

    save_checkpoint(tmp_path / "m.pt", ChessNet(NetConfig(blocks=1, filters=8, se=False)))
    player = MCTSPlayer(str(tmp_path / "m.pt"), device="cpu")
    start = time.perf_counter()
    player.choose(chess.Board(), Clock(2.0, 2.0, 0.0, 0.0))
    took = time.perf_counter() - start
    soft, hard = player.last_budget
    assert took < hard + 0.1 and player.last.nodes > 0
    # fixed node budgets are capped by the clock too
    fixed = MCTSPlayer(str(tmp_path / "m.pt"), device="cpu", nodes=10**9)
    start = time.perf_counter()
    fixed.choose(chess.Board(), Clock(0.8, 0.8, 0.0, 0.0))
    assert time.perf_counter() - start < 0.8


def test_mcts_player_book(tmp_path):
    from chess_rl.model.network import ChessNet, NetConfig, save_checkpoint
    from chess_rl.search.player import MCTSPlayer

    save_checkpoint(tmp_path / "m.pt", ChessNet(NetConfig(blocks=1, filters=8, se=False)))
    player = MCTSPlayer(str(tmp_path / "m.pt"), device="cpu", book=True, nodes=16, seed=3)
    board = chess.Board()
    move = player.choose(board)
    assert move in board.legal_moves and player.last is None  # from the book, no search
    out_of_book = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    player.choose(out_of_book)
    assert player.last is not None


def test_match_with_clock_records_time_control(tmp_path, monkeypatch):
    import csv

    from chess_rl.eval.match import append_elo_row, run_match

    result = run_match("random", "greedy", games=2, workers=1, time_control=TimeControl(5, 0.1))
    assert len(result.records) == 2 and all("TimeControl" in r.pgn for r in result.records)
    path = tmp_path / "elo.csv"
    append_elo_row(result, 8, 400, 0, csv_path=path, time_control=TimeControl(5, 0.1))
    (row,) = csv.DictReader(path.open())
    assert row["time_control"] == "5+0.1"
    path.write_text("date,old,header\n")
    with pytest.raises(ValueError, match="columns"):
        append_elo_row(result, 8, 400, 0, csv_path=path)
