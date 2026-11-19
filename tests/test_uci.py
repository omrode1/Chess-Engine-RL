import io
import sys
import threading
import time

import chess
import chess.engine
import pytest

from chess_rl.eval import RandomPlayer
from chess_rl.play.uci import UCIEngine, parse_go, parse_position
from tests.test_players import needs_stockfish


def test_parse_position():
    assert parse_position(["startpos"]) == chess.Board()
    b = parse_position(["startpos", "moves", "e2e4", "e7e5"])
    assert b.move_stack == [chess.Move.from_uci("e2e4"), chess.Move.from_uci("e7e5")]
    fen = "8/8/4k3/8/8/4K3/8/7R w - - 0 1"
    assert parse_position(["fen", *fen.split()]).fen() == fen
    assert parse_position(["fen", *fen.split(), "moves", "h1h6"]).fen().startswith("8/8/4k2R")
    with pytest.raises(ValueError):
        parse_position(["startpos", "moves", "e2e5"])


def test_parse_go():
    assert parse_go(["wtime", "1000", "btime", "2000", "winc", "10", "binc", "20", "movestogo", "30"]) == {
        "wtime": 1000,
        "btime": 2000,
        "winc": 10,
        "binc": 20,
        "movestogo": 30,
    }
    assert parse_go(["infinite"]) == {"infinite": True}
    assert parse_go(["searchmoves", "e2e4", "movetime", "50"]) == {"movetime": 50}


class Harness:
    def __init__(self, spec="greedy"):
        self.out, self.err = io.StringIO(), io.StringIO()
        self.engine = UCIEngine(spec, seed=0, out=self.out, err=self.err)

    def send(self, *lines):
        for line in lines:
            assert self.engine.handle(line)

    def lines(self):
        return self.out.getvalue().splitlines()

    def wait_bestmove(self, timeout=5.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            moves = [ln for ln in self.lines() if ln.startswith("bestmove")]
            if moves:
                return moves[-1].split()[1]
            time.sleep(0.01)
        raise AssertionError(f"no bestmove; output: {self.lines()}")


def test_handshake_only_protocol_on_stdout():
    h = Harness()
    h.send("uci", "isready")
    out = h.lines()
    assert out[0].startswith("id name chess-rl") and "uciok" in out and out[-1] == "readyok"
    assert all(ln.split()[0] in {"id", "option", "uciok", "readyok"} for ln in out)
    assert "player 'greedy' ready" in h.err.getvalue()  # diagnostics on stderr


def test_go_returns_legal_move_for_position():
    h = Harness()
    h.send("position startpos moves e2e4 e7e5 g1f3", "go wtime 1000 btime 1000 winc 0 binc 0")
    board = parse_position(["startpos", "moves", "e2e4", "e7e5", "g1f3"])
    assert chess.Move.from_uci(h.wait_bestmove()) in board.legal_moves


def test_infinite_waits_for_stop():
    h = Harness()
    h.send("position startpos", "go infinite")
    time.sleep(0.2)
    assert not any(ln.startswith("bestmove") for ln in h.lines())
    h.send("stop")
    assert chess.Move.from_uci(h.wait_bestmove()) in chess.Board().legal_moves


def test_game_over_position_and_bad_input_do_not_crash():
    h = Harness()
    h.send("position startpos moves f2f3 e7e5 g2g4 d8h4", "go movetime 10")  # White is mated
    assert h.wait_bestmove() == "0000"
    h.send("position startpos moves e2e5")  # illegal
    assert any("error" in ln for ln in h.lines())
    h.send("bogus command", "isready")
    assert h.lines()[-1] == "readyok"


def test_setoption_switches_player():
    h = Harness("greedy")
    h.send("setoption name Player value random", "setoption name Seed value 3", "isready")
    assert isinstance(h.engine.player(), RandomPlayer)
    assert "player 'random' ready" in h.err.getvalue()


def test_quit_stops_engine():
    h = Harness()
    assert h.engine.handle("quit") is False


@pytest.mark.timeout(120)
@needs_stockfish
def test_full_game_against_stockfish_over_real_uci():
    """Our engine as a subprocess, driven by python-chess's UCI client (as a GUI or lichess-bot would)."""
    from chess_rl.eval import StockfishPlayer

    ours = chess.engine.SimpleEngine.popen_uci(
        [sys.executable, "-m", "chess_rl.cli", "uci", "--player", "greedy", "--seed", "1"]
    )
    assert ours.id["name"].startswith("chess-rl")
    board = chess.Board()
    try:
        with StockfishPlayer(elo=1320, movetime=0.02) as sf:
            while board.outcome(claim_draw=True) is None and board.ply() < 300:
                if board.turn == chess.WHITE:
                    move = ours.play(board, chess.engine.Limit(white_clock=60, black_clock=60)).move
                else:
                    move = sf.choose(board)
                assert move in board.legal_moves
                board.push(move)
    finally:
        ours.quit()
    assert board.outcome(claim_draw=True) is not None or board.ply() == 300
    assert board.ply() > 10


def test_engine_thread_safety_under_rapid_commands():
    h = Harness("random")
    for _ in range(50):
        h.send("position startpos", "go movetime 1", "isready")
    h.send("stop")
    t = threading.Thread(target=h.engine.handle, args=("quit",))
    t.start()
    t.join(5)
    assert not t.is_alive()
    assert sum(ln.startswith("bestmove") for ln in h.lines()) == 50
