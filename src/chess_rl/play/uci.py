"""UCI front end: serve any chess_rl Player as a UCI engine (Arena, Cute Chess, En Croissant, lichess-bot).

    uv run chess-rl uci --player greedy
    uv run chess-rl uci --player "stockfish:elo=1500,movetime=0.2"

Supported: uci, debug, isready, setoption, ucinewgame, position (startpos/fen + moves), go (wtime, btime,
winc, binc, movestogo, movetime, depth, nodes, infinite), stop, ponderhit, quit.

Search runs on a background thread so `stop`, `isready`, and `quit` are always answered. Only protocol
lines go to stdout; diagnostics go to stderr. Options:

    Player   player spec (default from --player), e.g. greedy or stockfish:elo=1500,movetime=0.1
    Seed     RNG seed for stochastic players

Engine-specific options (model path, MCTS simulations, threads) arrive with those engines (M4/M5).
"""

from __future__ import annotations

import shlex
import sys
import threading
from collections.abc import Callable
from typing import TextIO

import chess

from chess_rl import __version__
from chess_rl.eval.players import Clock, Player

ENGINE_NAME = f"chess-rl {__version__}"
ENGINE_AUTHOR = "omrode1"

_GO_INT = {"wtime", "btime", "winc", "binc", "movestogo", "movetime", "depth", "nodes", "mate"}


def parse_position(tokens: list[str]) -> chess.Board:
    """`position startpos [moves ...]` or `position fen <6 fields> [moves ...]` (tokens after 'position')."""
    if not tokens:
        raise ValueError("position: missing startpos/fen")
    if tokens[0] == "startpos":
        board, rest = chess.Board(), tokens[1:]
    elif tokens[0] == "fen":
        if "moves" in tokens:
            i = tokens.index("moves")
            fen, rest = " ".join(tokens[1:i]), tokens[i:]
        else:
            fen, rest = " ".join(tokens[1:]), []
        board = chess.Board(fen)
    else:
        raise ValueError(f"position: expected startpos or fen, got {tokens[0]!r}")
    if rest:
        if rest[0] != "moves":
            raise ValueError(f"position: unexpected {rest[0]!r}")
        for uci in rest[1:]:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                raise ValueError(f"position: illegal move {uci} in {board.fen()}")
            board.push(move)
    return board


def parse_go(tokens: list[str]) -> dict[str, int | bool]:
    out: dict[str, int | bool] = {}
    i = 0
    while i < len(tokens):
        key = tokens[i]
        if key in ("infinite", "ponder"):
            out[key] = True
            i += 1
        elif key in _GO_INT and i + 1 < len(tokens):
            out[key] = int(tokens[i + 1])
            i += 2
        else:
            i += 1  # ignore unknown tokens (searchmoves, ...)
    return out


class UCIEngine:
    def __init__(
        self,
        player_spec: str = "greedy",
        seed: int | None = None,
        make_player: Callable[[str, int | None], Player] | None = None,
        out: TextIO = sys.stdout,
        err: TextIO = sys.stderr,
    ):
        if make_player is None:
            from chess_rl.eval.registry import make_player as registry_make

            def make_player(spec: str, s: int | None) -> Player:
                return registry_make(spec, seed=s)

        self._make_player = make_player
        self.player_spec = player_spec
        self.seed = seed
        self._player: Player | None = None
        self.board = chess.Board()
        self._out, self._err = out, err
        self._lock = threading.Lock()
        self._search: threading.Thread | None = None
        self._stop = threading.Event()
        self._hold = threading.Event()  # set while `go infinite`/`ponder` must wait for stop/ponderhit
        self.debug = False

    # -- output -----------------------------------------------------------------------------------
    def send(self, line: str) -> None:
        with self._lock:
            self._out.write(line + "\n")
            self._out.flush()

    def info(self, message: str) -> None:
        self.send(f"info string {message}")

    def log(self, message: str) -> None:
        self._err.write(f"[chess-rl uci] {message}\n")
        self._err.flush()

    # -- player -----------------------------------------------------------------------------------
    def player(self) -> Player:
        if self._player is None:
            self._player = self._make_player(self.player_spec, self.seed)
            self.log(f"player {self.player_spec!r} ready")
        return self._player

    def _replace_player(self) -> None:
        if self._player is not None:
            self._player.close()
            self._player = None

    # -- commands ---------------------------------------------------------------------------------
    def handle(self, line: str) -> bool:
        """Process one input line. Returns False when the engine should exit."""
        try:
            tokens = shlex.split(line) if '"' in line else line.split()
        except ValueError:
            tokens = line.split()
        if not tokens:
            return True
        cmd, args = tokens[0], tokens[1:]
        try:
            if cmd == "uci":
                self.send(f"id name {ENGINE_NAME}")
                self.send(f"id author {ENGINE_AUTHOR}")
                self.send(f"option name Player type string default {self.player_spec}")
                self.send("option name Seed type spin default 0 min 0 max 2147483647")
                self.send("uciok")
            elif cmd == "debug":
                self.debug = bool(args) and args[0] == "on"
            elif cmd == "isready":
                self.player()  # build the player now, so a GUI's first 'go' is not slowed down
                self.send("readyok")
            elif cmd == "setoption":
                self._setoption(args)
            elif cmd == "ucinewgame":
                self._wait_search()
                self.board = chess.Board()
                self.player().new_game()
            elif cmd == "position":
                self._wait_search()
                self.board = parse_position(args)
            elif cmd == "go":
                self._go(parse_go(args))
            elif cmd in ("stop", "ponderhit"):
                self._hold.clear()
                self._stop.set()
                if cmd == "stop":
                    self._wait_search()
            elif cmd == "quit":
                self._hold.clear()
                self._stop.set()
                self._wait_search()
                self._replace_player()
                return False
            elif self.debug:
                self.info(f"unknown command {cmd!r}")
        except Exception as e:  # never die on bad input; report it the UCI way
            self.info(f"error: {e}")
            self.log(f"error handling {line!r}: {e!r}")
        return True

    def _setoption(self, args: list[str]) -> None:
        if "name" not in args:
            raise ValueError("setoption: missing name")
        i = args.index("name") + 1
        j = args.index("value") if "value" in args else len(args)
        name, value = " ".join(args[i:j]).strip().lower(), " ".join(args[j + 1 :]).strip()
        if name == "player":
            self._wait_search()
            self.player_spec = value
            self._replace_player()
        elif name == "seed":
            self.seed = int(value)
            self._replace_player()
        else:
            self.info(f"ignoring unknown option {name!r}")

    def _go(self, params: dict[str, int | bool]) -> None:
        self._wait_search()
        board = self.board.copy()
        if board.is_game_over():
            self.send("bestmove 0000")
            return
        clock = None
        if "wtime" in params and "btime" in params:
            clock = Clock(
                white=int(params["wtime"]) / 1000,
                black=int(params["btime"]) / 1000,
                white_inc=int(params.get("winc", 0)) / 1000,
                black_inc=int(params.get("binc", 0)) / 1000,
            )
        self._stop.clear()
        if params.get("infinite") or params.get("ponder"):
            self._hold.set()
        else:
            self._hold.clear()
        self._search = threading.Thread(target=self._think, args=(board, clock), daemon=True)
        self._search.start()

    def _think(self, board: chess.Board, clock: Clock | None) -> None:
        try:
            move = self.player().choose(board, clock)
            if move not in board.legal_moves:
                raise RuntimeError(f"player returned illegal move {move}")
        except Exception as e:
            self.log(f"search failed: {e!r}")
            self.info(f"error: search failed: {e}")
            move = next(iter(board.legal_moves))  # never leave the GUI without a move
        while self._hold.is_set() and not self._stop.is_set():  # UCI: no bestmove before stop in infinite mode
            self._stop.wait(0.05)
        self.send(f"bestmove {move.uci()}")

    def _wait_search(self) -> None:
        if self._search is not None:
            self._hold.clear()
            self._search.join()
            self._search = None


def serve(player_spec: str, seed: int | None = None, inp: TextIO = sys.stdin, out: TextIO = sys.stdout) -> None:
    engine = UCIEngine(player_spec, seed=seed, out=out)
    for line in inp:
        if not engine.handle(line.strip()):
            break
    else:
        engine.handle("quit")
