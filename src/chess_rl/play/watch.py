"""Terminal game viewer: replay PGNs, list games in a PGN, or watch two players live.

uv run chess-rl watch list runs/matches/<file>.pgn
uv run chess-rl watch replay runs/matches/<file>.pgn --game 3 --delay 0.5
uv run chess-rl watch live greedy "stockfish:elo=1320,movetime=0.3"
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

import chess
import chess.pgn

CLEAR = "\033[2J\033[H"
_RESET = "\033[0m"
_LIGHT, _DARK, _LAST = "\033[48;5;180m", "\033[48;5;94m", "\033[48;5;71m"
_WHITE_FG, _BLACK_FG = "\033[38;5;231m", "\033[38;5;16m"


def render(board: chess.Board, white: str, black: str, note: str = "", color: bool = True) -> str:
    """The board as text (White at the bottom). The last move is highlighted when color=True."""
    last = board.peek() if board.move_stack else None
    lines = [f"  W: {white}", f"  B: {black}", ""]
    for rank in range(7, -1, -1):
        cells = []
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            if not color:
                cells.append(f" {piece.symbol() if piece else '.'} ")
                continue
            bg = _DARK if (rank + file) % 2 == 0 else _LIGHT
            if last and sq in (last.from_square, last.to_square):
                bg = _LAST
            sym = piece.unicode_symbol(invert_color=True) if piece else " "
            fg = _WHITE_FG if piece and piece.color == chess.WHITE else _BLACK_FG
            cells.append(f"{bg}{fg} {sym} {_RESET}")
        lines.append(f" {rank + 1} " + "".join(cells))
    lines.append("    " + "".join(f" {f} " for f in "abcdefgh"))
    lines.append("")
    lines.append(f"  move {board.fullmove_number}  {note}".rstrip())
    return "\n".join(lines)


def iter_headers(path: str | Path) -> Iterator[chess.pgn.Headers]:
    with open(path) as f:
        while (headers := chess.pgn.read_headers(f)) is not None:
            yield headers


def list_games(path: str | Path, out: TextIO | None = None) -> int:
    out = out or sys.stdout
    n = 0
    for n, h in enumerate(iter_headers(path), 1):
        out.write(
            f"{n:4d}  {h.get('White', '?'):30s} {h.get('Black', '?'):30s} {h.get('Result', '*'):8s} "
            f"{h.get('Termination', '')}\n"
        )
    return n


def read_game(path: str | Path, number: int) -> chess.pgn.Game:
    with open(path) as f:
        for _ in range(number - 1):
            if not chess.pgn.skip_game(f):
                raise ValueError(f"{path} has fewer than {number} games")
        game = chess.pgn.read_game(f)
    if game is None:
        raise ValueError(f"{path} has fewer than {number} games")
    return game


def replay(
    path: str | Path, number: int = 1, delay: float = 0.6, out: TextIO | None = None, color: bool = True
) -> None:
    out = out or sys.stdout
    game = read_game(path, number)
    board = game.board()
    white, black = game.headers.get("White", "?"), game.headers.get("Black", "?")
    out.write(CLEAR + render(board, white, black, "start", color) + "\n")
    for move in game.mainline_moves():
        san = board.san(move)
        board.push(move)
        out.write(CLEAR + render(board, white, black, san, color) + "\n")
        out.flush()
        time.sleep(delay)
    note = f"result {game.headers.get('Result', '*')} ({game.headers.get('Termination', '')})"
    out.write(CLEAR + render(board, white, black, note, color) + "\n")


def live(
    white_spec: str,
    black_spec: str,
    delay: float = 0.3,
    max_plies: int = 400,
    seed: int | None = None,
    out: TextIO | None = None,
    color: bool = True,
) -> chess.Board:
    out = out or sys.stdout
    from chess_rl.eval.registry import display_name, make_player

    seed = int(time.time()) if seed is None else seed
    white = make_player(white_spec, seed=seed, name=display_name(white_spec))
    black = make_player(black_spec, seed=seed + 1, name=display_name(black_spec))
    board = chess.Board()
    try:
        out.write(CLEAR + render(board, white.name, black.name, "start", color) + "\n")
        while board.outcome(claim_draw=True) is None and board.ply() < max_plies:
            mover = white if board.turn == chess.WHITE else black
            move = mover.choose(board.copy())
            san = board.san(move)
            board.push(move)
            out.write(CLEAR + render(board, white.name, black.name, san, color) + "\n")
            out.flush()
            time.sleep(delay)
        outcome = board.outcome(claim_draw=True)
        note = f"result {outcome.result()} ({outcome.termination.name.lower()})" if outcome else "stopped at max plies"
        out.write(CLEAR + render(board, white.name, black.name, note, color) + "\n")
    finally:
        white.close()
        black.close()
    return board
