"""Build the mate1/mate2 tactics suites from the Lichess puzzle database (CC0).

    curl -sS https://database.lichess.org/lichess_db_puzzle.csv.zst | zstd -dc | head -400000 > puzzles.csv
    uv run python scripts/make_mate_suites.py puzzles.csv src/chess_rl/eval/suites

Takes the first 150 puzzles (file order) tagged mateIn1 / mateIn2 with at least 1000 plays. Each puzzle's
FEN is the position before the opponent's last move, so that move is played first. The `bm` set is
computed exactly by brute force (every move that mates in <= N), not taken from the puzzle's answer.
"""

import csv
import sys
from pathlib import Path

import chess

COUNT = 150


def mates_in_1(board: chess.Board) -> list[chess.Move]:
    found = []
    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            found.append(move)
        board.pop()
    return found


def forces_mate_in_2(board: chess.Board, move: chess.Move) -> bool:
    board.push(move)
    try:
        if board.is_checkmate():
            return True
        replies = list(board.legal_moves)
        if not replies:
            return False  # stalemate
        for reply in replies:
            board.push(reply)
            ok = bool(mates_in_1(board))
            board.pop()
            if not ok:
                return False
        return True
    finally:
        board.pop()


def main(csv_path: str, out_dir: str) -> None:
    suites: dict[str, list[str]] = {"mate1": [], "mate2": []}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            themes = row["Themes"].split()
            name = "mate1" if "mateIn1" in themes else "mate2" if "mateIn2" in themes else None
            if name is None or len(suites[name]) >= COUNT or int(row["NbPlays"]) < 1000:
                continue
            board = chess.Board(row["FEN"])
            board.push_uci(row["Moves"].split()[0])
            if name == "mate1":
                best = mates_in_1(board)
            else:
                if mates_in_1(board):
                    continue  # a mate in 1 exists: not a mate-in-2 test
                best = [m for m in board.legal_moves if forces_mate_in_2(board, m)]
            if not best:
                raise SystemExit(f"puzzle {row['PuzzleId']}: no forced mate found")
            bm = " ".join(sorted(board.san(m) for m in best))
            suites[name].append(f'{board.epd()} bm {bm}; id "lichess-{row["PuzzleId"]}";')
    for name, lines in suites.items():
        if len(lines) < COUNT:
            raise SystemExit(f"only {len(lines)} {name} puzzles; read more of the database")
        Path(out_dir, f"{name}.epd").write_text("\n".join(lines) + "\n")
        print(name, len(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
