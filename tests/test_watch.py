import io

import chess

from chess_rl.cli import main
from chess_rl.eval import RandomPlayer, play_game
from chess_rl.play.watch import list_games, live, read_game, render, replay


def write_pgn(path, n=3):
    with open(path, "w") as f:
        for i in range(n):
            game = play_game(RandomPlayer(i), RandomPlayer(i + 100), max_plies=12, headers={"Round": str(i + 1)})
            print(game.pgn(), file=f, end="\n\n")


def test_render_plain_and_color():
    board = chess.Board()
    board.push_uci("e2e4")
    plain = render(board, "w", "b", "e4", color=False)
    assert " 2  P  P  P  P  .  P  P  P " in plain and " 4  .  .  .  .  P " in plain
    assert "W: w" in plain and "move 1  e4" in plain
    assert "\033[48;5;71m" in render(board, "w", "b")  # last move highlighted


def test_list_and_read(tmp_path):
    path = tmp_path / "g.pgn"
    write_pgn(path)
    out = io.StringIO()
    assert list_games(path, out) == 3
    assert len(out.getvalue().splitlines()) == 3 and "max_plies" in out.getvalue()
    assert read_game(path, 2).headers["Round"] == "2"


def test_replay_shows_every_position(tmp_path):
    path = tmp_path / "g.pgn"
    write_pgn(path, 1)
    out = io.StringIO()
    replay(path, 1, delay=0, out=out, color=False)
    frames = out.getvalue().count("\033[2J")
    assert frames == 12 + 2  # start, 12 moves, final result
    assert "result 1/2-1/2 (max_plies)" in out.getvalue()


def test_live_game_finishes():
    out = io.StringIO()
    board = live("greedy", "random", delay=0, max_plies=30, seed=1, out=out, color=False)
    assert board.ply() <= 30 and "W: greedy" in out.getvalue()


def test_cli_list(tmp_path, capsys):
    path = tmp_path / "g.pgn"
    write_pgn(path, 2)
    assert main(["watch", "list", str(path)]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 2
