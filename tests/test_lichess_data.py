import io
import json

import chess
import chess.pgn
import numpy as np
import pytest
import zstandard

from chess_rl.cli import main
from chess_rl.data.lichess import GameFilter, build, encode_game, is_validation, iter_games, time_class
from chess_rl.encoding import encode_board, index_to_move
from chess_rl.encoding.compact import planes_from_compact_np
from chess_rl.eval import RandomPlayer, play_game
from chess_rl.runlog import read_runs


def lichess_pgn(i: int, white_elo: int, black_elo: int, tc: str = "300+3", plies: int = 40, **extra) -> str:
    game = play_game(RandomPlayer(i), RandomPlayer(i + 1000), max_plies=plies).pgn()
    result = game.headers["Result"]
    if result == "*" or game.headers.get("Termination") == "max_plies":
        result = ["1-0", "0-1", "1/2-1/2"][i % 3]
    game.headers.clear()
    game.headers.update(
        {
            "Event": "Rated Blitz game",
            "Site": f"https://lichess.org/g{i:07d}",
            "White": f"w{i}",
            "Black": f"b{i}",
            "Result": result,
            "WhiteElo": str(white_elo),
            "BlackElo": str(black_elo),
            "TimeControl": tc,
            "Termination": "Normal",
            **extra,
        }
    )
    return str(game) + "\n\n"


def test_time_class():
    assert time_class("60+0") == "bullet" and time_class("180+2") == "blitz" and time_class("600+5") == "rapid"
    assert time_class("1800+0") == "classical" and time_class("15+0") == "ultrabullet" and time_class("-") is None


def test_filter():
    f = GameFilter(min_elo=1800)
    ok = {"WhiteElo": "1900", "BlackElo": "2000", "Result": "1-0", "TimeControl": "300+3", "Termination": "Normal"}
    assert f.accepts(ok)
    assert not f.accepts({**ok, "BlackElo": "1700"})
    assert not f.accepts({**ok, "TimeControl": "60+0"})  # bullet
    assert not f.accepts({**ok, "Termination": "Abandoned"})
    assert not f.accepts({**ok, "WhiteElo": "?"})
    assert not f.accepts({**ok, "Result": "*"})
    assert GameFilter(min_elo=1000, max_elo=1600).accepts({**ok, "WhiteElo": "1500", "BlackElo": "1100"})


def test_iter_games_splits_and_keeps_text():
    text = "".join(lichess_pgn(i, 2000, 2000) for i in range(5))
    games = list(iter_games(io.StringIO(text)))
    assert len(games) == 5 and games[2][0]["Site"].endswith("g0000002")
    assert chess.pgn.read_game(io.StringIO(games[3][1])).headers["White"] == "w3"


def test_encode_game_positions_and_targets():
    pgn = lichess_pgn(7, 2100, 1900, plies=30)
    rows = encode_game(pgn)
    game = chess.pgn.read_game(io.StringIO(pgn))
    board, moves = game.board(), list(game.mainline_moves())
    assert len(rows["move"]) == len(moves)
    planes = planes_from_compact_np(rows["bitboards"], rows["flags"], rows["halfmove"])
    result = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}[game.headers["Result"]]
    for k, move in enumerate(moves):
        np.testing.assert_array_equal(planes[k], encode_board(board))
        assert index_to_move(int(rows["move"][k]), board) == move
        assert rows["result"][k] == (result if board.turn == chess.WHITE else -result)
        assert rows["elo_us"][k] == (2100 if board.turn == chess.WHITE else 1900)
        board.push(move)
    assert rows["base_time"][0] == 300 and rows["increment"][0] == 3 and rows["ply"][-1] == len(moves) - 1


def test_validation_split_is_deterministic():
    h = {"Site": "https://lichess.org/abcdefgh"}
    assert is_validation(h, 50) == is_validation(dict(h), 50)
    share = sum(is_validation({"Site": f"https://lichess.org/{i}"}, 50) for i in range(5000)) / 5000
    assert 0.01 < share < 0.03


def _zst_file(tmp_path, games: list[str]):
    path = tmp_path / "db.pgn.zst"
    path.write_bytes(zstandard.ZstdCompressor().compress("".join(games).encode()))
    return path


@pytest.mark.timeout(120)
def test_build_end_to_end_from_file_url(tmp_path):
    games = [lichess_pgn(i, 2000 if i % 3 else 1500, 2000, tc="60+0" if i % 7 == 0 else "300+3") for i in range(120)]
    src = _zst_file(tmp_path, games)
    out = tmp_path / "ds"
    seen = []
    manifest = build(
        src.as_uri(),
        out,
        GameFilter(min_elo=1800),
        target_positions=10**9,
        shard_size=500,
        val_every=10,
        workers=2,
        progress=seen.append,
        progress_every=0,
    )
    stats = manifest["stats"]
    expected = sum(1 for i in range(120) if i % 3 and i % 7)
    assert stats["games_seen"] == 120 and stats["games_accepted"] == expected
    assert stats["train_positions"] > 0 and stats["val_positions"] > 0
    shard = np.load(out / manifest["train_shards"][0])
    assert shard["bitboards"].shape[1] == 13 and shard["move"].dtype == np.int16
    assert seen and json.loads((out / "manifest.json").read_text())["source"] == src.as_uri()
    with pytest.raises(FileExistsError):
        build(src.as_uri(), out, GameFilter(), target_positions=10)


@pytest.mark.timeout(120)
def test_build_stops_at_target(tmp_path):
    src = _zst_file(tmp_path, [lichess_pgn(i, 2000, 2000) for i in range(200)])
    manifest = build(src.as_uri(), tmp_path / "ds", GameFilter(), target_positions=300, workers=2)
    total = manifest["stats"]["train_positions"] + manifest["stats"]["val_positions"]
    assert 300 <= total < 300 + 64 * 40 * 8  # stops within the in-flight window


@pytest.mark.timeout(120)
def test_cli_build_and_stats(tmp_path, capsys):
    src = _zst_file(tmp_path, [lichess_pgn(i, 2000, 2000) for i in range(20)])
    args = ["data", "build", "--url", src.as_uri(), "--set", f"out_root={tmp_path}", "--set", "name=t"]
    args += ["--set", "workers=2", "--set", "target_positions=100000"]
    assert main(args) == 0
    (rec,) = read_runs()
    assert rec["kind"] == "data-build" and rec["outputs"]["games_accepted"] == 20
    assert main(["data", "stats", str(tmp_path / "t")]) == 0
    assert '"games_accepted": 20' in capsys.readouterr().out
