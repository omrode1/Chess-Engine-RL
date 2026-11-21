import itertools
import os

import chess
import pytest

from chess_rl.eval import MaterialGreedyPlayer, Player, RandomPlayer, StockfishPlayer, play_game
from chess_rl.eval import stockfish as sf

try:
    sf.find_stockfish()
    HAVE_STOCKFISH = True
except sf.StockfishNotFound:
    if os.environ.get("CHESS_RL_REQUIRE_STOCKFISH"):  # set in CI: never skip silently
        raise
    HAVE_STOCKFISH = False

needs_stockfish = pytest.mark.skipif(not HAVE_STOCKFISH, reason="Stockfish not installed (chess-rl stockfish install)")


def make_players():
    players = [RandomPlayer(seed=1), MaterialGreedyPlayer(seed=2)]
    if HAVE_STOCKFISH:
        players.append(StockfishPlayer(elo=1320, movetime=0.01))
    return players


def test_players_satisfy_protocol():
    for p in make_players():
        assert isinstance(p, Player)
        p.close()


def test_every_player_completes_a_game_against_every_other():
    players = make_players()
    try:
        for white, black in itertools.permutations(players, 2):
            game = play_game(white, black, max_plies=120)
            assert game.result in {"1-0", "0-1", "1/2-1/2"}
            assert game.termination != "illegal_move"
            assert game.board.ply() <= 120
            assert game.pgn().headers["White"] == white.name
    finally:
        for p in players:
            p.close()


def test_greedy_takes_free_material_and_mates():
    board = chess.Board("4k3/8/8/3q4/8/8/8/3QK3 w - - 0 1")  # Qxd5 wins a queen
    assert MaterialGreedyPlayer(seed=0).choose(board) == chess.Move.from_uci("d1d5")
    scholar = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    assert MaterialGreedyPlayer(seed=0).choose(scholar) == chess.Move.from_uci("h5f7")  # mate beats material


def test_choose_does_not_modify_board():
    board = chess.Board()
    fen = board.fen()
    for p in (RandomPlayer(0), MaterialGreedyPlayer(0)):
        p.choose(board)
    assert board.fen() == fen


class _Illegal:
    name = "illegal"

    def new_game(self):
        pass

    def choose(self, board, clock=None):
        return chess.Move.from_uci("a1a8")

    def close(self):
        pass


def test_illegal_move_loses():
    game = play_game(_Illegal(), RandomPlayer(0))
    assert game.result == "0-1" and game.termination == "illegal_move"


def test_max_plies_is_adjudicated_draw():
    game = play_game(RandomPlayer(0), RandomPlayer(1), max_plies=10)
    assert game.result == "1/2-1/2" and game.termination == "max_plies" and game.board.ply() == 10
    assert game.score_for(chess.WHITE) == game.score_for(chess.BLACK) == 0.5


def test_start_position_and_headers():
    start = chess.Board()
    start.push_uci("e2e4")
    game = play_game(RandomPlayer(0), RandomPlayer(1), start=start, max_plies=6, headers={"Event": "t"})
    assert game.board.move_stack[0] == chess.Move.from_uci("e2e4")
    assert game.pgn().headers["Event"] == "t"


@needs_stockfish
def test_full_strength_stockfish_beats_random():
    # a node budget, not a time budget: the same game on any machine (a 0.02 s search drew once on a slow CI runner)
    with StockfishPlayer(nodes=20_000) as strong:
        game = play_game(strong, RandomPlayer(0), max_plies=300)
    assert game.result == "1-0" and game.termination == "checkmate"


@needs_stockfish
def test_pinned_version_is_used():
    assert sf.engine_name(sf.find_stockfish()) == sf.VERSION_NAME


def test_stockfish_path_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("STOCKFISH_PATH", str(tmp_path / "missing"))
    with pytest.raises(sf.StockfishNotFound):
        sf.find_stockfish()


def test_install_rejects_bad_checksum(monkeypatch, tmp_path):
    monkeypatch.setenv("CHESS_RL_HOME", str(tmp_path))

    def fake_download(url, dest):
        with open(dest, "wb") as f:
            f.write(b"not stockfish")

    monkeypatch.setattr(sf.urllib.request, "urlretrieve", fake_download)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        sf.install()
    assert not sf.pinned_path().exists()
