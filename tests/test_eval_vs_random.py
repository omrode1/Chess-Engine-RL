import random

from scripts.eval_vs_random import play_game, run


def test_random_vs_random_games_finish_and_are_scored():
    score = run(None, games=4, max_plies=40, seed=0, pgn_dir=None, label="random")
    assert score.games == 4
    assert all(p <= 40 for p in score.plies)
    assert 0 <= score.points <= 4


def test_agent_side_alternates_and_pgn_is_written(tmp_path):
    from tests.test_training_smoke import _model

    model = _model()
    score = run(model, games=2, max_plies=20, seed=0, pgn_dir=tmp_path, label="agent")
    pgn = (tmp_path / "agent_vs_random.pgn").read_text()
    assert pgn.count('[White "agent"]') == 1 and pgn.count('[Black "agent"]') == 1
    assert score.games == 2


def test_adjudicated_games_are_draws():
    result, board, adjudicated = play_game(None, True, random.Random(0), max_plies=6)
    assert adjudicated and result == "1/2-1/2" and board.ply() == 6
