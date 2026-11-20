import json

import numpy as np
import pytest
import torch
import zstandard

from chess_rl.cli import main
from chess_rl.eval import human
from chess_rl.model.human import HumanNet, HumanNetConfig, save_human
from chess_rl.model.network import ChessNet, NetConfig, save_checkpoint
from tests.test_lichess_data import SCHOLARS_MATE, lichess_pgn


def test_game_rows_from_annotated_pgn():
    rows = human.game_rows(SCHOLARS_MATE, game_id="g1")
    assert [r["move"] for r in rows][:2] == ["e2e4", "e7e5"]
    assert rows[5]["wp_loss"] == 50 and rows[5]["clock"] == 290 and rows[5]["elo_us"] == 1050
    assert rows[0]["id"] == "g1:0" and rows[0]["time_class"] == "blitz"


def test_bootstrap_ci_contains_mean_and_resamples_games():
    values = np.array([1, 1, 1, 0, 0, 0], dtype=float)
    lo, hi = human.bootstrap_ci(values, np.array(["a", "a", "a", "b", "b", "b"]))
    assert lo == 0.0 and hi == 1.0  # two games: a resample is all-a, all-b, or mixed
    lo, hi = human.bootstrap_ci(np.ones(10), np.arange(10))
    assert lo == hi == 1.0


def test_summarize_with_perfect_and_wrong_predictions():
    rows = human.game_rows(SCHOLARS_MATE, game_id="g1")
    perfect = [{"id": r["id"], "top1": r["move"], "p_move": 1.0, "p_blunder": float(r["wp_loss"] >= 10)} for r in rows]
    s = human.summarize(rows, perfect)
    assert s["top1"] == 1.0 and s["nll"] == pytest.approx(0.0) and s["blunder_auc"] == 1.0
    wrong = [{"id": r["id"], "top1": "a1a2", "p_move": 0.0} for r in rows]
    s = human.summarize(rows, wrong)
    assert s["top1"] == 0.0 and "blunder_auc" not in s
    with pytest.raises(ValueError):
        human.summarize(rows, [])


@pytest.mark.parametrize("kind", ["human", "chessnet"])
def test_predict_returns_legal_moves(tmp_path, kind):
    rows = human.game_rows(SCHOLARS_MATE, game_id="g1")
    if kind == "human":
        model = HumanNet(HumanNetConfig(blocks=1, filters=8, se=False, cond_dim=4))
        save_human(tmp_path / "m.pt", model)
    else:
        save_checkpoint(tmp_path / "m.pt", ChessNet(NetConfig(blocks=1, filters=8, se=False)))
    loaded = human.load_model(tmp_path / "m.pt", torch.device("cpu"))
    preds = human.predict(loaded, rows, torch.device("cpu"), batch_size=3)
    assert len(preds) == len(rows)
    for r, p in zip(rows, preds, strict=True):
        import chess

        assert chess.Move.from_uci(p["top1"]) in chess.Board(r["fen"]).legal_moves
        assert 0 <= p["p_move"] <= 1
        assert ("p_blunder" in p) == (kind == "human")


def test_testset_and_eval_cli(tmp_path, monkeypatch):
    monkeypatch.setenv("CHESS_RL_RUNS_FILE", str(tmp_path / "runs.jsonl"))
    src = tmp_path / "db.pgn.zst"
    pgns = "".join(lichess_pgn(i, 900, 1500 if i % 2 else 2000) for i in range(20))
    src.write_bytes(zstandard.ZstdCompressor().compress(pgns.encode()))
    test_set = tmp_path / "test.jsonl"
    assert main(["human", "testset", "--url", src.as_uri(), "--games", "5", "--out", str(test_set)]) == 0
    rows = human.read_jsonl(test_set)
    assert len({r["game"] for r in rows}) == 5 and all(r["elo_them"] == 1500 for r in rows if r["elo_us"] == 900)
    assert json.loads(test_set.with_suffix(".meta.json").read_text())["games"] == 5

    save_human(tmp_path / "h.pt", HumanNet(HumanNetConfig(blocks=1, filters=8, se=False, cond_dim=4)))
    oracle = tmp_path / "oracle.jsonl"
    human.write_jsonl(oracle, [{"id": r["id"], "top1": r["move"], "p_move": 1.0} for r in rows])
    out = tmp_path / "res.json"
    args = [
        "human",
        "eval",
        "--test-set",
        str(test_set),
        "--model",
        f"tiny={tmp_path / 'h.pt'}",
        "--preds",
        f"oracle={oracle}",
        "--save-preds",
        str(tmp_path / "p"),
        "--out",
        str(out),
    ]
    assert main(args) == 0
    results = json.loads(out.read_text())["results"]
    assert results["oracle"]["top1"] == 1.0 and 0 <= results["tiny"]["top1"] < 1
    assert (tmp_path / "p" / "tiny.jsonl").exists()
