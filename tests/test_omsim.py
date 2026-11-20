import json

import pytest
import torch

from chess_rl.cli import main
from chess_rl.data.personal import Collection
from chess_rl.eval import stockfish as sf
from chess_rl.eval.human import read_jsonl, write_jsonl
from chess_rl.model.human import HumanNet, HumanNetConfig, save_human
from chess_rl.train import omsim
from tests.test_personal import pgn

try:
    sf.find_stockfish()
    HAVE_STOCKFISH = True
except sf.StockfishNotFound:
    HAVE_STOCKFISH = False

CPU = torch.device("cpu")
TINY = HumanNetConfig(blocks=1, filters=8, se=False, cond_dim=8)
GAMES = [
    ("1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. Ng5 d5 5. exd5 Nxd5 6. Nxf7", "2026.01.0{}"),
    ("1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O", "2026.02.0{}"),
]


@pytest.fixture
def collection(tmp_path):
    col = Collection(tmp_path / "om", aliases={"owner"})
    for i in range(6):
        moves, date = GAMES[i % 2]
        white, black = ("Owner", "opp") if i % 2 == 0 else ("opp", "Owner")
        text = pgn(white, black, moves, "1-0", link=f"https://chess.com/game/{i}").replace(
            "2026.11.01", date.format(i + 1)
        )
        assert col.add_pgn(text, "chess.com")
    col.save()
    return col


def test_player_rows_are_the_owners_moves_in_time_order(collection):
    rows = omsim.player_rows(collection.pgn_path)
    assert [r["game_index"] for r in rows] == sorted(r["game_index"] for r in rows)
    assert len({r["game"] for r in rows}) == 6
    import chess

    for r in rows:
        board = chess.Board(r["fen"])
        # sorted by date: the three January games (owner White), then the three February ones (owner Black)
        assert board.turn == (chess.WHITE if r["game_index"] < 3 else chess.BLACK)
        assert chess.Move.from_uci(r["move"]) in board.legal_moves
        assert r["elo_us"] in (500, 480) and r["wp_loss"] == -1


def test_fit_player_changes_only_its_row_and_lowers_the_loss(collection):
    rows = omsim.player_rows(collection.pgn_path)
    torch.manual_seed(0)
    model = HumanNet(TINY)
    torch.nn.init.normal_(model.tower[0].film.weight)  # an untrained FiLM layer ignores the conditioning
    before = {k: v.clone() for k, v in model.state_dict().items()}
    player = model.add_players(1)
    curve = omsim.fit_player(model, rows, CPU, 0, omsim.OmSimConfig(steps=50, lr=0.1, l2=0.0), player)
    assert curve[-1] < curve[0]
    after = model.state_dict()
    assert torch.equal(after["player.weight"][0], before["player.weight"][0])
    assert after["player.weight"][player].abs().sum() > 0
    for k, v in before.items():
        if k != "player.weight":
            assert torch.equal(after[k], v), k


def test_fit_offset_picks_from_the_grid(collection):
    rows = omsim.player_rows(collection.pgn_path)
    assert omsim.fit_offset(HumanNet(TINY), rows, CPU, (0, 300, 600)) in (0, 300, 600)


def test_run_eval_never_trains_on_test_games(collection, monkeypatch):
    rows = omsim.player_rows(collection.pgn_path)
    seen: list[tuple[set, set]] = []
    real = omsim.evaluate_split

    def spy(make_model, train_rows, test_rows, device, conf):
        seen.append(({r["game_index"] for r in train_rows}, {r["game_index"] for r in test_rows}))
        return real(make_model, train_rows, test_rows, device, conf)

    monkeypatch.setattr(omsim, "evaluate_split", spy)
    result = omsim.run_eval(lambda: HumanNet(TINY), rows, CPU, omsim.OmSimConfig(steps=3), 0.34, 3, log=lambda s: None)
    for train_games, test_games in seen:
        assert max(train_games) < min(test_games)
    assert seen[0] == ({0, 1, 2, 3}, {4, 5})  # time-ordered split: the 2 most recent games
    assert [t for _, t in seen[1:]] == [{3}, {4}, {5}]  # forward chaining
    assert result["split"]["om-sim"]["games"] == 2 and result["forward_chaining"]["om-sim"]["games"] == 3


def test_paired_difference():
    rows = [{"id": str(i), "game": str(i // 2), "move": "e2e4"} for i in range(6)]
    good = [{"id": str(i), "top1": "e2e4"} for i in range(6)]
    bad = [{"id": str(i), "top1": "d2d4"} for i in range(6)]
    d = omsim.paired_difference(rows, good, bad)
    assert d["diff"] == 1.0 and d["ci95"] == (1.0, 1.0) and d["games"] == 3


@pytest.mark.skipif(not HAVE_STOCKFISH, reason="needs the pinned Stockfish")
def test_label_blunders_with_stockfish():
    rows = [
        {"fen": "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "move": "a1a8", "wp_loss": -1},  # mates
        {"fen": "4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1", "move": "e1f1", "wp_loss": -1},  # ignores the hanging queen
    ]
    omsim.label_blunders(rows, sf.find_stockfish(), depth=8)
    assert rows[0]["wp_loss"] == 0 and rows[1]["wp_loss"] >= 10


def test_omsim_cli_eval_and_fit(collection, tmp_path, monkeypatch):
    monkeypatch.setenv("CHESS_RL_RUNS_FILE", str(tmp_path / "runs.jsonl"))
    root = str(collection.root)
    model = tmp_path / "h.pt"
    save_human(model, HumanNet(TINY))
    assert main(["omsim", "--root", root, "eval", "--model", str(model)]) == 2  # not labelled yet
    rows = omsim.player_rows(collection.pgn_path)
    for i, r in enumerate(rows):
        r["wp_loss"] = 15 if i % 4 == 0 else 2
    write_jsonl(collection.root / "rows.jsonl", rows)
    out = tmp_path / "omsim.json"
    args = [
        "omsim",
        "--root",
        root,
        "eval",
        "--model",
        str(model),
        "--steps",
        "3",
        "--min-train",
        "4",
        "--out",
        str(out),
    ]
    assert main(args) == 0
    result = json.loads(out.read_text())
    assert {"base@raw", "base@offset", "om-sim", "om-sim_minus_base@offset"} <= set(result["split"])
    assert "predictions" not in result  # only aggregates are written
    fitted = tmp_path / "om.pt"
    assert main(["omsim", "--root", root, "fit", "--model", str(model), "--steps", "3", "--out", str(fitted)]) == 0
    from chess_rl.model.human import load_human

    loaded, meta = load_human(fitted)
    assert (
        loaded.config.players == 2
        and meta["player"] == 1
        and meta["moves"] == len(read_jsonl(collection.root / "rows.jsonl"))
    )


def test_omsim_report_cli(collection, tmp_path, monkeypatch):
    rows = omsim.player_rows(collection.pgn_path)
    for i, r in enumerate(rows):
        r.update(wp_loss=40 if i % 5 == 0 else 1, best=r["move"], refutation="", mate_after=0, pv_best=[], pv_after=[])
    write_jsonl(collection.root / "rows.jsonl", rows)
    model = tmp_path / "om.pt"
    net = HumanNet(TINY)
    net.add_players(1)
    save_human(model, net, offset=300, player=1)
    out = tmp_path / "report.html"
    args = [
        "omsim",
        "--root",
        str(collection.root),
        "report",
        "--model",
        str(model),
        "--out",
        str(out),
        "--population",
        "none",
    ]
    assert main(args) == 0
    page = out.read_text()
    assert "<svg" in page and "What Om-sim expects" in page and "positional" in page
