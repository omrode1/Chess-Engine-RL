import chess

from chess_rl.eval import weakness


def row(fen: str, move: str, wp_loss: int = 30, **kw) -> dict:
    return {"id": f"{fen}:{move}", "game": "g1", "fen": fen, "move": move, "wp_loss": wp_loss, "clock": 100, **kw}


def test_allowed_mate():
    assert weakness.classify(row(chess.STARTING_FEN, "f2f3", mate_after=2)) == "allowed mate"


def test_hung_the_moved_piece():
    fen = "4k3/8/3q4/8/8/8/8/3QK3 w - - 0 1"  # Qd5?? walks into ...Qxd5
    assert weakness.classify(row(fen, "d1d5", refutation="d6d5")) == "hung the moved piece"


def test_ignored_a_threat():
    fen = "4k3/8/5b2/8/8/8/8/R3K3 w - - 0 1"  # the bishop already attacks a1; Kd2 leaves the rook there
    assert weakness.classify(row(fen, "e1d2", refutation="f6a1")) == "ignored a threat"


def test_exposed_a_piece():
    fen = "4k3/8/8/4b3/8/2N5/1R6/4K3 w - - 0 1"  # the knight on c3 shields the rook on b2 from the bishop
    assert weakness.classify(row(fen, "c3e4", refutation="e5b2")) == "exposed a piece"


def test_allowed_a_fork():
    fen = "4k3/7Q/8/4n3/8/8/8/6K1 w - - 0 1"  # Qh2?? allows ...Nf3+, forking king and queen
    assert weakness.classify(row(fen, "h7h2", refutation="e5f3")) == "allowed a fork"


def test_material_swing_and_lines():
    fen = "4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1"
    assert weakness.material_swing(fen, ["d2d5"], chess.WHITE) == 9
    assert weakness.material_swing(fen, ["d2d5", "e8e7", "a1a2"], chess.WHITE) == 9  # stops at the illegal move
    r = row(fen, "e1f1", pv_best=["d2d5"], pv_after=["e8e7"])
    assert weakness.classify(r) == "missed a win"
    r = row(fen, "e1f1", pv_after=["e8e7"], pv_best=["e1f1"])
    assert weakness.classify(r) == "positional"


def test_build_and_render():
    rows = [
        row("4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1", "e1f1", 40, best="d2d5", pv_best=["d2d5"], pv_after=[]),
        row(chess.STARTING_FEN, "e2e4", 0, best="e2e4"),
        row(chess.STARTING_FEN, "f2f3", 25, mate_after=2, best="e2e4", clock=10),
    ]
    rows[1]["id"] = "other"
    risk = {r["id"]: 0.5 for r in rows}
    report = weakness.build(rows, risk, population_rate=0.13)
    assert report["blunders"] == 2 and report["moves"] == 3
    assert report["categories"] == {"allowed mate": 1, "missed a win": 1}
    assert report["time_trouble_blunders"] == 1
    assert report["by_clock"]["<30s"] == {"moves": 1, "blunders": 1, "rate": 1.0}
    assert report["risk"]["expected_rate"] == 0.5
    page = weakness.render_html(report, rows, risk, title="T")
    assert page.startswith("<!doctype html>") and "<svg" in page and "allowed mate" in page and "Om-sim" in page
    assert "missed a win" in weakness.markdown_summary(report)
