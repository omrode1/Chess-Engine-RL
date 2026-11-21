"""Weakness report (M6, #37): where and how the player blunders, from his real games and Om-sim.

Each blunder (a move that lost >= 10 win-% points, Stockfish labels from `chess-rl omsim label`) gets
one category, from Stockfish's refutation of it:

    allowed mate            the opponent now has a forced mate
    allowed a fork          the refutation attacks two valuable pieces (or the king and one) at once
    hung the moved piece    the refutation captures the piece that just moved
    ignored a threat        the refutation captures a piece that was already attacked before the move
    exposed a piece         the refutation captures a piece the move left undefended or unblocked
    allowed a combination   nothing is taken at once, but Stockfish's line wins material within 8 plies
    missed a win            Stockfish's own move won material (within 8 plies) and he played something else
    positional              no material changes hands in either line: a slower, positional loss

plus a time-trouble flag (under 30 s on the clock). The report adds blunder rates by game phase and by
clock, and, with an Om-sim model, the blunder risk the model predicts: overall, in his blunders vs his
other moves, and the positions where it expected the most trouble.

The report contains the player's games, so it is written under data/ (git-ignored) by default.
"""

from __future__ import annotations

import html
from collections import Counter
from collections.abc import Iterable

import chess
import chess.svg

BLUNDER = 10
TIME_TROUBLE = 30
VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
CATEGORIES = (
    "allowed mate",
    "allowed a fork",
    "hung the moved piece",
    "ignored a threat",
    "exposed a piece",
    "allowed a combination",
    "missed a win",
    "positional",
)


def material_swing(fen: str, line: list[str], color: chess.Color) -> int:
    """Material (pawn units) gained by `color` along the line (stops at the first illegal move)."""
    board = chess.Board(fen)

    def balance() -> int:
        return sum(
            (1 if p.color == color else -1) * VALUES[p.piece_type]
            for p in board.piece_map().values()
            if p.piece_type != chess.KING
        )

    start = balance()
    for uci in line:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            break
        board.push(move)
    return balance() - start


def classify(row: dict) -> str:
    board = chess.Board(row["fen"])
    me = board.turn
    move = chess.Move.from_uci(row["move"])
    if row.get("mate_after", 0) > 0:
        return "allowed mate"
    after = board.copy()
    after.push(move)
    refutation = chess.Move.from_uci(row["refutation"]) if row.get("refutation") else None
    if refutation is not None and refutation in after.legal_moves:
        reply = after.copy()
        reply.push(refutation)
        attacked = [
            sq
            for sq in reply.attacks(refutation.to_square)
            if (p := reply.piece_at(sq)) is not None and p.color == me and VALUES[p.piece_type] >= 3
        ]
        if len(attacked) >= 2:
            return "allowed a fork"
        if after.is_capture(refutation):
            target = refutation.to_square
            if target == move.to_square:
                return "hung the moved piece"
            if board.is_attacked_by(not me, target):
                return "ignored a threat"
            return "exposed a piece"
    if material_swing(after.fen(), row.get("pv_after", []), me) <= -2:
        return "allowed a combination"
    if material_swing(row["fen"], row.get("pv_best", []), me) >= 2:
        return "missed a win"
    return "positional"


def phase(row: dict) -> str:
    board = chess.Board(row["fen"])
    if board.ply() < 20:
        return "opening"
    material = sum(
        VALUES[p.piece_type] for p in board.piece_map().values() if p.piece_type not in (chess.KING, chess.PAWN)
    )
    return "endgame" if material <= 26 else "middlegame"


def clock_band(row: dict) -> str:
    c = row.get("clock", -1)
    if c < 0:
        return "unknown"
    return "<30s" if c < TIME_TROUBLE else "30s-2min" if c < 120 else "2-5min" if c < 300 else ">5min"


def build(rows: list[dict], risk: dict[str, float] | None = None, population_rate: float | None = None) -> dict:
    """Aggregates for the report. `risk`: id -> Om-sim's predicted blunder probability."""
    labelled = [r for r in rows if r.get("wp_loss", -1) >= 0]
    blunders = [r for r in labelled if r["wp_loss"] >= BLUNDER]
    for r in blunders:
        r["category"] = classify(r)
    report: dict = {
        "games": len({r["game"] for r in rows}),
        "moves": len(labelled),
        "blunders": len(blunders),
        "blunder_rate": len(blunders) / max(len(labelled), 1),
        "population_rate": population_rate,
        "categories": {c: n for c in CATEGORIES if (n := sum(r["category"] == c for r in blunders))},
        "time_trouble_blunders": sum(0 <= r.get("clock", -1) < TIME_TROUBLE for r in blunders),
    }
    for name, key in (("by_phase", phase), ("by_clock", clock_band)):
        moves, bl = Counter(key(r) for r in labelled), Counter(key(r) for r in blunders)
        report[name] = {k: {"moves": n, "blunders": bl[k], "rate": bl[k] / n} for k, n in moves.items()}
    if risk:
        scored = [r for r in labelled if r["id"] in risk]
        mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
        report["risk"] = {
            "expected_rate": mean([risk[r["id"]] for r in scored]),
            "mean_on_blunders": mean([risk[r["id"]] for r in scored if r["wp_loss"] >= BLUNDER]),
            "mean_on_other_moves": mean([risk[r["id"]] for r in scored if r["wp_loss"] < BLUNDER]),
        }
    return report


# --------------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------------


def _san(fen: str, uci: str) -> str:
    board = chess.Board(fen)
    try:
        return board.san(chess.Move.from_uci(uci))
    except (ValueError, AssertionError):
        return uci


def _board_svg(row: dict, size: int = 300) -> str:
    board = chess.Board(row["fen"])
    arrows = [chess.svg.Arrow(*_squares(row["move"]), color="#d33")]
    if row.get("best") and row["best"] != row["move"]:
        arrows.append(chess.svg.Arrow(*_squares(row["best"]), color="#2a2"))
    return chess.svg.board(board, orientation=board.turn, arrows=arrows, size=size)


def _squares(uci: str) -> tuple[int, int]:
    m = chess.Move.from_uci(uci)
    return m.from_square, m.to_square


def _example(row: dict, risk: dict[str, float] | None) -> str:
    board = chess.Board(row["fen"])
    after = board.copy()
    after.push_uci(row["move"])
    move_no = f"{board.fullmove_number}{'.' if board.turn == chess.WHITE else '...'}"
    lines = [
        f"<b>{move_no} {html.escape(_san(row['fen'], row['move']))}??</b> lost {row['wp_loss']} win-% points",
        f"Better: {html.escape(_san(row['fen'], row['best']))}" if row.get("best") else "",
        f"Refutation: {html.escape(_san(after.fen(), row['refutation']))}" if row.get("refutation") else "",
        f"Clock: {row['clock'] // 60}:{row['clock'] % 60:02d}" if row.get("clock", -1) >= 0 else "",
        f"Om-sim blunder risk here: {risk[row['id']]:.0%}" if risk and row["id"] in risk else "",
    ]
    link = row["game"] if str(row["game"]).startswith("http") else ""
    if link:
        lines.append(f'<a href="{html.escape(link)}">game</a>')
    text = "<br>".join(x for x in lines if x)
    return f'<figure class="pos">{_board_svg(row)}<figcaption>{text}</figcaption></figure>'


CSS = """
:root { --bg:#fbfaf7; --fg:#1d1d1f; --muted:#6b6b70; --card:#fff; --line:#e4e2dc; --bar:#c0392b; }
@media (prefers-color-scheme: dark) { :root { --bg:#151517; --fg:#ececec; --muted:#9a9aa0; --card:#1f1f23;
  --line:#33333a; --bar:#e0705f; } }
body { background:var(--bg); color:var(--fg); font:16px/1.5 system-ui, sans-serif; margin:0 auto;
  max-width:960px; padding:24px 16px; }
h1 { margin:0 0 4px; } .sub { color:var(--muted); margin:0 0 24px; }
table { border-collapse:collapse; width:100%; margin:8px 0 24px; }
td, th { padding:6px 8px; border-bottom:1px solid var(--line); text-align:left; }
td.n { text-align:right; font-variant-numeric:tabular-nums; }
.bar { height:10px; background:var(--bar); border-radius:3px; }
.grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(300px, 1fr)); gap:16px; }
figure.pos { margin:0; background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px; }
figure.pos svg { width:100%; height:auto; } figcaption { font-size:14px; margin-top:6px; }
.note { color:var(--muted); font-size:14px; }
"""


def render_html(
    report: dict, rows: list[dict], risk: dict[str, float] | None = None, title: str = "Weakness report"
) -> str:
    blunders = [r for r in rows if r.get("wp_loss", -1) >= BLUNDER and "category" in r]
    total = max(report["blunders"], 1)
    parts = [
        f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p class='sub'>{report['games']} games, {report['moves']} moves, {report['blunders']} blunders "
        f"({report['blunder_rate']:.1%} of moves)"
        + (f"; 600–1600 Lichess players: {report['population_rate']:.1%}" if report.get("population_rate") else "")
        + "</p>",
        "<h2>How the blunders happen</h2><table><tr><th>Category</th><th class='n'>Blunders</th><th></th></tr>",
    ]
    for cat, n in sorted(report["categories"].items(), key=lambda kv: -kv[1]):
        parts.append(
            f"<tr><td>{cat}</td><td class='n'>{n}</td>"
            f"<td style='width:50%'><div class='bar' style='width:{100 * n / total:.0f}%'></div></td></tr>"
        )
    parts.append("</table>")
    parts.append(
        f"<p class='note'>{report['time_trouble_blunders']} of the blunders were played with under 30 s left.</p>"
    )
    for name, label in (("by_phase", "Game phase"), ("by_clock", "Time left")):
        parts.append(
            f"<h2>By {label.lower()}</h2><table><tr><th>{label}</th><th class='n'>Moves</th>"
            "<th class='n'>Blunders</th><th class='n'>Rate</th></tr>"
        )
        for k, v in report[name].items():
            parts.append(
                f"<tr><td>{k}</td><td class='n'>{v['moves']}</td><td class='n'>{v['blunders']}</td>"
                f"<td class='n'>{v['rate']:.0%}</td></tr>"
            )
        parts.append("</table>")
    if report.get("risk"):
        r = report["risk"]
        parts.append(
            "<h2>What Om-sim expects</h2>"
            f"<p>Predicted blunder risk on his moves: {r['expected_rate']:.1%} on average; "
            f"{r['mean_on_blunders']:.1%} on the moves that were blunders vs {r['mean_on_other_moves']:.1%} on the rest.</p>"
            "<p class='note'>Om-sim was fitted on these same games, so this is an in-sample check, not a test.</p>"
        )
    parts.append("<h2>Examples from real games</h2><p class='note'>Red: the move played. Green: Stockfish's move.</p>")
    for cat in CATEGORIES:
        examples = sorted((b for b in blunders if b["category"] == cat), key=lambda b: -b["wp_loss"])[:3]
        if examples:
            parts.append(f"<h3>{cat}</h3><div class='grid'>{''.join(_example(b, risk) for b in examples)}</div>")
    if risk:
        top = sorted((r for r in rows if r["id"] in risk), key=lambda r: -risk[r["id"]])[:6]
        parts.append("<h2>Where Om-sim saw the most danger</h2><div class='grid'>")
        for r in top:
            outcome = f"he lost {r['wp_loss']} points" if r.get("wp_loss", -1) >= 0 else ""
            parts.append(
                f"<figure class='pos'>{_board_svg(r)}<figcaption>Risk {risk[r['id']]:.0%}: he played "
                f"{html.escape(_san(r['fen'], r['move']))}; {outcome}</figcaption></figure>"
            )
        parts.append("</div>")
    parts.append("</body></html>")
    return "\n".join(parts)


def markdown_summary(report: dict) -> str:
    lines = [
        f"{report['games']} games, {report['moves']} moves, {report['blunders']} blunders ({report['blunder_rate']:.1%})",
        "",
        "| Category | Blunders |",
        "|---|---|",
        *(f"| {c} | {n} |" for c, n in sorted(report["categories"].items(), key=lambda kv: -kv[1])),
    ]
    return "\n".join(lines)


def risk_from_predictions(preds: Iterable[dict]) -> dict[str, float]:
    return {p["id"]: p["p_blunder"] for p in preds if "p_blunder" in p}
