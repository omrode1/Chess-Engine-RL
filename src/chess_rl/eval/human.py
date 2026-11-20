"""How well does a model predict human moves? (M6: #35 rating-band model, #36 Om-sim)

A test set is a JSONL file, one row per position:

    {"id": "<game>:<ply>", "game": ..., "fen": ..., "move": "e2e4", "elo_us": 1180, "elo_them": 1220,
     "clock": 143, "wp_loss": 3, "time_class": "blitz"}

`build_test_set` takes the first N games of a Lichess month that pass a filter (use a month that no
model trained on). `predict` scores a human model (HumanNet) or a plain ChessNet on the rows; other
models (Maia-2: scripts/maia2_baseline.py) write the same prediction format:

    {"id": ..., "top1": "e2e4", "p_move": 0.31[, "p_blunder": 0.08]}

`summarize` reports move-match accuracy (the Maia metric) with a 95% bootstrap interval over games,
accuracy per rating band, and blunder ROC AUC where labels exist.
"""

from __future__ import annotations

import io
import json
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch

from chess_rl.data.lichess import GameFilter, iter_games, iter_positions, open_source, time_class
from chess_rl.encoding import encode_board
from chess_rl.encoding.moves import index_to_move, legal_mask, move_to_index
from chess_rl.model.human import BLUNDER_POINTS, HumanNet, load_human
from chess_rl.model.network import ChessNet, load_checkpoint, prepare_input
from chess_rl.train.human import RATING_BANDS, roc_auc


def game_rows(pgn_text: str, game_id: str | None = None) -> list[dict]:
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None or game.errors:
        return []
    h = game.headers
    gid = game_id or h.get("Site", "?").rsplit("/", 1)[-1]
    elos = {chess.WHITE: int(h["WhiteElo"]), chess.BLACK: int(h["BlackElo"])}
    tc = time_class(h.get("TimeControl", "-")) or "unknown"
    rows = []
    for board, move, clock, wp_loss in iter_positions(game):
        rows.append(
            {
                "id": f"{gid}:{board.ply()}",
                "game": gid,
                "fen": board.fen(),
                "move": move.uci(),
                "elo_us": elos[board.turn],
                "elo_them": elos[not board.turn],
                "clock": clock,
                "wp_loss": wp_loss,
                "time_class": tc,
            }
        )
    return rows


def build_test_set(
    url: str,
    out: str | Path,
    game_filter: GameFilter,
    games: int,
    stall_timeout: float = 120.0,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    import zstandard

    raw, last_modified = open_source(url, timeout=stall_timeout)
    text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw), encoding="utf-8", errors="replace")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = positions = seen = 0
    last = time.monotonic()
    try:
        with out.open("w") as f:
            for headers, pgn in iter_games(text):
                seen += 1
                if not game_filter.accepts(headers):
                    continue
                rows = game_rows(pgn)
                if not rows:
                    continue
                for row in rows:
                    f.write(json.dumps(row) + "\n")
                kept += 1
                positions += len(rows)
                if progress and time.monotonic() - last > 10:
                    progress(kept, positions)
                    last = time.monotonic()
                if kept >= games:
                    break
    finally:
        raw.raw.close()
    meta = {
        "source": url,
        "source_last_modified": last_modified,
        "games_seen": seen,
        "games": kept,
        "positions": positions,
        "filter": asdict(game_filter),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _batches(rows: list[dict], size: int) -> Iterator[list[dict]]:
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


@torch.no_grad()
def predict(
    model: HumanNet | ChessNet,
    rows: list[dict],
    device: torch.device,
    player: int | None = None,
    elo_override: int | None = None,
    batch_size: int = 1024,
) -> list[dict]:
    """Top-1 move, probability of the played move and (human models) blunder probability per row."""
    preds = []
    for chunk in _batches(rows, batch_size):
        boards = [chess.Board(r["fen"]) for r in chunk]
        planes = torch.from_numpy(np.stack([encode_board(b) for b in boards])).to(device)
        mask = torch.from_numpy(np.stack([legal_mask(b) for b in boards])).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            if isinstance(model, HumanNet):
                elo = torch.tensor([elo_override or r["elo_us"] for r in chunk], device=device)
                them = torch.tensor([r["elo_them"] for r in chunk], device=device)
                clock = torch.tensor([r["clock"] for r in chunk], device=device)
                who = None if player is None else torch.full((len(chunk),), player, device=device)
                logits, _, blunder = model(prepare_input(planes), elo, them, clock, who)
                p_blunder = torch.sigmoid(blunder.float()).cpu().numpy()
            else:
                logits, _ = model(prepare_input(planes))
                p_blunder = None
        probs = torch.softmax(logits.float().masked_fill(~mask, float("-inf")), dim=1).cpu().numpy()
        for i, (r, b) in enumerate(zip(chunk, boards, strict=True)):
            played = move_to_index(chess.Move.from_uci(r["move"]), b.turn)
            pred = {
                "id": r["id"],
                "top1": index_to_move(int(probs[i].argmax()), b).uci(),
                "p_move": float(probs[i, played]),
            }
            if p_blunder is not None:
                pred["p_blunder"] = float(p_blunder[i])
            preds.append(pred)
    return preds


def load_model(path: str | Path, device: torch.device) -> HumanNet | ChessNet:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data["config"].get("kind") == "human":
        return load_human(path, device)[0]
    return load_checkpoint(path, device)[0]


def bootstrap_ci(values: np.ndarray, groups: np.ndarray, reps: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% interval of the mean, resampling whole games (positions within a game are not independent)."""
    uniq, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=values)
    counts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(reps):
        pick = rng.integers(0, len(uniq), len(uniq))
        stats.append(sums[pick].sum() / counts[pick].sum())
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(rows: list[dict], preds: Iterable[dict]) -> dict:
    by_id = {p["id"]: p for p in preds}
    rows = [r for r in rows if r["id"] in by_id]
    if not rows:
        raise ValueError("no predictions match the test set")
    hit = np.array([by_id[r["id"]]["top1"] == r["move"] for r in rows], dtype=float)
    games = np.array([r["game"] for r in rows])
    out: dict = {"positions": len(rows), "games": len(set(games)), "top1": float(hit.mean())}
    out["top1_ci95"] = bootstrap_ci(hit, games)
    p_move = np.array([max(by_id[r["id"]].get("p_move", 0.0), 1e-9) for r in rows])
    out["nll"] = float(-np.log(p_move).mean())
    elo = np.array([r["elo_us"] for r in rows])
    for lo, hi in RATING_BANDS:
        m = (elo >= lo) & (elo < hi)
        if m.sum():
            out[f"top1_{lo}_{hi}"] = float(hit[m].mean())
            out[f"n_{lo}_{hi}"] = int(m.sum())
    labelled = [i for i, r in enumerate(rows) if r["wp_loss"] >= 0 and "p_blunder" in by_id[r["id"]]]
    if labelled:
        y = np.array([rows[i]["wp_loss"] >= BLUNDER_POINTS for i in labelled], dtype=float)
        s = np.array([by_id[rows[i]["id"]]["p_blunder"] for i in labelled])
        out.update(blunder_auc=roc_auc(s, y), blunder_rate=float(y.mean()), blunder_labelled=len(labelled))
    return out


def write_jsonl(path: str | Path, items: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
