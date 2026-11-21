"""Om-sim (M6, #36): a few-shot model of one player (the owner, or someone who opted in).

Starts from the rating-band human model (#35) and learns as little as possible from the player's games:

1. **Rating calibration.** The player's ratings come from chess.com, the model's from Lichess. One offset
   (Lichess = chess.com + offset, for both players) is fitted by maximum likelihood of the player's moves.
   "base@offset" is the rating-band model with this calibration: the strongest baseline we can give it.
2. **Player embedding.** One row of the model's player table (64 numbers) is trained on the player's moves,
   everything else frozen (as in Maia4All). L2 pulls it towards the population row (zero), so with few
   games it stays close to the base model.

Evaluation never trains on the games it scores:
- time-ordered split: the most recent `holdout` fraction of games is the test set
- forward chaining: for each game k after `min_train`, fit on games 0..k-1 and predict game k

Blunder labels for the player's moves come from Stockfish (chess.com PGNs carry no evaluations).
"""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch
from torch.nn import functional as F

from chess_rl.data.lichess import iter_games, iter_positions, time_class, win_percent
from chess_rl.encoding import encode_board
from chess_rl.encoding.moves import legal_mask, move_to_index
from chess_rl.model.human import HumanNet
from chess_rl.model.network import prepare_input


@dataclass(frozen=True)
class OmSimConfig:
    steps: int = 300
    lr: float = 0.05
    l2: float = 1e-4  # pull towards the population row (zero); rating embeddings have norm ~8
    offsets: tuple[int, ...] = tuple(range(0, 1201, 50))
    seed: int = 0


def player_rows(pgn_path: str | Path, default_elo: int = 0) -> list[dict]:
    """The player's moves (positions where the player is to move), in game order (oldest first).

    Uses the [OwnerColor] header written by `chess-rl collect`. Opponents without a rating (e.g. the bot)
    get the player's own rating.
    """
    games = []
    with open(pgn_path) as f:
        for _, text in iter_games(f):
            game = chess.pgn.read_game(io.StringIO(text))
            if game is None or game.errors:
                continue
            h = game.headers
            owner = chess.WHITE if h.get("OwnerColor") == "white" else chess.BLACK
            key = (h.get("UTCDate") or h.get("Date", ""), h.get("UTCTime") or h.get("StartTime", ""))
            games.append((key, h, game, owner))
    games.sort(key=lambda g: g[0])
    rows = []
    for index, (_, h, game, owner) in enumerate(games):
        side = "White" if owner == chess.WHITE else "Black"
        other = "Black" if owner == chess.WHITE else "White"
        own_elo = _int(h.get(f"{side}Elo")) or default_elo
        opp_elo = _int(h.get(f"{other}Elo")) or own_elo
        for board, move, clock, _ in iter_positions(game):
            if board.turn != owner:
                continue
            rows.append(
                {
                    "id": f"{h.get('GameId', index)}:{board.ply()}",
                    "game": str(h.get("GameId", index)),
                    "game_index": index,
                    "fen": board.fen(),
                    "move": move.uci(),
                    "elo_us": own_elo,
                    "elo_them": opp_elo,
                    "clock": clock,
                    "wp_loss": -1,
                    "time_class": time_class(_tc(h.get("TimeControl", "-"))) or "unknown",
                }
            )
    return rows


def label_blunders(rows: list[dict], engine_path: str | Path, depth: int = 16) -> None:
    """Fill each row's Stockfish labels at a fixed depth:

    wp_loss      win-% points the played move lost (0-100)
    cp_before    evaluation before the move, mover's view (centipawns; mate = +-10000)
    best         Stockfish's move
    refutation   the opponent's best reply to the played move ("" if the game ended)
    mate_after   moves until the opponent mates after the played move (0 if none)
    pv_best      Stockfish's line from the position (up to 8 plies, UCI)
    pv_after     Stockfish's line after the played move, starting with the refutation
    """
    limit = chess.engine.Limit(depth=depth)
    with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as engine:
        for r in rows:
            board = chess.Board(r["fen"])
            info = engine.analyse(board, limit)
            before = info["score"].pov(board.turn).score(mate_score=10_000)
            pv_best = [m.uci() for m in info.get("pv", [])[:8]]
            best = pv_best[0] if pv_best else ""
            board.push_uci(r["move"])
            refutation, mate_after, pv_after = "", 0, []
            if board.is_checkmate():
                after = 10_000
            elif board.is_game_over():
                after = 0
            else:
                info = engine.analyse(board, limit)
                score = info["score"].pov(board.turn)
                after = -score.score(mate_score=10_000)
                pv_after = [m.uci() for m in info.get("pv", [])[:8]]
                refutation = pv_after[0] if pv_after else ""
                mate_after = score.mate() if (score.mate() or 0) > 0 else 0
            r["wp_loss"] = int(round(min(max(win_percent(before) - win_percent(after), 0.0), 100.0)))
            r.update(
                cp_before=before,
                best=best,
                refutation=refutation,
                mate_after=mate_after,
                pv_best=pv_best,
                pv_after=pv_after,
            )


@dataclass
class Tensors:
    planes: torch.Tensor
    mask: torch.Tensor
    move: torch.Tensor
    elo_us: torch.Tensor
    elo_them: torch.Tensor
    clock: torch.Tensor


def to_tensors(rows: list[dict], device: torch.device, offset: int = 0) -> Tensors:
    boards = [chess.Board(r["fen"]) for r in rows]
    return Tensors(
        planes=prepare_input(torch.from_numpy(np.stack([encode_board(b) for b in boards])).to(device)),
        mask=torch.from_numpy(np.stack([legal_mask(b) for b in boards])).to(device),
        move=torch.tensor(
            [move_to_index(chess.Move.from_uci(r["move"]), b.turn) for r, b in zip(rows, boards, strict=True)],
            device=device,
        ),
        elo_us=torch.tensor([r["elo_us"] + offset for r in rows], device=device),
        elo_them=torch.tensor([r["elo_them"] + offset for r in rows], device=device),
        clock=torch.tensor([r["clock"] for r in rows], device=device),
    )


def log_probs(model: HumanNet, t: Tensors, player: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    who = None if player is None else torch.full((len(t.move),), player, device=t.move.device)
    logits, _, blunder = model(t.planes, t.elo_us, t.elo_them, t.clock, who)
    logp = torch.log_softmax(logits.float().masked_fill(~t.mask, float("-inf")), dim=1)
    return logp, blunder.float()


@torch.no_grad()
def fit_offset(model: HumanNet, rows: list[dict], device: torch.device, offsets: tuple[int, ...]) -> int:
    """The rating offset (chess.com -> Lichess) that maximises the likelihood of the player's moves."""
    model.eval()
    best, best_nll = 0, float("inf")
    for off in offsets:
        t = to_tensors(rows, device, off)
        logp, _ = log_probs(model, t, None)
        nll = -logp.gather(1, t.move[:, None]).mean().item()
        if nll < best_nll:
            best, best_nll = off, nll
    return best


def fit_player(
    model: HumanNet, rows: list[dict], device: torch.device, offset: int, conf: OmSimConfig, player: int
) -> list[float]:
    """Train only `player`'s embedding row on rows; returns the loss curve."""
    model.eval()  # batch-norm statistics stay frozen
    for p in model.parameters():
        p.requires_grad_(False)
    model.player.weight.requires_grad_(True)
    t = to_tensors(rows, device, offset)
    torch.manual_seed(conf.seed)
    optimizer = torch.optim.Adam([model.player.weight], lr=conf.lr)
    others = torch.ones(model.player.weight.shape[0], 1, device=device)
    others[player] = 0
    curve = []
    for _ in range(conf.steps):
        logp, _ = log_probs(model, t, player)
        loss = F.nll_loss(logp, t.move) + conf.l2 * model.player.weight[player].pow(2).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        model.player.weight.grad *= 1 - others  # only this player's row moves
        optimizer.step()
        curve.append(loss.item())
    model.player.weight.requires_grad_(False)
    return curve


@torch.no_grad()
def predict(model: HumanNet, rows: list[dict], device: torch.device, offset: int, player: int | None) -> list[dict]:
    model.eval()
    t = to_tensors(rows, device, offset)
    logp, blunder = log_probs(model, t, player)
    top = logp.argmax(1)
    out = []
    for i, r in enumerate(rows):
        board = chess.Board(r["fen"])
        top1 = next(m for m in board.legal_moves if move_to_index(m, board.turn) == int(top[i]))
        out.append(
            {
                "id": r["id"],
                "top1": top1.uci(),
                "p_move": float(logp[i, t.move[i]].exp()),
                "p_blunder": float(torch.sigmoid(blunder[i])),
            }
        )
    return out


def evaluate_split(
    make_model: Callable[[], HumanNet],
    train_rows: list[dict],
    test_rows: list[dict],
    device: torch.device,
    conf: OmSimConfig,
) -> dict[str, list[dict] | int]:
    """Predictions on test_rows from: base at raw ratings, base@offset (calibrated), Om-sim."""
    model = make_model()
    offset = fit_offset(model, train_rows, device, conf.offsets)
    preds: dict[str, list[dict] | int] = {
        "base@raw": predict(model, test_rows, device, 0, None),
        "base@offset": predict(model, test_rows, device, offset, None),
    }
    player = model.add_players(1)
    fit_player(model, train_rows, device, offset, conf, player)
    preds["om-sim"] = predict(model, test_rows, device, offset, player)
    preds["offset"] = offset
    return preds


def _int(value: str | None) -> int | None:
    try:
        return int(value) if value else None
    except ValueError:
        return None


def _tc(tc: str) -> str:
    return tc if "+" in tc or "/" in tc else f"{tc}+0"


METHODS = ("base@raw", "base@offset", "om-sim")


def run_eval(
    make_model: Callable[[], HumanNet],
    rows: list[dict],
    device: torch.device,
    conf: OmSimConfig,
    holdout: float = 0.2,
    min_train: int = 5,
    log: Callable[[str], None] = print,
) -> dict:
    """Time-ordered split + forward chaining. Returns predictions per method for both schemes."""
    from chess_rl.eval.human import summarize

    n_games = max(r["game_index"] for r in rows) + 1
    cut = n_games - max(1, round(holdout * n_games))
    split = evaluate_split(
        make_model,
        [r for r in rows if r["game_index"] < cut],
        [r for r in rows if r["game_index"] >= cut],
        device,
        conf,
    )
    log(f"time-ordered split: train games 0-{cut - 1}, test games {cut}-{n_games - 1}, offset {split['offset']}")
    chained: dict[str, list[dict]] = {m: [] for m in METHODS}
    offsets = []
    for k in range(min_train, n_games):
        fold = evaluate_split(
            make_model,
            [r for r in rows if r["game_index"] < k],
            [r for r in rows if r["game_index"] == k],
            device,
            conf,
        )
        offsets.append(fold["offset"])
        for m in METHODS:
            chained[m].extend(fold[m])  # type: ignore[arg-type]
        log(f"forward chaining: game {k} (offset {fold['offset']})")
    test_rows = [r for r in rows if r["game_index"] >= cut]
    chain_rows = [r for r in rows if r["game_index"] >= min_train]
    return {
        "games": n_games,
        "moves": len(rows),
        "split": {
            "train_games": cut,
            "test_games": n_games - cut,
            "offset": split["offset"],
            **{m: summarize(test_rows, split[m]) for m in METHODS},  # type: ignore[arg-type]
        },
        "forward_chaining": {
            "test_games": n_games - min_train,
            "offsets": offsets,
            **{m: summarize(chain_rows, chained[m]) for m in METHODS},
        },
        "predictions": {"split": {m: split[m] for m in METHODS}, "forward_chaining": chained},
    }


def paired_difference(rows: list[dict], a: list[dict], b: list[dict], reps: int = 2000, seed: int = 0) -> dict:
    """Move-match accuracy of a minus b, with a 95% interval from resampling games (paired)."""
    pa, pb = {p["id"]: p for p in a}, {p["id"]: p for p in b}
    rows = [r for r in rows if r["id"] in pa and r["id"] in pb]
    diff = np.array([(pa[r["id"]]["top1"] == r["move"]) - (pb[r["id"]]["top1"] == r["move"]) for r in rows], float)
    games = np.array([r["game"] for r in rows])
    uniq, inv = np.unique(games, return_inverse=True)
    sums, counts = np.bincount(inv, weights=diff), np.bincount(inv)
    rng = np.random.default_rng(seed)
    stats = [sums[p].sum() / counts[p].sum() for p in (rng.integers(0, len(uniq), len(uniq)) for _ in range(reps))]
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"diff": float(diff.mean()), "ci95": (float(lo), float(hi)), "positions": len(rows), "games": len(uniq)}
