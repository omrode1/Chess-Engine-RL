"""Train the human move model (M6, #35) on a rating-band Lichess set built with clock and wp_loss fields.

    uv run chess-rl train human --config configs/train/human.yaml
    uv run chess-rl train human --config configs/train/human.yaml --resume runs/human/latest.pt

Loss: cross-entropy(policy, move played) + value_weight * cross-entropy(WDL, result)
      + blunder_weight * BCE(blunder logit, wp_loss >= 10), the last only on positions with a label.

Validation reports move-match accuracy (top-1, the Maia metric) overall and per mover rating band,
and the blunder head's ROC AUC. Same safety rails as the M4 trainer: progress lines, periodic validation,
latest/best checkpoints, resume, a max_hours limit, and an abort after repeated non-finite losses.
"""

from __future__ import annotations

import faulthandler
import json
import random
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from chess_rl.encoding.compact import planes_from_compact
from chess_rl.model.human import BLUNDER_POINTS, HumanNet, HumanNetConfig, save_human
from chess_rl.model.network import count_parameters, to_device
from chess_rl.train.supervised import load_manifest, lr_at, shard_batches

FIELDS = ("bitboards", "flags", "halfmove", "move", "result", "elo_us", "elo_them", "clock", "wp_loss")
RATING_BANDS = ((0, 800), (800, 1000), (1000, 1200), (1200, 1400), (1400, 1600), (1600, 9999))


@dataclass(frozen=True)
class HumanTrainConfig:
    data_dir: str = "data/lichess/human-600-1600-2026-10"
    out_dir: str = "runs/human"
    init_from: str = ""  # optional ChessNet checkpoint: copy every tensor whose name and shape match
    blocks: int = 10
    filters: int = 128
    se: bool = True
    cond_dim: int = 64
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    epochs: float = 2.0
    value_weight: float = 0.5
    blunder_weight: float = 1.0
    grad_clip: float = 1.0
    log_every: int = 200
    val_every: int = 5000
    val_positions: int = 200_000
    ckpt_every: int = 2000
    max_hours: float = 6.0
    max_nonfinite: int = 20
    stall_seconds: float = 900.0  # watchdog: no finished step for this long -> dump all stacks and exit
    seed: int = 0
    device: str = "auto"

    def net_config(self) -> HumanNetConfig:
        return HumanNetConfig(blocks=self.blocks, filters=self.filters, se=self.se, cond_dim=self.cond_dim)


def to_tensors(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    nb = device.type == "cuda"

    def t(a: np.ndarray, dtype=None) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(a))
        return x.to(device, dtype=dtype, non_blocking=nb)

    planes = planes_from_compact(t(batch["bitboards"].view(np.int64)), t(batch["flags"]), t(batch["halfmove"]))
    if device.type == "cuda":
        planes = planes.contiguous(memory_format=torch.channels_last)
    wp_loss = t(batch["wp_loss"], torch.int16)
    return {
        "planes": planes,
        "move": t(batch["move"], torch.int64),
        "wdl": t(batch["result"], torch.int64) + 1,
        "elo_us": t(batch["elo_us"], torch.int64),
        "elo_them": t(batch["elo_them"], torch.int64),
        "clock": t(batch["clock"], torch.int64),
        "blunder": (wp_loss >= BLUNDER_POINTS).float(),
        "labelled": wp_loss >= 0,
    }


def losses(model: HumanNet, b: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    policy, value, blunder = model(b["planes"], b["elo_us"], b["elo_them"], b["clock"])
    policy, value, blunder = policy.float(), value.float(), blunder.float()
    policy_loss = F.cross_entropy(policy, b["move"])
    value_loss = F.cross_entropy(value, b["wdl"])
    mask = b["labelled"]
    blunder_loss = (
        F.binary_cross_entropy_with_logits(blunder[mask], b["blunder"][mask]) if mask.any() else blunder.sum() * 0
    )
    return policy, blunder, policy_loss, value_loss, blunder_loss


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Probability that a random positive outranks a random negative (ties count half)."""
    pos, neg = int(labels.sum()), int(len(labels) - labels.sum())
    if not pos or not neg:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    s = scores[order]
    i = 0
    while i < len(s):  # average ranks over ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels.astype(bool)].sum() - pos * (pos + 1) / 2) / (pos * neg))


@torch.no_grad()
def evaluate(
    model: HumanNet, val_paths: list[Path], device: torch.device, max_positions: int, batch_size: int = 2048
) -> dict[str, float]:
    model.eval()
    n = 0
    sums = {"policy_loss": 0.0, "value_loss": 0.0, "top1": 0.0, "top5": 0.0}
    band_hits = np.zeros(len(RATING_BANDS))
    band_n = np.zeros(len(RATING_BANDS))
    scores, labels = [], []
    for _, batch in shard_batches(
        val_paths, batch_size, random.Random(1234), prefetch=1, drop_last=False, fields=FIELDS
    ):
        b = to_tensors(batch, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            policy, value, blunder = model(b["planes"], b["elo_us"], b["elo_them"], b["clock"])
        policy, value = policy.float(), value.float()
        k = len(b["move"])
        sums["policy_loss"] += F.cross_entropy(policy, b["move"], reduction="sum").item()
        sums["value_loss"] += F.cross_entropy(value, b["wdl"], reduction="sum").item()
        top5 = policy.topk(5, dim=1).indices
        hit = (top5[:, 0] == b["move"]).cpu().numpy()
        sums["top1"] += hit.sum()
        sums["top5"] += (top5 == b["move"][:, None]).any(dim=1).sum().item()
        elo = batch["elo_us"]
        for i, (lo, hi) in enumerate(RATING_BANDS):
            m = (elo >= lo) & (elo < hi)
            band_hits[i] += hit[m].sum()
            band_n[i] += m.sum()
        mask = b["labelled"].cpu().numpy()
        scores.append(blunder.float().cpu().numpy()[mask])
        labels.append(b["blunder"].cpu().numpy()[mask])
        n += k
        if n >= max_positions:
            break
    model.train()
    out = {k: v / max(n, 1) for k, v in sums.items()}
    sc, lb = (np.concatenate(scores), np.concatenate(labels)) if scores else (np.zeros(0), np.zeros(0))
    out.update(
        positions=n,
        blunder_auc=roc_auc(sc, lb),
        blunder_rate=float(lb.mean()) if len(lb) else float("nan"),
        blunder_labelled=int(len(lb)),
    )
    for (lo, hi), h, c in zip(RATING_BANDS, band_hits, band_n, strict=True):
        if c:
            out[f"top1_{lo}_{hi}"] = float(h / c)
    return out


def init_from_chessnet(model: HumanNet, path: str | Path) -> int:
    """Copy matching tensors (stem, tower convs/BN/SE, policy and value convs) from an M4 ChessNet."""
    src = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
    own = model.state_dict()
    copied = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
    own.update(copied)
    model.load_state_dict(own)
    return len(copied)


def _save(path: Path, model, optimizer, scaler, state: dict, conf: HumanTrainConfig) -> None:
    save_human(
        path,
        model,
        train_state=state,
        train_config=asdict(conf),
        optimizer=optimizer.state_dict(),
        scaler=scaler.state_dict(),
    )


def train(
    conf: HumanTrainConfig,
    resume: str | Path | None = None,
    log: Callable[[str], None] = lambda s: print(s, file=sys.stderr, flush=True),
) -> dict:
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if conf.device == "auto" else conf.device)
    torch.manual_seed(conf.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    out = Path(conf.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(conf.data_dir)
    missing = set(FIELDS) - set(manifest["fields"])
    if missing:
        raise ValueError(f"{conf.data_dir} lacks {sorted(missing)}: rebuild it with the M6 data pipeline")
    data_dir = Path(conf.data_dir)
    train_paths = [data_dir / s for s in manifest["train_shards"]]
    val_paths = [data_dir / s for s in manifest["val_shards"]]
    positions = manifest["stats"]["train_positions"]
    total_steps = max(1, int(positions // conf.batch_size * conf.epochs))

    model = HumanNet(conf.net_config())
    if conf.init_from and not resume:
        log(f"initialised {init_from_chessnet(model, conf.init_from)} tensors from {conf.init_from}")
    model = to_device(model, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    state = {"step": 0, "epoch": 0, "shard": 0, "best_val": float("inf"), "positions_seen": 0}
    if resume:
        data = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(data["state_dict"])
        optimizer.load_state_dict(data["meta"]["optimizer"])
        scaler.load_state_dict(data["meta"]["scaler"])
        state.update(data["meta"]["train_state"])
        log(f"resumed from {resume} at step {state['step']}")
    log(
        f"human model {conf.blocks}x{conf.filters}: {count_parameters(model) / 1e6:.2f}M params | "
        f"{positions:,} train positions, {total_steps:,} steps | {device}"
    )

    history: list[dict] = []
    t_start = time.monotonic()
    t_log, seen_at_log, nonfinite = t_start, state["positions_seen"], 0
    stop_reason = "completed"
    run = {"policy": 0.0, "value": 0.0, "blunder": 0.0, "top1": 0.0, "n": 0}
    model.train()
    faulthandler.dump_traceback_later(conf.stall_seconds, exit=True)
    while state["step"] < total_steps and stop_reason == "completed":
        rng = random.Random(conf.seed * 1000 + state["epoch"])
        order = list(range(len(train_paths)))
        rng.shuffle(order)
        for shard_i, batch in shard_batches(
            [train_paths[i] for i in order], conf.batch_size, rng, start_shard=state["shard"], fields=FIELDS
        ):
            if state["step"] >= total_steps:
                break
            state["shard"] = shard_i
            for group in optimizer.param_groups:
                group["lr"] = lr_at(state["step"], total_steps, conf.lr, conf.warmup_steps)
            b = to_tensors(batch, device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                policy, _, policy_loss, value_loss, blunder_loss = losses(model, b)
                loss = policy_loss + conf.value_weight * value_loss + conf.blunder_weight * blunder_loss
            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                nonfinite += 1
                if nonfinite >= conf.max_nonfinite:
                    stop_reason = "nonfinite_loss"
                    break
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), conf.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            state["step"] += 1
            faulthandler.dump_traceback_later(conf.stall_seconds, exit=True)  # re-arm the watchdog
            state["positions_seen"] += len(b["move"])
            run["policy"] += policy_loss.item()
            run["value"] += value_loss.item()
            run["blunder"] += blunder_loss.item()
            run["top1"] += (policy.argmax(1) == b["move"]).float().mean().item()
            run["n"] += 1

            step = state["step"]
            if step % conf.log_every == 0:
                now = time.monotonic()
                rate = (state["positions_seen"] - seen_at_log) / (now - t_log)
                eta = (total_steps - step) * conf.batch_size / max(rate, 1) / 3600
                k = run["n"]
                log(
                    f"step {step:,}/{total_steps:,} ep {state['epoch']} | policy {run['policy'] / k:.3f} "
                    f"value {run['value'] / k:.3f} blunder {run['blunder'] / k:.3f} top1 {run['top1'] / k:.3f} "
                    f"| lr {optimizer.param_groups[0]['lr']:.2e} | {rate:,.0f} pos/s | ETA {eta:.2f} h"
                )
                t_log, seen_at_log = now, state["positions_seen"]
                run = dict.fromkeys(run, 0.0) | {"n": 0}
            if step % conf.val_every == 0 or step == total_steps:
                metrics = evaluate(model, val_paths, device, conf.val_positions)
                metrics.update(step=step, hours=(time.monotonic() - t_start) / 3600)
                history.append(metrics)
                with (out / "history.jsonl").open("a") as f:
                    f.write(json.dumps(metrics) + "\n")
                tag = ""
                if metrics["policy_loss"] < state["best_val"]:
                    state["best_val"] = metrics["policy_loss"]
                    _save(out / "best.pt", model, optimizer, scaler, state, conf)
                    tag = " (new best)"
                bands = " ".join(f"{k[5:]}:{v:.3f}" for k, v in metrics.items() if k.startswith("top1_"))
                log(
                    f"[val] step {step:,} | policy {metrics['policy_loss']:.3f} top1 {metrics['top1']:.3f} "
                    f"top5 {metrics['top5']:.3f} | blunder AUC {metrics['blunder_auc']:.3f} "
                    f"(rate {metrics['blunder_rate']:.3f}) | {bands}{tag}"
                )
            if step % conf.ckpt_every == 0:
                _save(out / "latest.pt", model, optimizer, scaler, state, conf)
            if (time.monotonic() - t_start) / 3600 >= conf.max_hours:
                stop_reason = "max_hours"
                break
        else:
            state["epoch"] += 1
            state["shard"] = 0
            continue
        if stop_reason != "completed":
            break
    faulthandler.cancel_dump_traceback_later()
    _save(out / "latest.pt", model, optimizer, scaler, state, conf)
    summary = {
        "stop_reason": stop_reason,
        "steps": state["step"],
        "total_steps": total_steps,
        "positions_seen": state["positions_seen"],
        "hours": (time.monotonic() - t_start) / 3600,
        "best_val_policy_loss": state["best_val"],
        "final_val": history[-1] if history else None,
        "params": count_parameters(model),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(f"done: {stop_reason} after {state['step']:,} steps, {summary['hours']:.2f} h")
    return summary
