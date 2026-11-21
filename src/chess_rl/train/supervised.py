"""Supervised pretraining on Lichess shards (M4): predict the move played (policy) and the result (value).

    uv run chess-rl train sl --config configs/train/m4.yaml
    uv run chess-rl train sl --config configs/train/m4.yaml --resume runs/m4/latest.pt

Data: compact shards (chess_rl.data.lichess) are loaded one at a time by a background thread, shuffled
within the shard, and expanded to planes on the GPU (chess_rl.encoding.compact.planes_from_compact).

Loss: cross-entropy(policy logits, move played) + value_weight * cross-entropy(WDL logits, result).
Optimisation: AdamW, linear warmup then cosine decay, gradient clipping, fp16 autocast with GradScaler,
channels_last on CUDA.

Safety: progress lines every `log_every` steps, validation every `val_every` steps (top-1/top-5 move
accuracy, value loss, value calibration), checkpoints (latest + best by validation loss), resume, a hard
`max_hours` limit, and an abort (with checkpoint) after repeated non-finite losses.
"""

from __future__ import annotations

import faulthandler
import json
import math
import queue
import random
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from chess_rl.encoding.compact import planes_from_compact
from chess_rl.model.network import ChessNet, NetConfig, count_parameters, save_checkpoint, to_device

BATCH_FIELDS = ("bitboards", "flags", "halfmove", "move", "result")


@dataclass(frozen=True)
class TrainConfig:
    data_dir: str = "data/lichess/strength-1800-2026-10"
    out_dir: str = "runs/m4"
    blocks: int = 10
    filters: int = 128
    se: bool = True
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    epochs: float = 2.0
    value_weight: float = 1.0
    grad_clip: float = 1.0
    log_every: int = 200
    val_every: int = 5000
    val_positions: int = 200_000
    ckpt_every: int = 5000
    max_hours: float = 10.0
    max_nonfinite: int = 20
    stall_seconds: float = 900.0  # watchdog: no finished step for this long -> dump all stacks and exit
    seed: int = 0
    device: str = "auto"

    def net_config(self) -> NetConfig:
        return NetConfig(blocks=self.blocks, filters=self.filters, se=self.se)


# --------------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------------


def load_manifest(data_dir: str | Path) -> dict:
    return json.loads((Path(data_dir) / "manifest.json").read_text())


def load_shard(path: Path, fields: tuple[str, ...] = BATCH_FIELDS) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in fields}


def shard_batches(
    shard_paths: list[Path],
    batch_size: int,
    rng: random.Random,
    start_shard: int = 0,
    prefetch: int = 2,
    drop_last: bool = True,
    fields: tuple[str, ...] = BATCH_FIELDS,
) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
    """Yield (shard_index, batch) over shards in the given order, shuffling within each shard.

    A background thread loads the next shards while the current one is being trained on.
    """
    q: queue.Queue = queue.Queue(maxsize=prefetch)
    stop = threading.Event()

    def loader() -> None:
        try:
            for i in range(start_shard, len(shard_paths)):
                if stop.is_set():
                    return
                q.put((i, load_shard(shard_paths[i], fields)))
        except BaseException as e:  # surface loader errors in the training thread
            q.put(e)
            return
        q.put(None)

    t = threading.Thread(target=loader, daemon=True, name="shard-loader")
    t.start()
    try:
        while True:
            item = q.get(timeout=600)  # a stuck disk read must not hang training silently
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            i, shard = item
            n = len(shard["move"])
            perm = np.random.default_rng(rng.getrandbits(64)).permutation(n)
            stop_at = n - batch_size + 1 if drop_last else n
            for s in range(0, max(stop_at, 0), batch_size):
                idx = perm[s : s + batch_size]
                yield i, {k: v[idx] for k, v in shard.items()}
    finally:
        stop.set()


def to_tensors(batch: dict[str, np.ndarray], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compact batch -> (planes, move targets, WDL targets) on device."""
    nb = device.type == "cuda"
    bb = torch.from_numpy(np.ascontiguousarray(batch["bitboards"]).view(np.int64)).to(device, non_blocking=nb)
    flags = torch.from_numpy(batch["flags"]).to(device, non_blocking=nb)
    half = torch.from_numpy(batch["halfmove"]).to(device, non_blocking=nb)
    planes = planes_from_compact(bb, flags, half)
    if device.type == "cuda":
        planes = planes.contiguous(memory_format=torch.channels_last)
    move = torch.from_numpy(batch["move"].astype(np.int64)).to(device, non_blocking=nb)
    wdl = torch.from_numpy(batch["result"].astype(np.int64) + 1).to(device, non_blocking=nb)  # 0 L, 1 D, 2 W
    return planes, move, wdl


# --------------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------------


def lr_at(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(
    model: ChessNet, val_paths: list[Path], device: torch.device, max_positions: int, batch_size: int = 2048
) -> dict[str, float]:
    """Policy loss/top-1/top-5, value loss, and value calibration on up to max_positions validation rows."""
    model.eval()
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "top1": 0.0, "top5": 0.0, "n": 0}
    pred_score, true_score = [], []
    rng = random.Random(1234)  # same validation positions every time
    for _, batch in shard_batches(val_paths, batch_size, rng, prefetch=1, drop_last=False):
        planes, move, wdl = to_tensors(batch, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            policy, value = model(planes)
        policy, value = policy.float(), value.float()
        n = len(move)
        totals["policy_loss"] += F.cross_entropy(policy, move, reduction="sum").item()
        totals["value_loss"] += F.cross_entropy(value, wdl, reduction="sum").item()
        top5 = policy.topk(5, dim=1).indices
        totals["top1"] += (top5[:, 0] == move).sum().item()
        totals["top5"] += (top5 == move[:, None]).any(dim=1).sum().item()
        p = torch.softmax(value, dim=1)
        pred_score.append((p[:, 2] + 0.5 * p[:, 1]).cpu())
        true_score.append((wdl.float() / 2).cpu())
        totals["n"] += n
        if totals["n"] >= max_positions:
            break
    model.train()
    if not totals["n"]:
        nan = float("nan")
        return dict.fromkeys(("policy_loss", "value_loss", "top1", "top5", "value_calibration_error"), nan) | {
            "positions": 0
        }
    n = totals["n"]
    pred, true = torch.cat(pred_score), torch.cat(true_score)
    # calibration: mean |predicted - actual score| over 10 bins of predicted score
    bins = torch.clamp((pred * 10).long(), 0, 9)
    calib = [
        abs(pred[bins == b].mean().item() - true[bins == b].mean().item()) for b in range(10) if (bins == b).sum() > 100
    ]
    return {
        "policy_loss": totals["policy_loss"] / n,
        "value_loss": totals["value_loss"] / n,
        "top1": totals["top1"] / n,
        "top5": totals["top5"] / n,
        "value_calibration_error": float(np.mean(calib)) if calib else float("nan"),
        "positions": totals["n"],
    }


# --------------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------------


def _save(path: Path, model, optimizer, scaler, state: dict, conf: TrainConfig) -> None:
    save_checkpoint(
        path,
        model,
        train_state=state,
        train_config=asdict(conf),
        optimizer=optimizer.state_dict(),
        scaler=scaler.state_dict(),
    )


def train(
    conf: TrainConfig,
    resume: str | Path | None = None,
    log: Callable[[str], None] = lambda s: print(s, file=sys.stderr, flush=True),
) -> dict:
    """Run supervised training; returns a summary (also written to out_dir/summary.json)."""
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if conf.device == "auto" else conf.device)
    torch.manual_seed(conf.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    out = Path(conf.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(conf.data_dir)
    data_dir = Path(conf.data_dir)
    train_paths = [data_dir / s for s in manifest["train_shards"]]
    val_paths = [data_dir / s for s in manifest["val_shards"]]
    positions = manifest["stats"]["train_positions"]
    steps_per_epoch = positions // conf.batch_size
    total_steps = max(1, int(steps_per_epoch * conf.epochs))

    model = to_device(ChessNet(conf.net_config()), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    state = {"step": 0, "epoch": 0, "shard": 0, "best_val": float("inf"), "positions_seen": 0}
    if resume:
        data = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(data["state_dict"])
        optimizer.load_state_dict(data["meta"]["optimizer"])
        scaler.load_state_dict(data["meta"]["scaler"])
        state.update(data["meta"]["train_state"])
        log(f"resumed from {resume} at step {state['step']} (epoch {state['epoch']}, shard {state['shard']})")
    log(
        f"model {conf.blocks}x{conf.filters} se={conf.se}: {count_parameters(model) / 1e6:.2f}M params | "
        f"{positions:,} train positions, {steps_per_epoch:,} steps/epoch, {total_steps:,} steps total | {device}"
    )

    history: list[dict] = []
    t_start = time.monotonic()
    t_log, seen_at_log, nonfinite = t_start, state["positions_seen"], 0
    stop_reason = "completed"
    run_loss = {"policy": 0.0, "value": 0.0, "top1": 0.0, "n": 0}
    model.train()
    faulthandler.dump_traceback_later(conf.stall_seconds, exit=True)
    while state["step"] < total_steps and stop_reason == "completed":
        rng = random.Random(conf.seed * 1000 + state["epoch"])
        order = list(range(len(train_paths)))
        rng.shuffle(order)
        shard_order = [train_paths[i] for i in order]
        for shard_i, batch in shard_batches(shard_order, conf.batch_size, rng, start_shard=state["shard"]):
            if state["step"] >= total_steps:
                break
            state["shard"] = shard_i
            for group in optimizer.param_groups:
                group["lr"] = lr_at(state["step"], total_steps, conf.lr, conf.warmup_steps)
            planes, move, wdl = to_tensors(batch, device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                policy, value = model(planes)
                policy_loss = F.cross_entropy(policy.float(), move)
                value_loss = F.cross_entropy(value.float(), wdl)
                loss = policy_loss + conf.value_weight * value_loss
            if not torch.isfinite(loss):
                nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                if nonfinite >= conf.max_nonfinite:
                    stop_reason = "nonfinite_loss"
                    break
                continue
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), conf.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            state["step"] += 1
            faulthandler.dump_traceback_later(conf.stall_seconds, exit=True)  # re-arm the watchdog
            state["positions_seen"] += len(move)
            run_loss["policy"] += policy_loss.item()
            run_loss["value"] += value_loss.item()
            run_loss["top1"] += (policy.argmax(1) == move).float().mean().item()
            run_loss["n"] += 1

            step = state["step"]
            if step % conf.log_every == 0:
                now = time.monotonic()
                rate = (state["positions_seen"] - seen_at_log) / (now - t_log)
                eta = (total_steps - step) * conf.batch_size / max(rate, 1) / 3600
                n = run_loss["n"]
                log(
                    f"step {step:,}/{total_steps:,} ep {state['epoch']} "
                    f"| policy {run_loss['policy'] / n:.3f} value {run_loss['value'] / n:.3f} "
                    f"top1 {run_loss['top1'] / n:.3f} | lr {optimizer.param_groups[0]['lr']:.2e} "
                    f"| {rate:,.0f} pos/s | ETA {eta:.2f} h"
                )
                t_log, seen_at_log = now, state["positions_seen"]
                run_loss = {"policy": 0.0, "value": 0.0, "top1": 0.0, "n": 0}
            if step % conf.val_every == 0 or step == total_steps:
                metrics = evaluate(model, val_paths, device, conf.val_positions)
                metrics.update(step=step, hours=(time.monotonic() - t_start) / 3600)
                history.append(metrics)
                with (out / "history.jsonl").open("a") as f:
                    f.write(json.dumps(metrics) + "\n")
                best = metrics["policy_loss"] + conf.value_weight * metrics["value_loss"]
                tag = ""
                if best < state["best_val"]:
                    state["best_val"] = best
                    _save(out / "best.pt", model, optimizer, scaler, state, conf)
                    tag = " (new best)"
                log(
                    f"[val] step {step:,} | policy {metrics['policy_loss']:.3f} top1 {metrics['top1']:.3f} "
                    f"top5 {metrics['top5']:.3f} | value {metrics['value_loss']:.3f} "
                    f"calib {metrics['value_calibration_error']:.3f}{tag}"
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
        "best_val_loss": state["best_val"],
        "final_val": history[-1] if history else None,
        "params": count_parameters(model),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(f"done: {stop_reason} after {state['step']:,} steps, {summary['hours']:.2f} h")
    return summary
