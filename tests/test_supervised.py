import dataclasses
import json
import math

import pytest
import torch
import zstandard

from chess_rl.data.lichess import GameFilter, build
from chess_rl.model.network import load_checkpoint
from chess_rl.train.supervised import TrainConfig, evaluate, lr_at, shard_batches, train
from tests.test_lichess_data import lichess_pgn


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("ds")
    src = root / "db.pgn.zst"
    src.write_bytes(
        zstandard.ZstdCompressor().compress("".join(lichess_pgn(i, 2000, 2000) for i in range(150)).encode())
    )
    build(src.as_uri(), root / "data", GameFilter(), target_positions=10**9, shard_size=1500, val_every=8, workers=2)
    return root / "data"


def conf(dataset, out, **kw):
    base = TrainConfig(
        data_dir=str(dataset),
        out_dir=str(out),
        blocks=1,
        filters=16,
        se=False,
        batch_size=64,
        epochs=1.0,
        warmup_steps=5,
        log_every=10,
        val_every=20,
        val_positions=500,
        ckpt_every=20,
        device="cpu",
    )
    return dataclasses.replace(base, **kw)


def test_lr_schedule():
    assert lr_at(0, 100, 1.0, 10) == pytest.approx(0.1)
    assert lr_at(9, 100, 1.0, 10) == pytest.approx(1.0)
    assert lr_at(100, 100, 1.0, 10) == pytest.approx(0.0, abs=1e-9)
    assert 0 < lr_at(55, 100, 1.0, 10) < 1


def test_shard_batches_cover_each_shard_once(dataset):
    import random

    manifest = json.loads((dataset / "manifest.json").read_text())
    paths = [dataset / s for s in manifest["train_shards"]]
    seen = [b["move"].shape[0] for _, b in shard_batches(paths, 100, random.Random(0))]
    assert all(n == 100 for n in seen) and sum(seen) > 0.9 * manifest["stats"]["train_positions"] - 100 * len(paths)


@pytest.mark.timeout(300)
def test_training_reduces_loss_and_writes_checkpoints(dataset, tmp_path):
    out = tmp_path / "run"
    logs = []
    summary = train(conf(dataset, out, epochs=3.0, lr=3e-3), log=logs.append)
    assert summary["stop_reason"] == "completed" and summary["steps"] == summary["total_steps"]
    history = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
    assert history[-1]["policy_loss"] < math.log(4672)  # better than uniform over all indices
    assert 0 <= history[-1]["top1"] <= 1 and history[-1]["top5"] >= history[-1]["top1"]
    assert (out / "best.pt").exists() and (out / "latest.pt").exists()
    model, meta = load_checkpoint(out / "best.pt")
    assert meta["train_state"]["step"] > 0 and model.config.filters == 16
    assert any(line.startswith("[val]") for line in logs) and any("pos/s" in line for line in logs)


@pytest.mark.timeout(300)
def test_resume_continues_from_checkpoint(dataset, tmp_path):
    out = tmp_path / "run"
    first = train(conf(dataset, out, epochs=1.0), log=lambda s: None)
    logs = []
    second = train(conf(dataset, out, epochs=2.0), resume=out / "latest.pt", log=logs.append)
    assert second["steps"] == second["total_steps"] > first["steps"]
    assert any("resumed from" in line for line in logs)


@pytest.mark.timeout(120)
def test_max_hours_stops_early(dataset, tmp_path):
    summary = train(conf(dataset, tmp_path / "run", epochs=50.0, max_hours=1e-9), log=lambda s: None)
    assert summary["stop_reason"] == "max_hours" and summary["steps"] < summary["total_steps"]


@pytest.mark.timeout(120)
def test_nonfinite_loss_aborts(dataset, tmp_path, monkeypatch):
    import chess_rl.train.supervised as sup

    monkeypatch.setattr(sup.F, "cross_entropy", lambda *a, **k: torch.tensor(float("nan"), requires_grad=True))
    summary = train(conf(dataset, tmp_path / "run", max_nonfinite=3), log=lambda s: None)
    assert summary["stop_reason"] == "nonfinite_loss" and summary["steps"] == 0


def test_evaluate_metrics(dataset):
    from chess_rl.model.network import ChessNet, NetConfig

    manifest = json.loads((dataset / "manifest.json").read_text())
    m = evaluate(
        ChessNet(NetConfig(blocks=1, filters=8, se=False)),
        [dataset / s for s in manifest["val_shards"]],
        torch.device("cpu"),
        400,
        batch_size=100,
    )
    assert m["positions"] > 0 and m["top5"] >= m["top1"] and m["policy_loss"] > 0


@pytest.mark.timeout(300)
def test_cli_train_records_run(dataset, tmp_path):
    from chess_rl.cli import main
    from chess_rl.runlog import read_runs

    args = ["train", "sl", "--set", f"data_dir={dataset}", "--set", f"out_dir={tmp_path / 'run'}"]
    for kv in ("blocks=1", "filters=8", "se=false", "batch_size=64", "epochs=0.5", "val_every=10", "device=cpu"):
        args += ["--set", kv]
    assert main(args) == 0
    (rec,) = read_runs()
    assert rec["kind"] == "train-sl" and rec["outputs"]["stop_reason"] == "completed" and rec["config"]["filters"] == 8
