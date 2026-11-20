import dataclasses
import json

import numpy as np
import pytest
import torch
import zstandard

from chess_rl.cli import main
from chess_rl.data.lichess import GameFilter, build
from chess_rl.model.human import HumanNet, HumanNetConfig, clock_bucket, clock_bucket_np, elo_bucket, load_human
from chess_rl.model.network import ChessNet, NetConfig, save_checkpoint
from chess_rl.train.human import HumanTrainConfig, init_from_chessnet, roc_auc, train
from tests.test_lichess_data import lichess_pgn

TINY = HumanNetConfig(blocks=1, filters=16, se=False, cond_dim=8)


def inputs(n: int, elo: int = 1200, clock: int = 100):
    planes = torch.randn(n, 21, 8, 8)
    full = torch.full((n,), elo)
    return planes, full, full.clone(), torch.full((n,), clock)


def test_shapes_and_conditioning_starts_as_identity():
    torch.manual_seed(0)
    model = HumanNet(TINY).eval()
    planes, elo, them, clock = inputs(4)
    policy, value, blunder = model(planes, elo, them, clock)
    assert policy.shape == (4, 4672) and value.shape == (4, 3) and blunder.shape == (4,)
    # FiLM starts at zero, so the policy cannot depend on the ratings before training
    other, _, _ = model(planes, torch.full((4,), 2500), them, torch.full((4,), 5))
    assert torch.allclose(policy, other, atol=1e-6)


def test_buckets():
    assert elo_bucket(torch.tensor([0, 400, 1250, 5000])).tolist() == [0, 0, 8, 25]
    secs = np.array([-1, 0, 9, 10, 59, 60, 5000])
    assert clock_bucket(torch.from_numpy(secs)).tolist() == clock_bucket_np(secs).tolist() == [0, 1, 1, 2, 4, 5, 10]


def test_add_players_keeps_population_behaviour():
    torch.manual_seed(0)
    model = HumanNet(TINY).eval()
    planes, elo, them, clock = inputs(2)
    before = model(planes, elo, them, clock, torch.zeros(2, dtype=torch.long))[0]
    first = model.add_players(1)
    assert first == 1 and model.config.players == 2
    after = model(planes, elo, them, clock, torch.ones(2, dtype=torch.long))[0]
    assert torch.allclose(before, after, atol=1e-6)  # a new player starts as the population


def test_roc_auc():
    assert roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) == 0.0
    assert roc_auc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 1, 0, 1])) == 0.5
    assert np.isnan(roc_auc(np.array([0.1, 0.2]), np.array([0, 0])))


def test_init_from_chessnet(tmp_path):
    path = tmp_path / "net.pt"
    net = ChessNet(NetConfig(blocks=1, filters=16, se=False))
    save_checkpoint(path, net)
    model = HumanNet(TINY)
    assert init_from_chessnet(model, path) > 5
    assert torch.equal(model.tower[0].conv1.weight, net.tower[0].conv1.weight)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("human")
    src = root / "db.pgn.zst"
    games = "".join(lichess_pgn(i, 1000 + 5 * i, 1100) for i in range(120))
    src.write_bytes(zstandard.ZstdCompressor().compress(games.encode()))
    build(
        src.as_uri(),
        root / "data",
        GameFilter(min_elo=600, max_elo=1800),
        10**9,
        shard_size=1500,
        val_every=6,
        workers=2,
    )
    # random games carry no [%eval]: give half the positions synthetic blunder labels
    rng = np.random.default_rng(0)
    for path in (root / "data").glob("*.npz"):
        with np.load(path) as z:
            arrays = dict(z)
        n = len(arrays["move"])
        arrays["wp_loss"] = np.where(rng.random(n) < 0.5, -1, rng.integers(0, 30, n)).astype(np.int8)
        np.savez(path, **arrays)
    return root / "data"


def conf(dataset, out, **kw):
    base = HumanTrainConfig(
        data_dir=str(dataset),
        out_dir=str(out),
        blocks=1,
        filters=16,
        se=False,
        cond_dim=8,
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


def test_train_end_to_end_and_resume(dataset, tmp_path):
    summary = train(conf(dataset, tmp_path / "a"), log=lambda s: None)
    assert summary["stop_reason"] == "completed" and summary["steps"] == summary["total_steps"]
    final = summary["final_val"]
    assert 0 <= final["top1"] <= 1 and 0 <= final["blunder_auc"] <= 1 and final["blunder_labelled"] > 0
    assert any(k.startswith("top1_1000_1200") for k in final)
    model, meta = load_human(tmp_path / "a" / "best.pt")
    assert meta["train_state"]["step"] > 0 and isinstance(model, HumanNet)

    # resuming a finished run does no more steps
    again = train(conf(dataset, tmp_path / "a"), resume=tmp_path / "a" / "latest.pt", log=lambda s: None)
    assert again["steps"] == summary["steps"]


def test_rejects_old_datasets(tmp_path, dataset):
    old = tmp_path / "old"
    old.mkdir()
    manifest = json.loads((dataset / "manifest.json").read_text())
    for k in ("clock", "wp_loss"):
        manifest["fields"].pop(k)
    (old / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="clock"):
        train(conf(old, tmp_path / "o"), log=lambda s: None)


def test_cli_train_human(dataset, tmp_path, monkeypatch):
    monkeypatch.setenv("CHESS_RL_RUNS_FILE", str(tmp_path / "runs.jsonl"))
    args = [
        f"data_dir={dataset}",
        f"out_dir={tmp_path / 'cli'}",
        "blocks=1",
        "filters=16",
        "se=false",
        "cond_dim=8",
        "batch_size=64",
        "epochs=0.2",
        "val_every=5",
        "val_positions=200",
        "device=cpu",
    ]
    assert main(["train", "human", *[x for a in args for x in ("--set", a)]]) == 0
    assert json.loads((tmp_path / "runs.jsonl").read_text())["kind"] == "train-human"
