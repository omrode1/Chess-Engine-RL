from pathlib import Path

import pytest

from chess_rl import config as cfg

REPO = Path(__file__).resolve().parents[1]


def test_defaults():
    conf = cfg.load(cfg.BaselineTrainConfig)
    assert conf == cfg.BaselineTrainConfig()


def test_yaml_then_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("timesteps: 1_000\nuse_opening_book: false\nseed: 7\n")
    conf = cfg.load(cfg.BaselineTrainConfig, path, {"seed": "3", "num_envs": None})
    assert conf.timesteps == 1000 and conf.use_opening_book is False
    assert conf.seed == 3  # override wins; None-valued overrides are ignored
    assert conf.num_envs == cfg.BaselineTrainConfig().num_envs


def test_unknown_key_is_an_error(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("timestep: 10\n")  # typo
    with pytest.raises(ValueError, match="unknown BaselineTrainConfig keys"):
        cfg.load(cfg.BaselineTrainConfig, path)


@pytest.mark.parametrize(("raw", "expected"), [("true", True), ("off", False), ("1", True), (False, False)])
def test_bool_coercion(raw, expected):
    assert cfg.build(cfg.BaselineTrainConfig, {"use_curriculum": raw}).use_curriculum is expected


@pytest.mark.parametrize("bad", [{"use_curriculum": "maybe"}, {"timesteps": 1.5}, {"timesteps": True}])
def test_bad_values_are_rejected(bad):
    with pytest.raises(ValueError):
        cfg.build(cfg.BaselineTrainConfig, bad)


def test_optional_fields_accept_none():
    assert cfg.build(cfg.BaselineTrainConfig, {"seed": "none"}).seed is None
    assert cfg.build(cfg.BaselineEvalConfig, {"pgn_dir": None}).pgn_dir is None


def test_parse_set_args():
    assert cfg.parse_set_args(["a=1", " b = x=y "]) == {"a": "1", "b": "x=y"}
    with pytest.raises(ValueError):
        cfg.parse_set_args(["novalue"])


def test_shipped_baseline_config_matches_v0_1_0():
    conf = cfg.load(cfg.BaselineTrainConfig, REPO / "configs" / "baseline.yaml")
    assert conf.timesteps == 300_000 and conf.num_envs == 8 and conf.use_curriculum
