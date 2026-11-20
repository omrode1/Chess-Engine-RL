import dataclasses
from pathlib import Path

import pytest

from chess_rl import config as cfg

REPO = Path(__file__).resolve().parents[1]


@dataclasses.dataclass(frozen=True)
class Example:
    timesteps: int = 3_000_000
    num_envs: int = 8
    use_opening_book: bool = True
    use_curriculum: bool = True
    seed: int | None = 0
    pgn_dir: str | None = None


def test_defaults():
    conf = cfg.load(Example)
    assert conf == Example()


def test_yaml_then_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("timesteps: 1_000\nuse_opening_book: false\nseed: 7\n")
    conf = cfg.load(Example, path, {"seed": "3", "num_envs": None})
    assert conf.timesteps == 1000 and conf.use_opening_book is False
    assert conf.seed == 3  # override wins; None-valued overrides are ignored
    assert conf.num_envs == Example().num_envs


def test_unknown_key_is_an_error(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("timestep: 10\n")  # typo
    with pytest.raises(ValueError, match="unknown Example keys"):
        cfg.load(Example, path)


@pytest.mark.parametrize(("raw", "expected"), [("true", True), ("off", False), ("1", True), (False, False)])
def test_bool_coercion(raw, expected):
    assert cfg.build(Example, {"use_curriculum": raw}).use_curriculum is expected


@pytest.mark.parametrize("bad", [{"use_curriculum": "maybe"}, {"timesteps": 1.5}, {"timesteps": True}])
def test_bad_values_are_rejected(bad):
    with pytest.raises(ValueError):
        cfg.build(Example, bad)


def test_optional_fields_accept_none():
    assert cfg.build(Example, {"seed": "none"}).seed is None
    assert cfg.build(Example, {"pgn_dir": None}).pgn_dir is None


def test_parse_set_args():
    assert cfg.parse_set_args(["a=1", " b = x=y "]) == {"a": "1", "b": "x=y"}
    with pytest.raises(ValueError):
        cfg.parse_set_args(["novalue"])


@pytest.mark.parametrize(
    ("conf_class", "path"),
    [
        (cfg.DataBuildConfig, "configs/data/strength.yaml"),
        (cfg.DataBuildConfig, "configs/data/human-band.yaml"),
        ("train", "configs/train/m4.yaml"),
    ],
)
def test_shipped_configs_load(conf_class, path):
    if conf_class == "train":
        from chess_rl.train.supervised import TrainConfig

        conf_class = TrainConfig
    assert cfg.load(conf_class, REPO / path)
