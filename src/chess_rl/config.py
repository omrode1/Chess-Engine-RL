"""Typed run configuration: dataclasses loaded from YAML, with command-line overrides.

Every experiment is described by a frozen dataclass. A YAML file provides the values, and
`--set key=value` style overrides (or explicit CLI flags) replace individual fields.
Unknown keys are an error, so a typo in a config file cannot silently fall back to a default.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from pathlib import Path
from typing import Any

import yaml


@dataclasses.dataclass(frozen=True)
class BaselineTrainConfig:
    """v0 DQN curriculum training (baseline/train_tactical_agent.py)."""

    timesteps: int = 3_000_000
    num_envs: int = 8
    model_dir: str = "runs/baseline/models"
    log_dir: str = "runs/baseline/logs"
    use_opening_book: bool = True
    use_curriculum: bool = True
    seed: int | None = 0


@dataclasses.dataclass(frozen=True)
class BaselineEvalConfig:
    """v0 model vs uniformly random player (baseline/eval_vs_random.py)."""

    model: str = "runs/baseline/models/chess_tactical_final"
    games: int = 100
    max_plies: int = 300
    seed: int = 0
    pgn_dir: str | None = None
    json_out: str | None = None


@dataclasses.dataclass(frozen=True)
class DataBuildConfig:
    """Lichess shard build (chess_rl.data.lichess). One directory per dataset under out_root/name."""

    name: str = "strength-1800"
    month: str = "2026-10"
    min_elo: int = 1800
    max_elo: int = 4000
    time_classes: str = "blitz,rapid,classical"
    target_positions: int = 30_000_000
    shard_size: int = 1_000_000
    val_every: int = 50
    workers: int = 12
    out_root: str = "data/lichess"
    max_gb: float = 12.0  # hard cap on compressed bytes streamed
    stall_timeout: float = 120.0


def to_dict(config: Any) -> dict[str, Any]:
    return dataclasses.asdict(config)


def _coerce(value: Any, annotation: Any, name: str) -> Any:
    """Convert YAML/CLI values to the dataclass field type, rejecting mismatches."""
    hint = annotation
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        if value is None or (isinstance(value, str) and value.lower() in {"none", "null"}):
            return None
        hint = args[0]
    if hint is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "1", "yes", "on", "false", "0", "no", "off"}:
            return value.lower() in {"true", "1", "yes", "on"}
        raise ValueError(f"{name}: expected bool, got {value!r}")
    if hint is int:
        if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
            raise ValueError(f"{name}: expected int, got {value!r}")
        return int(str(value).replace("_", ""))
    if hint is float:
        return float(value)
    if hint is str:
        return str(value)
    return value


def build[T](cls: type[T], values: dict[str, Any]) -> T:
    hints = typing.get_type_hints(cls)
    fields = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - fields
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**{k: _coerce(v, hints[k], k) for k, v in values.items()})


def load[T](cls: type[T], path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> T:
    """Defaults <- YAML file <- overrides (None-valued overrides are ignored)."""
    values: dict[str, Any] = {}
    if path is not None:
        data = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path}: top level must be a mapping")
        values.update(data)
    values.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return build(cls, values)


def parse_set_args(pairs: list[str]) -> dict[str, str]:
    """Parse ['a=1', 'b=x'] from repeated --set flags."""
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"--set expects key=value, got {pair!r}")
        out[key.strip()] = value.strip()
    return out
