"""Build Players from short spec strings, so matches can run players in worker processes.

    random                          uniformly random legal moves
    greedy                          MaterialGreedyPlayer
    stockfish:elo=1500,movetime=0.1 StockfishPlayer (keys: elo, skill, movetime, nodes, threads, hash_mb)
    v0:path=runs/v0.1.0/models/chess_tactical_final   the frozen v0 DQN baseline

New engines register themselves here as they are built (M4: policy network, M5: MCTS, ...).
"""

from __future__ import annotations

from collections.abc import Callable

from chess_rl.eval.players import MaterialGreedyPlayer, Player, RandomPlayer, StockfishPlayer


def parse_spec(spec: str) -> tuple[str, dict[str, str]]:
    kind, _, rest = spec.partition(":")
    params: dict[str, str] = {}
    for item in filter(None, (s.strip() for s in rest.split(","))):
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"bad player spec {spec!r}: expected key=value, got {item!r}")
        params[key.strip()] = value.strip()
    return kind.strip(), params


def _num(value: str) -> int | float:
    f = float(value)
    return int(f) if f.is_integer() and "." not in value else f


def _stockfish(params: dict[str, str], seed: int | None, name: str | None) -> Player:
    allowed = {"elo", "skill", "movetime", "nodes", "threads", "hash_mb"}
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"unknown stockfish options {sorted(unknown)}; allowed: {sorted(allowed)}")
    kwargs = {k: _num(v) for k, v in params.items()}
    if "movetime" in kwargs:
        kwargs["movetime"] = float(kwargs["movetime"])
    return StockfishPlayer(name=name, **kwargs)  # type: ignore[arg-type]


def _v0(params: dict[str, str], seed: int | None, name: str | None) -> Player:
    from baseline.player import V0Player

    if "path" not in params:
        raise ValueError("v0 player needs path=<model zip>")
    return V0Player(params["path"], name=name or "v0")


_BUILDERS: dict[str, Callable[[dict[str, str], int | None, str | None], Player]] = {
    "random": lambda p, seed, name: RandomPlayer(seed=seed, name=name or "random"),
    "greedy": lambda p, seed, name: MaterialGreedyPlayer(seed=seed, name=name or "greedy"),
    "stockfish": _stockfish,
    "v0": _v0,
}


def make_player(spec: str, seed: int | None = None, name: str | None = None) -> Player:
    kind, params = parse_spec(spec)
    if kind not in _BUILDERS:
        raise ValueError(f"unknown player kind {kind!r}; known: {sorted(_BUILDERS)}")
    if kind in {"random", "greedy"} and params:
        raise ValueError(f"{kind} takes no options")
    return _BUILDERS[kind](params, seed, name)


def display_name(spec: str) -> str:
    """Stable, readable name for result tables: the spec itself, minus file paths' directories."""
    kind, params = parse_spec(spec)
    if not params:
        return kind
    shown = {k: (v.rsplit("/", 1)[-1] if k == "path" else v) for k, v in params.items()}
    return kind + ":" + ",".join(f"{k}={v}" for k, v in shown.items())
