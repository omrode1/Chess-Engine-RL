"""`chess-rl` command line. Subcommands are added per milestone (data, train, match, uci, play, ...)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from chess_rl import config as cfg
from chess_rl.runlog import track_run
from chess_rl.system import git_info, hardware_info, pick_device, seed_everything, versions


def _cmd_info(args: argparse.Namespace) -> int:
    info = {"versions": versions(), "hardware": hardware_info(), "device": pick_device("auto"), "git": git_info()}
    print(json.dumps(info, indent=2))
    return 0


def _overrides(args: argparse.Namespace, fields: Sequence[str]) -> dict:
    out = cfg.parse_set_args(args.set or [])
    out.update({f: getattr(args, f) for f in fields if getattr(args, f, None) is not None})
    return out


def _cmd_baseline_train(args: argparse.Namespace) -> int:
    from baseline.train_tactical_agent import train_model

    conf = cfg.load(
        cfg.BaselineTrainConfig,
        args.config,
        _overrides(
            args, ["timesteps", "num_envs", "model_dir", "log_dir", "seed", "use_opening_book", "use_curriculum"]
        ),
    )
    device = pick_device("auto")
    seed_everything(conf.seed)
    with track_run("baseline-train", conf, seed=conf.seed, device=device) as run:
        train_model(
            total_timesteps=conf.timesteps,
            model_dir=conf.model_dir,
            log_dir=conf.log_dir,
            use_opening_book=conf.use_opening_book,
            num_envs=conf.num_envs,
            use_curriculum=conf.use_curriculum,
            seed=conf.seed,
        )
        run.outputs["final_model"] = f"{conf.model_dir}/chess_tactical_final.zip"
    return 0


def _cmd_baseline_eval_random(args: argparse.Namespace) -> int:
    from pathlib import Path

    from stable_baselines3 import DQN

    from baseline.eval_vs_random import run as play

    conf = cfg.load(
        cfg.BaselineEvalConfig,
        args.config,
        _overrides(args, ["model", "games", "max_plies", "seed", "pgn_dir", "json_out"]),
    )
    with track_run("baseline-eval-random", conf, seed=conf.seed, device="cpu") as run:
        model = DQN.load(conf.model, device="cpu")
        pgn_dir = Path(conf.pgn_dir) if conf.pgn_dir else None
        results = {
            "agent_vs_random": play(model, conf.games, conf.max_plies, conf.seed, pgn_dir, "agent").summary(),
            "random_vs_random": play(None, conf.games, conf.max_plies, conf.seed, None, "random").summary(),
        }
        run.outputs.update(results)
    print(json.dumps(results, indent=2))
    if conf.json_out:
        Path(conf.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(conf.json_out).write_text(json.dumps(results, indent=2) + "\n")
    return 0


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="YAML config file (see configs/)")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config field (repeatable)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chess-rl", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="show versions, hardware, device, and git state")
    p.set_defaults(func=_cmd_info)

    base = sub.add_parser("baseline", help="v0 DQN baseline (frozen; see docs/postmortem-v0.md)")
    bsub = base.add_subparsers(dest="baseline_command", required=True)

    p = bsub.add_parser("train", help="train the v0 curriculum DQN agent")
    _add_config_args(p)
    p.add_argument("--timesteps", type=int)
    p.add_argument("--num-envs", type=int)
    p.add_argument("--model-dir")
    p.add_argument("--log-dir")
    p.add_argument("--seed", type=int)
    p.add_argument("--no-opening-book", dest="use_opening_book", action="store_const", const=False)
    p.add_argument("--no-curriculum", dest="use_curriculum", action="store_const", const=False)
    p.set_defaults(func=_cmd_baseline_train)

    p = bsub.add_parser("eval-random", help="score a v0 model against a random-move player")
    _add_config_args(p)
    p.add_argument("--model")
    p.add_argument("--games", type=int)
    p.add_argument("--max-plies", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--pgn-dir")
    p.add_argument("--json-out")
    p.set_defaults(func=_cmd_baseline_eval_random)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
