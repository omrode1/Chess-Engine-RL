"""`chess-rl` command line. Subcommands are added per milestone (data, train, match, uci, play, ...)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from chess_rl import config as cfg
from chess_rl.runlog import track_run
from chess_rl.system import git_info, hardware_info, pick_device, versions


def _cmd_info(args: argparse.Namespace) -> int:
    from chess_rl.eval.stockfish import StockfishNotFound, engine_name, find_stockfish

    try:
        sf_path = find_stockfish()
        stockfish = {"path": str(sf_path), "name": engine_name(sf_path)}
    except StockfishNotFound as e:
        stockfish = {"path": None, "error": str(e)}
    info = {
        "versions": versions(),
        "hardware": hardware_info(),
        "device": pick_device("auto"),
        "git": git_info(),
        "stockfish": stockfish,
    }
    print(json.dumps(info, indent=2))
    return 0


def _overrides(args: argparse.Namespace, fields: Sequence[str]) -> dict:
    out = cfg.parse_set_args(args.set or [])
    out.update({f: getattr(args, f) for f in fields if getattr(args, f, None) is not None})
    return out


def _cmd_stockfish(args: argparse.Namespace) -> int:
    from chess_rl.eval import stockfish

    if args.stockfish_command == "install":
        path = stockfish.install(force=args.force)
        print(f"{stockfish.engine_name(path)} installed at {path}")
    else:
        path = stockfish.find_stockfish()
        print(path)
    return 0


def _cmd_model(args: argparse.Namespace) -> int:
    from chess_rl.model.network import export_release

    meta = export_release(args.src, args.dst)
    print(f"wrote {args.dst} ({meta})")
    return 0


def _cmd_match(args: argparse.Namespace) -> int:
    from datetime import UTC, datetime

    from chess_rl.eval.match import append_elo_row, run_match

    scores: list[float] = []

    def progress(record, done: int, total: int) -> None:
        """One line per finished game on stderr, with player A's running score."""
        scores.append(record.score_a)
        print(
            f"[{done}/{total}] game {record.index + 1}: {record.result} ({record.termination}, {record.plies} plies)"
            f"  running score {100 * sum(scores) / len(scores):.1f}%",
            file=sys.stderr,
            flush=True,
        )

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    pgn = args.pgn or f"runs/matches/{stamp}.pgn"
    from chess_rl.eval.game import TimeControl

    tc = TimeControl.parse(args.tc) if args.tc else None
    conf = {k: getattr(args, k) for k in ("a", "b", "games", "workers", "opening_plies", "max_plies", "seed", "tc")}
    from chess_rl.eval.registry import parse_spec

    uses_net = any(parse_spec(spec)[0] in {"net", "mcts"} for spec in (args.a, args.b))
    with track_run("match", conf, seed=args.seed, device=pick_device("auto") if uses_net else "cpu") as run:
        result = run_match(
            args.a,
            args.b,
            games=args.games,
            workers=args.workers,
            opening_plies=args.opening_plies,
            max_plies=args.max_plies,
            seed=args.seed,
            pgn_path=pgn,
            stall_timeout=args.stall_timeout,
            progress=progress,
            time_control=tc,
        )
        run.outputs.update(result.estimate.as_dict(), pgn=str(result.pgn_path), terminations=result.terminations())
    print(f"{result.player_a} vs {result.player_b}: {result.estimate}")
    print(f"terminations: {result.terminations()}  games: {result.pgn_path}")
    if not args.no_record:
        append_elo_row(result, args.opening_plies, args.max_plies, args.seed, time_control=tc)
    return 0


def _cmd_uci(args: argparse.Namespace) -> int:
    from chess_rl.play.uci import serve

    serve(args.player, seed=args.seed)
    return 0


def _cmd_tactics(args: argparse.Namespace) -> int:
    from chess_rl.eval import tactics
    from chess_rl.eval.registry import display_name, make_player

    suites = list(tactics.SUITES) if args.suite == "all" else [args.suite]
    player = make_player(args.player, seed=args.seed, name=display_name(args.player))
    try:
        for suite in suites:

            def progress(done: int, total: int, solved: int) -> None:
                if done % 25 == 0 or done == total:
                    print(f"[{done}/{total}] solved {solved}", file=sys.stderr, flush=True)

            conf = {"player": args.player, "suite": suite, "seed": args.seed}
            with track_run("tactics", conf, seed=args.seed, device=str(getattr(player, "device", "cpu"))) as run:
                result = tactics.run_suite(player, suite, progress=progress)
                run.outputs.update(solved=len(result.solved), total=result.total, failed=result.failed)
            print(result)
            if not args.no_record:
                tactics.append_tactics_row(result)
    finally:
        player.close()
    return 0


def _cmd_watch(args: argparse.Namespace) -> int:
    from chess_rl.play import watch

    color = not getattr(args, "no_color", False)
    if args.watch_command == "list":
        watch.list_games(args.pgn)
    elif args.watch_command == "replay":
        watch.replay(args.pgn, args.game, args.delay, color=color)
    else:
        watch.live(args.white, args.black, args.delay, args.max_plies, seed=args.seed, color=color)
    return 0


def _cmd_data(args: argparse.Namespace) -> int:
    from pathlib import Path

    from chess_rl.data import lichess

    if args.data_command == "stats":
        manifest = json.loads((Path(args.dir) / "manifest.json").read_text())
        print(json.dumps({k: manifest[k] for k in ("source", "filter", "stats")}, indent=2))
        return 0
    conf = cfg.load(cfg.DataBuildConfig, args.config, cfg.parse_set_args(args.set or []))
    game_filter = lichess.GameFilter(
        min_elo=conf.min_elo, max_elo=conf.max_elo, time_classes=tuple(conf.time_classes.split(","))
    )
    out_dir = Path(conf.out_root) / conf.name
    url = args.url or lichess.month_url(conf.month)
    print(f"building {out_dir} from {url}", file=sys.stderr, flush=True)
    with track_run("data-build", conf, device="cpu") as run:
        manifest = lichess.build(
            url,
            out_dir,
            game_filter,
            target_positions=conf.target_positions,
            shard_size=conf.shard_size,
            val_every=conf.val_every,
            workers=conf.workers,
            max_bytes=int(conf.max_gb * 1e9),
            stall_timeout=conf.stall_timeout,
            progress=lichess.print_progress,
        )
        run.outputs.update(out_dir=str(out_dir), **manifest["stats"])
    print(json.dumps(manifest["stats"], indent=2))
    return 0


def _cmd_train_sl(args: argparse.Namespace) -> int:
    from chess_rl.train.supervised import TrainConfig, train

    conf = cfg.load(TrainConfig, args.config, cfg.parse_set_args(args.set or []))
    device = pick_device(conf.device)
    with track_run("train-sl", conf, seed=conf.seed, device=device) as run:
        summary = train(conf, resume=args.resume)
        run.outputs.update(summary)
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["stop_reason"] in ("completed", "max_hours") else 1


def _cmd_collect(args: argparse.Namespace) -> int:
    import logging

    from chess_rl.data.personal import Collection, format_report

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    col = Collection.open(args.root)
    if args.collect_command == "sync":
        if not (args.chesscom or args.lichess):
            print("nothing to sync: pass --chesscom USER and/or --lichess USER", file=sys.stderr)
            return 2
        if args.chesscom:
            print(f"chess.com {args.chesscom}: {col.sync_chesscom(args.chesscom)} new games", file=sys.stderr)
        if args.lichess:
            print(f"lichess {args.lichess}: {col.sync_lichess(args.lichess)} new games", file=sys.stderr)
    elif args.collect_command == "import":
        col.aliases.update(a.lower() for a in args.alias or [])
        if not col.aliases:
            print("no owner names known yet: pass --alias NAME (the owner's name in the PGN)", file=sys.stderr)
            return 2
        print(f"imported {col.import_files(args.files)} new games", file=sys.stderr)
    report = col.report()
    print(json.dumps(report, indent=2) if getattr(args, "json", False) else format_report(report))
    return 0


def _cmd_train_human(args: argparse.Namespace) -> int:
    from chess_rl.train.human import HumanTrainConfig, train

    conf = cfg.load(HumanTrainConfig, args.config, cfg.parse_set_args(args.set or []))
    device = pick_device(conf.device)
    with track_run("train-human", conf, seed=conf.seed, device=device) as run:
        summary = train(conf, resume=args.resume)
        run.outputs.update(summary)
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["stop_reason"] in ("completed", "max_hours") else 1


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="YAML config file (see configs/)")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config field (repeatable)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chess-rl", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="show versions, hardware, device, and git state")
    p.set_defaults(func=_cmd_info)

    p = sub.add_parser("stockfish", help="install or locate the pinned Stockfish used for evaluation")
    ssub = p.add_subparsers(dest="stockfish_command", required=True)
    inst = ssub.add_parser("install", help="download and verify the pinned official release (no sudo)")
    inst.add_argument("--force", action="store_true", help="reinstall even if present")
    ssub.add_parser("path", help="print the Stockfish binary that will be used")
    p.set_defaults(func=_cmd_stockfish)

    p = sub.add_parser("model", help="model files")
    msub = p.add_subparsers(dest="model_command", required=True)
    rel = msub.add_parser("release", help="strip a training checkpoint for publishing (no optimizer, no paths)")
    rel.add_argument("src")
    rel.add_argument("dst")
    p.set_defaults(func=_cmd_model)

    p = sub.add_parser("match", help="play a match between two players and estimate Elo")
    p.add_argument("a", help="player spec, e.g. greedy, stockfish:elo=1500,movetime=0.1, mcts:path=...")
    p.add_argument("b", help="opponent spec")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--workers", type=int, default=4, help="parallel game processes")
    p.add_argument("--opening-plies", type=int, default=8, help="max plies of opening-book moves per pair")
    p.add_argument("--max-plies", type=int, default=400, help="adjudicate as a draw after this many plies")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pgn", help="where to write the games (default runs/matches/<timestamp>.pgn)")
    p.add_argument("--no-record", action="store_true", help="do not append to results/elo.csv")
    p.add_argument("--stall-timeout", type=float, default=300.0, help="abort if no game finishes for this long (s)")
    p.add_argument("--tc", help="real clocks, minutes+increment seconds, e.g. 10+5 (default: per-move budgets)")
    p.set_defaults(func=_cmd_match)

    p = sub.add_parser("uci", help="run a player as a UCI engine (for chess GUIs and lichess-bot)")
    p.add_argument("--player", default="greedy", help="player spec served over UCI (changeable via setoption)")
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=_cmd_uci)

    p = sub.add_parser("tactics", help="tactics test suites (Win At Chess, mate in 1/2)")
    tsub = p.add_subparsers(dest="tactics_command", required=True)
    run = tsub.add_parser("run", help="score a player on a suite")
    run.add_argument("--player", required=True, help="player spec")
    run.add_argument("--suite", default="all", choices=["wac", "mate1", "mate2", "all"])
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--no-record", action="store_true", help="do not append to results/tactics.csv")
    p.set_defaults(func=_cmd_tactics)

    p = sub.add_parser("watch", help="view games in the terminal (replay PGNs or watch live)")
    wsub = p.add_subparsers(dest="watch_command", required=True)
    w = wsub.add_parser("list", help="one line per game in a PGN file")
    w.add_argument("pgn")
    w = wsub.add_parser("replay", help="animate a game from a PGN file")
    w.add_argument("pgn")
    w.add_argument("--game", type=int, default=1, help="game number (see watch list)")
    w.add_argument("--delay", type=float, default=0.6, help="seconds per move")
    w.add_argument("--no-color", action="store_true")
    w = wsub.add_parser("live", help="watch two players play a game")
    w.add_argument("white", help="player spec")
    w.add_argument("black", help="player spec")
    w.add_argument("--delay", type=float, default=0.3)
    w.add_argument("--max-plies", type=int, default=400)
    w.add_argument("--seed", type=int, default=None)
    w.add_argument("--no-color", action="store_true")
    p.set_defaults(func=_cmd_watch)

    p = sub.add_parser("data", help="build and inspect training datasets")
    dsub = p.add_subparsers(dest="data_command", required=True)
    b = dsub.add_parser("build", help="stream a Lichess month into train/val shards")
    _add_config_args(b)
    b.add_argument("--url", help="override the source URL (e.g. a local file:// copy)")
    st = dsub.add_parser("stats", help="show a dataset's manifest summary")
    st.add_argument("dir")
    p.set_defaults(func=_cmd_data)

    p = sub.add_parser("collect", help="collect the owner's games for Om-sim (stored locally, never committed)")
    p.add_argument("--root", default="data/personal/om", help="collection directory")
    csub = p.add_subparsers(dest="collect_command", required=True)
    c = csub.add_parser("sync", help="fetch new games from chess.com / Lichess (incremental)")
    c.add_argument("--chesscom", help="chess.com username")
    c.add_argument("--lichess", help="Lichess username")
    c = csub.add_parser("import", help="add games from PGN files (e.g. games vs the bot saved by a GUI)")
    c.add_argument("files", nargs="+")
    c.add_argument("--alias", action="append", help="the owner's player name in these PGNs (repeatable)")
    c = csub.add_parser("report", help="game counts by time control")
    c.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_collect)

    p = sub.add_parser("train", help="train networks")
    trsub = p.add_subparsers(dest="train_command", required=True)
    sl = trsub.add_parser("sl", help="supervised pretraining on a Lichess dataset (M4)")
    _add_config_args(sl)
    sl.add_argument("--resume", help="checkpoint to resume from (e.g. runs/m4/latest.pt)")
    sl.set_defaults(func=_cmd_train_sl)
    hu = trsub.add_parser("human", help="human move model on a rating-band Lichess set (M6)")
    _add_config_args(hu)
    hu.add_argument("--resume", help="checkpoint to resume from (e.g. runs/human/latest.pt)")
    hu.set_defaults(func=_cmd_train_human)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
