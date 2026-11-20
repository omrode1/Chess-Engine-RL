"""`chess-rl` command line. Subcommands are added per milestone (data, train, match, uci, play, ...)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

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


def _cmd_human(args: argparse.Namespace) -> int:
    from chess_rl.data import lichess
    from chess_rl.eval import human

    if args.human_command == "testset":
        game_filter = lichess.GameFilter(
            min_elo=args.min_elo, max_elo=args.max_elo, time_classes=tuple(args.time_classes.split(","))
        )
        url = args.url or lichess.month_url(args.month)
        print(f"test set from {url} -> {args.out}", file=sys.stderr, flush=True)
        with track_run("human-testset", vars(args) | {"func": None}, device="cpu") as run:
            meta = human.build_test_set(
                url,
                args.out,
                game_filter,
                args.games,
                progress=lambda g, p: print(f"  {g:,} games, {p:,} positions", file=sys.stderr, flush=True),
            )
            run.outputs.update(meta)
        print(json.dumps(meta, indent=2))
        return 0

    rows = human.read_jsonl(args.test_set)
    results = {}
    import torch

    device = torch.device(pick_device("auto"))
    with track_run("human-eval", vars(args) | {"func": None}, device=str(device)) as run:
        for spec in args.model or []:
            name, _, path = spec.rpartition("=")
            name = name or Path(path).stem
            print(f"scoring {name} ({path}) on {len(rows):,} positions", file=sys.stderr, flush=True)
            preds = human.predict(human.load_model(path, device), rows, device)
            if args.save_preds:
                human.write_jsonl(Path(args.save_preds) / f"{name}.jsonl", preds)
            results[name] = human.summarize(rows, preds)
        for spec in args.preds or []:
            name, _, path = spec.rpartition("=")
            results[name or Path(path).stem] = human.summarize(rows, human.read_jsonl(path))
        run.outputs.update(results=results)
    print(json.dumps(results, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"test_set": args.test_set, "results": results}, indent=2) + "\n")
    return 0


def _cmd_omsim(args: argparse.Namespace) -> int:
    import torch

    from chess_rl.eval import human
    from chess_rl.eval.stockfish import find_stockfish
    from chess_rl.model.human import load_human, save_human
    from chess_rl.train import omsim

    root = Path(args.root)
    rows_path = root / "rows.jsonl"
    if args.omsim_command == "label":
        rows = omsim.player_rows(root / "games.pgn")
        cached = {r["id"]: r["wp_loss"] for r in human.read_jsonl(rows_path)} if rows_path.exists() else {}
        cached = {} if args.relabel else cached
        todo = [r for r in rows if cached.get(r["id"], -1) < 0]
        print(f"{len(rows)} moves, {len(todo)} to label with Stockfish depth {args.depth}", file=sys.stderr)
        omsim.label_blunders(todo, find_stockfish(), depth=args.depth)
        old = {r["id"]: r for r in human.read_jsonl(rows_path)} if rows_path.exists() and not args.relabel else {}
        rows = [old.get(r["id"], r) if r["wp_loss"] < 0 else r for r in rows]
        human.write_jsonl(rows_path, rows)
        labelled = [r for r in rows if r["wp_loss"] >= 0]
        rate = sum(r["wp_loss"] >= 10 for r in labelled) / max(len(labelled), 1)
        print(f"{len(labelled)} labelled moves, blunder rate {rate:.3f} -> {rows_path}", file=sys.stderr)
        return 0

    if not rows_path.exists():
        print(f"{rows_path} missing: run `chess-rl omsim label` first", file=sys.stderr)
        return 2
    rows = human.read_jsonl(rows_path)
    if args.omsim_command == "report":
        from chess_rl.eval import weakness

        risk = None
        if args.model:
            model, meta = load_human(args.model, torch.device("cpu"))
            preds = omsim.predict(model, rows, torch.device("cpu"), meta.get("offset", 0), meta.get("player"))
            risk = weakness.risk_from_predictions(preds)
        population = None
        if args.population and Path(args.population).exists():
            labelled = [r["wp_loss"] for r in human.read_jsonl(args.population) if r["wp_loss"] >= 0]
            population = sum(w >= weakness.BLUNDER for w in labelled) / max(len(labelled), 1)
        report = weakness.build(rows, risk, population)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(weakness.render_html(report, rows, risk, title=args.title))
        print(weakness.markdown_summary(report))
        print(f"report -> {args.out}", file=sys.stderr)
        return 0
    device = torch.device(pick_device("auto"))
    conf = omsim.OmSimConfig(steps=args.steps, lr=args.lr, l2=args.l2)
    with track_run(f"omsim-{args.omsim_command}", vars(args) | {"func": None}, device=str(device)) as run:
        if args.omsim_command == "fit":
            model, meta = load_human(args.model, device)
            offset = omsim.fit_offset(model, rows, device, conf.offsets)
            player = model.add_players(1)
            curve = omsim.fit_player(model, rows, device, offset, conf, player)
            save_human(args.out, model, base=str(args.model), offset=offset, player=player, moves=len(rows))
            run.outputs.update(offset=offset, player=player, moves=len(rows), final_loss=curve[-1], out=args.out)
            print(json.dumps(run.outputs, indent=2))
            return 0
        result = omsim.run_eval(
            lambda: load_human(args.model, device)[0],
            rows,
            device,
            conf,
            args.holdout,
            args.min_train,
            log=lambda s: print(s, file=sys.stderr, flush=True),
        )
        preds = result.pop("predictions")
        for scheme in ("split", "forward_chaining"):
            for name, path in (p.split("=", 1) for p in args.preds or []):
                result[scheme][name] = human.summarize(
                    [r for r in rows if r["id"] in {p["id"] for p in preds[scheme]["om-sim"]}], human.read_jsonl(path)
                )
            result[scheme]["om-sim_minus_base@offset"] = omsim.paired_difference(
                rows, preds[scheme]["om-sim"], preds[scheme]["base@offset"]
            )
        if args.save_preds:
            for scheme, by_method in preds.items():
                for name, items in by_method.items():
                    human.write_jsonl(Path(args.save_preds) / f"{scheme}-{name.replace('@', '_')}.jsonl", items)
        run.outputs.update(result)
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    return 0


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

    p = sub.add_parser("human", help="human move prediction: test sets and evaluation (M6)")
    hsub = p.add_subparsers(dest="human_command", required=True)
    h = hsub.add_parser("testset", help="positions from N games of a Lichess month (use one no model trained on)")
    h.add_argument("--month", default="2026-09")
    h.add_argument("--url", help="override the source URL")
    h.add_argument("--games", type=int, default=3000)
    h.add_argument("--min-elo", type=int, default=600)
    h.add_argument("--max-elo", type=int, default=1600)
    h.add_argument("--time-classes", default="blitz,rapid,classical")
    h.add_argument("--out", default="data/lichess/human-test-2026-09.jsonl")
    h = hsub.add_parser("eval", help="move-match accuracy and blunder AUC on a test set")
    h.add_argument("--test-set", required=True)
    h.add_argument("--model", action="append", help="[name=]checkpoint (HumanNet or ChessNet), repeatable")
    h.add_argument("--preds", action="append", help="[name=]predictions.jsonl from another model (e.g. Maia-2)")
    h.add_argument("--save-preds", help="directory to write each model's predictions")
    h.add_argument("--out", help="write the results JSON here (e.g. results/m6/human-model.json)")
    p.set_defaults(func=_cmd_human)

    p = sub.add_parser("omsim", help="Om-sim: few-shot model of the owner from his games (M6)")
    p.add_argument("--root", default="data/personal/om", help="collection directory (chess-rl collect)")
    osub = p.add_subparsers(dest="omsim_command", required=True)
    o = osub.add_parser("label", help="Stockfish blunder labels for the owner's moves (cached in rows.jsonl)")
    o.add_argument("--depth", type=int, default=16)
    o.add_argument("--relabel", action="store_true", help="ignore cached labels")
    for name, help_ in (("eval", "held-out evaluation vs the rating-band base"), ("fit", "fit on all games")):
        o = osub.add_parser(name, help=help_)
        o.add_argument("--model", default="runs/human/best.pt", help="rating-band human model (#35)")
        o.add_argument("--steps", type=int, default=300)
        o.add_argument("--lr", type=float, default=0.05)
        o.add_argument("--l2", type=float, default=1e-4)
        if name == "eval":
            o.add_argument("--holdout", type=float, default=0.2, help="most recent fraction of games held out")
            o.add_argument("--min-train", type=int, default=5, help="forward chaining starts after this many games")
            o.add_argument("--preds", action="append", help="name=predictions.jsonl of another model (e.g. Maia-2)")
            o.add_argument("--save-preds", help="directory for each method's predictions")
            o.add_argument("--out", help="results JSON (aggregates only; e.g. results/m6/omsim.json)")
        else:
            o.add_argument("--out", default="runs/omsim/om-sim.pt")
    o = osub.add_parser("report", help="weakness report (HTML) from his blunders and Om-sim's risk estimates")
    o.add_argument("--model", help="Om-sim checkpoint from `omsim fit` (adds predicted blunder risk)")
    o.add_argument(
        "--population",
        default="data/lichess/human-test-2026-09.jsonl",
        help="human test set to quote the population blunder rate from",
    )
    o.add_argument("--out", default="data/personal/om/weakness-report.html")
    o.add_argument("--title", default="Weakness report")
    p.set_defaults(func=_cmd_omsim)

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
