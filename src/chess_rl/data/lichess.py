"""Build supervised training shards from the Lichess open database (https://database.lichess.org, CC0).

The monthly `.pgn.zst` file is streamed over HTTP and decompressed on the fly. Nothing raw is written
to disk, and the stream stops once `target_positions` is reached. Headers are filtered in the main
process (fast), and only accepted games are parsed and encoded, in worker processes.

For every position of an accepted game the shards store:

    bitboards uint64[13], flags uint8, halfmove uint8   compact board (chess_rl.encoding.compact)
    move      int16    policy index of the move played (chess_rl.encoding.moves)
    result    int8     game result for the side to move: +1 win, 0 draw, -1 loss
    elo_us, elo_them   int16   ratings of the side to move and its opponent
    base_time int16, increment int16   time control in seconds
    ply       int16    ply number of the position
    clock     int16    seconds left on the mover's clock before the move ([%clk]); -1 if unknown
    wp_loss   int8     win-probability points (0-100) the move lost for the mover, from Lichess's [%eval]
                       annotations (about 1 game in 8 has them); -1 if unknown. A blunder is wp_loss >= 10.

Shards built before M6 lack clock and wp_loss; readers load only the fields they need.

About 2% of games (by a hash of the game URL) go to the validation split. A manifest records the
source URL, its Last-Modified date, the bytes consumed, the filter, and the counts.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import multiprocessing as mp
import re
import sys
import time
import urllib.request
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import chess.pgn
import numpy as np

from chess_rl.encoding.compact import compact_board
from chess_rl.encoding.moves import move_to_index

log = logging.getLogger(__name__)

USER_AGENT = "chess-rl-data-pipeline (https://github.com/omrode1/Chess-Engine-RL)"
FIELDS = {
    "bitboards": (np.uint64, (13,)),
    "flags": (np.uint8, ()),
    "halfmove": (np.uint8, ()),
    "move": (np.int16, ()),
    "result": (np.int8, ()),
    "elo_us": (np.int16, ()),
    "elo_them": (np.int16, ()),
    "base_time": (np.int16, ()),
    "increment": (np.int16, ()),
    "ply": (np.int16, ()),
    "clock": (np.int16, ()),
    "wp_loss": (np.int8, ()),
}
_HEADER = re.compile(r'^\[(\w+) "(.*)"\]\s*$')


def month_url(month: str) -> str:
    return f"https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"


# --------------------------------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------------------------------


def time_class(time_control: str) -> str | None:
    """Lichess speed class from a TimeControl header ('300+3'); None for correspondence/unknown."""
    try:
        base, inc = (int(x) for x in time_control.split("+"))
    except ValueError:
        return None
    estimate = base + 40 * inc
    if estimate < 30:
        return "ultrabullet"
    if estimate < 180:
        return "bullet"
    if estimate < 480:
        return "blitz"
    if estimate < 1500:
        return "rapid"
    return "classical"


@dataclass(frozen=True)
class GameFilter:
    min_elo: int = 1800  # both players
    max_elo: int = 4000  # both players
    time_classes: tuple[str, ...] = ("blitz", "rapid", "classical")
    min_plies_time_forfeit: int = 40  # drop games lost on time before move 20

    def accepts(self, headers: dict[str, str]) -> bool:
        try:
            white, black = int(headers["WhiteElo"]), int(headers["BlackElo"])
        except (KeyError, ValueError):
            return False
        if not (self.min_elo <= white <= self.max_elo and self.min_elo <= black <= self.max_elo):
            return False
        if headers.get("Result") not in ("1-0", "0-1", "1/2-1/2"):
            return False
        if headers.get("Termination") in ("Abandoned", "Rules infraction", "Unterminated"):
            return False
        if "FEN" in headers or headers.get("Variant", "Standard") != "Standard":
            return False
        return time_class(headers.get("TimeControl", "-")) in self.time_classes


# --------------------------------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------------------------------


class CountingReader(io.RawIOBase):
    """Wraps a byte stream and counts bytes read (for progress and the manifest)."""

    def __init__(self, raw):
        self.raw, self.count = raw, 0

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        data = self.raw.read(len(b))
        n = len(data)
        b[:n] = data
        self.count += n
        return n


def open_source(url: str, timeout: float = 60.0):
    """(counting reader, Last-Modified). Socket reads time out instead of hanging forever."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    resp = urllib.request.urlopen(req, timeout=timeout)
    return CountingReader(resp), resp.headers.get("Last-Modified")


def iter_games(text_stream) -> Iterator[tuple[dict[str, str], str]]:
    """Yield (headers, full PGN text) per game from a PGN text stream, without parsing moves."""
    headers: dict[str, str] = {}
    lines: list[str] = []
    in_moves = False
    for line in text_stream:
        if line.startswith("["):
            if in_moves:  # next game started
                yield headers, "".join(lines)
                headers, lines, in_moves = {}, [], False
            m = _HEADER.match(line)
            if m:
                headers[m.group(1)] = m.group(2)
            lines.append(line)
        elif line.strip():
            in_moves = True
            lines.append(line)
        else:
            lines.append(line)
    if headers and in_moves:
        yield headers, "".join(lines)


def is_validation(headers: dict[str, str], every: int) -> bool:
    key = headers.get("Site") or f"{headers.get('White')}-{headers.get('Black')}-{headers.get('UTCDate')}"
    return int(hashlib.blake2b(key.encode(), digest_size=8).hexdigest(), 16) % every == 0


# --------------------------------------------------------------------------------------------------
# Encoding (runs in worker processes)
# --------------------------------------------------------------------------------------------------


START_CP = 20  # roughly Stockfish's evaluation of the starting position, in centipawns


def win_percent(cp: float) -> float:
    """Lichess's win% for the side the centipawns are counted for (lichess.org/page/accuracy)."""
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def _white_cp(node: chess.pgn.ChildNode) -> int | None:
    score = node.eval()
    return None if score is None else score.white().score(mate_score=10_000)


def iter_positions(game: chess.pgn.Game) -> Iterator[tuple[chess.Board, chess.Move, int, int]]:
    """(board before the move, move, mover's clock in seconds or -1, wp_loss or -1) for each mainline move.

    The board is mutated after each yield: copy it if you keep it.
    """
    base = int(game.headers.get("TimeControl", "0+0").split("+")[0] or 0)
    board = game.board()
    nodes = list(game.mainline())
    has_evals = any(n.eval() is not None for n in nodes)
    before_cp: int | None = START_CP  # White's view
    for node in nodes:
        # the mover's clock now = the [%clk] after their previous move, two plies back
        if board.ply() < 2:
            clock: float | None = float(base)
        else:
            clock = node.parent.parent.clock()  # type: ignore[union-attr]
        after_cp = _white_cp(node)
        if node.board().is_checkmate():
            after_cp = 10_000 if board.turn == chess.WHITE else -10_000
        wp_loss = -1
        if has_evals and before_cp is not None and after_cp is not None:
            sign = 1 if board.turn == chess.WHITE else -1
            drop = win_percent(sign * before_cp) - win_percent(sign * after_cp)
            wp_loss = int(round(min(max(drop, 0.0), 100.0)))
        before_cp = after_cp
        yield board, node.move, -1 if clock is None else min(int(clock), 32767), wp_loss
        board.push(node.move)


def encode_game(pgn_text: str) -> dict[str, np.ndarray] | None:
    """All positions of one game as column arrays, or None if the game can't be replayed."""
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None or game.errors:
        return None
    h = game.headers
    result = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}[h["Result"]]
    elos = {chess.WHITE: int(h["WhiteElo"]), chess.BLACK: int(h["BlackElo"])}
    base, inc = (int(x) for x in h["TimeControl"].split("+"))
    rows: dict[str, list] = {k: [] for k in FIELDS}
    for board, move, clock, wp_loss in iter_positions(game):
        bb, flags, half = compact_board(board)
        rows["bitboards"].append(bb)
        rows["flags"].append(flags)
        rows["halfmove"].append(half)
        rows["move"].append(move_to_index(move, board.turn))
        rows["result"].append(result if board.turn == chess.WHITE else -result)
        rows["elo_us"].append(elos[board.turn])
        rows["elo_them"].append(elos[not board.turn])
        rows["base_time"].append(min(base, 32767))
        rows["increment"].append(min(inc, 32767))
        rows["ply"].append(min(board.ply(), 32767))
        rows["clock"].append(clock)
        rows["wp_loss"].append(wp_loss)
    if not rows["move"]:
        return None
    return {k: np.asarray(v, dtype=FIELDS[k][0]) for k, v in rows.items()}


def _encode_batch(batch: list[tuple[str, bool]]) -> list[tuple[dict[str, np.ndarray] | None, bool]]:
    return [(encode_game(text), val) for text, val in batch]


# --------------------------------------------------------------------------------------------------
# Shards
# --------------------------------------------------------------------------------------------------


class ShardWriter:
    def __init__(self, out_dir: Path, prefix: str, shard_size: int):
        self.out_dir, self.prefix, self.shard_size = out_dir, prefix, shard_size
        self.buffer: dict[str, list[np.ndarray]] = {k: [] for k in FIELDS}
        self.buffered = 0
        self.shards: list[str] = []
        self.positions = 0

    def add(self, game: dict[str, np.ndarray]) -> None:
        for k, v in game.items():
            self.buffer[k].append(v)
        self.buffered += len(game["move"])
        if self.buffered >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffered:
            return
        name = f"{self.prefix}_{len(self.shards):05d}.npz"
        arrays = {k: np.concatenate(v) for k, v in self.buffer.items()}
        np.savez(self.out_dir / name, **arrays)
        self.shards.append(name)
        self.positions += self.buffered
        self.buffer = {k: [] for k in FIELDS}
        self.buffered = 0


@dataclass
class BuildStats:
    bytes_read: int = 0
    games_seen: int = 0
    games_accepted: int = 0
    games_failed: int = 0
    train_positions: int = 0
    val_positions: int = 0
    seconds: float = 0.0


def build(
    url: str,
    out_dir: str | Path,
    game_filter: GameFilter,
    target_positions: int,
    shard_size: int = 1_000_000,
    val_every: int = 50,
    workers: int = 8,
    max_bytes: int | None = None,
    stall_timeout: float = 120.0,
    progress: Callable[[BuildStats], None] | None = None,
    progress_every: float = 15.0,
) -> dict:
    """Stream `url`, write train/val shards to out_dir, return the manifest (also written as manifest.json).

    Stops at target_positions (train + val) or max_bytes of compressed input. A socket read that stalls
    for stall_timeout seconds raises instead of hanging.
    """
    import zstandard

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("*.npz")):
        raise FileExistsError(f"{out} already has shards; use a new directory")
    raw, last_modified = open_source(url, timeout=stall_timeout)
    text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw), encoding="utf-8", errors="replace")
    train, val = ShardWriter(out, "train", shard_size), ShardWriter(out, "val", shard_size)
    stats = BuildStats()
    t0 = last_report = time.monotonic()

    def batches() -> Iterator[list[tuple[str, bool]]]:
        batch: list[tuple[str, bool]] = []
        for headers, pgn in iter_games(text):
            stats.games_seen += 1
            stats.bytes_read = raw.count
            if game_filter.accepts(headers):
                batch.append((pgn, is_validation(headers, val_every)))
                if len(batch) == 64:
                    yield batch
                    batch = []
            if max_bytes is not None and raw.count >= max_bytes:
                break
        if batch:
            yield batch

    def consume(results: list[tuple[dict[str, np.ndarray] | None, bool]]) -> None:
        for game, is_val in results:
            if game is None:
                stats.games_failed += 1
                continue
            stats.games_accepted += 1
            (val if is_val else train).add(game)
        stats.train_positions = train.positions + train.buffered
        stats.val_positions = val.positions + val.buffered

    # Bounded in-flight window: reading stays at most `max_inflight` batches ahead of the workers, so
    # memory stays flat (Pool.imap would drain the whole stream into its task queue).
    max_inflight = workers * 4
    pool = mp.get_context("spawn").Pool(workers)
    pending: deque = deque()
    try:
        for batch in batches():
            pending.append(pool.apply_async(_encode_batch, (batch,)))
            while len(pending) >= max_inflight:
                consume(pending.popleft().get(timeout=stall_timeout))
            now = time.monotonic()
            if progress and now - last_report >= progress_every:
                stats.seconds = now - t0
                progress(stats)
                last_report = now
            if stats.train_positions + stats.val_positions >= target_positions:
                break
        while pending and stats.train_positions + stats.val_positions < target_positions:
            consume(pending.popleft().get(timeout=stall_timeout))
    finally:
        pool.terminate()
        pool.join()
        raw.raw.close()
    train.flush()
    val.flush()
    stats.bytes_read = raw.count
    stats.train_positions, stats.val_positions = train.positions, val.positions
    stats.seconds = time.monotonic() - t0
    if progress:
        progress(stats)
    manifest = {
        "source": url,
        "source_last_modified": last_modified,
        "compressed_bytes_consumed": stats.bytes_read,
        "filter": asdict(game_filter),
        "target_positions": target_positions,
        "val_every": val_every,
        "fields": {k: [np.dtype(t).name, list(s)] for k, (t, s) in FIELDS.items()},
        "stats": asdict(stats),
        "train_shards": train.shards,
        "val_shards": val.shards,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def format_progress(s: BuildStats) -> str:
    rate = (s.train_positions + s.val_positions) / s.seconds if s.seconds else 0.0
    return (
        f"[{s.seconds / 60:5.1f} min] {s.bytes_read / 1e9:6.2f} GB read | games {s.games_seen:,} seen, "
        f"{s.games_accepted:,} kept | positions train {s.train_positions:,} val {s.val_positions:,} "
        f"({rate:,.0f}/s)"
    )


def print_progress(s: BuildStats) -> None:
    print(format_progress(s), file=sys.stderr, flush=True)
