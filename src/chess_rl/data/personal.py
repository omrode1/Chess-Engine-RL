"""Personal game collection for Om-sim (M6): the owner's games from chess.com, Lichess and games vs the bot.

Only the owner, or players who explicitly opt in, are collected. Everything is stored locally and never
committed (`/data/` is git-ignored):

    data/personal/<player>/games.pgn    every game once, with [Source], [GameId] and [OwnerColor] headers
    data/personal/<player>/index.json   one entry per game + sync state (finished chess.com months, Lichess cursor)

Syncing is incremental: finished chess.com months are fetched once, the current month is re-fetched,
Lichess is asked only for games after the newest one seen. Games are keyed by their URL (or, for imported
PGNs, a hash of the players, date and moves), so running a sync twice adds nothing.

    uv run chess-rl collect sync --chesscom omrode31 [--lichess NAME]
    uv run chess-rl collect import data/om/games/*.pgn --alias "Om Rode"
    uv run chess-rl collect report
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import chess
import chess.pgn

from chess_rl.data.lichess import iter_games, time_class

log = logging.getLogger(__name__)

DEFAULT_ROOT = Path("data/personal")
USER_AGENT = "chess-rl-collector (https://github.com/omrode1/Chess-Engine-RL)"
CHESSCOM_ARCHIVES = "https://api.chess.com/pub/player/{user}/games/archives"
LICHESS_GAMES = "https://lichess.org/api/games/user/{user}?since={since}&clocks=true&opening=true&sort=dateAsc"

Fetcher = Callable[[str, dict[str, str]], bytes]


def http_get(url: str, headers: dict[str, str], timeout: float = 60.0, retries: int = 4) -> bytes:
    """GET with a timeout and backoff on 429/5xx (both APIs ask clients to back off when rate-limited)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **headers})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries:
                raise
            wait = 60.0 if e.code == 429 else 2.0**attempt
            log.warning("%s: HTTP %d, retrying in %.0fs", url, e.code, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


@dataclass
class GameEntry:
    source: str  # chess.com | lichess | import
    owner_color: str  # white | black
    date: str  # YYYY.MM.DD
    time_control: str
    time_class: str
    result: str  # owner's result: win | draw | loss | unknown
    plies: int
    opponent: str
    opponent_elo: int | None
    owner_elo: int | None


@dataclass
class Collection:
    root: Path
    games: dict[str, GameEntry] = field(default_factory=dict)
    aliases: set[str] = field(default_factory=set)  # lower-case names the owner plays under
    sync: dict[str, dict] = field(default_factory=dict)

    @property
    def pgn_path(self) -> Path:
        return self.root / "games.pgn"

    @property
    def index_path(self) -> Path:
        return self.root / "index.json"

    @classmethod
    def open(cls, root: Path | str) -> Collection:
        root = Path(root)
        index = root / "index.json"
        if not index.exists():
            return cls(root)
        data = json.loads(index.read_text())
        return cls(
            root,
            games={k: GameEntry(**v) for k, v in data["games"].items()},
            aliases=set(data.get("aliases", [])),
            sync=data.get("sync", {}),
        )

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        data = {
            "aliases": sorted(self.aliases),
            "sync": self.sync,
            "games": {k: vars(v) for k, v in sorted(self.games.items(), key=lambda kv: (kv[1].date, kv[0]))},
        }
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(self.index_path)

    # ----------------------------------------------------------------------------------------------

    def add_pgn(self, pgn_text: str, source: str) -> bool:
        """Add one game if it is new, standard chess, and the owner is one of the players. True if added."""
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None or game.errors:
            return False
        h = game.headers
        if h.get("Variant", "Standard").lower() not in ("standard", "chess") or "FEN" in h:
            return False
        names = {"white": h.get("White", "").lower(), "black": h.get("Black", "").lower()}
        sides = [c for c, n in names.items() if n in self.aliases]
        if len(sides) != 1:
            return False
        owner = sides[0]
        moves = [m.uci() for m in game.mainline_moves()]
        if not moves:
            return False
        game_id = h.get("Link") or h.get("Site", "")
        if not game_id.startswith("http") or game_id.rstrip("/").endswith(("chess.com", "lichess.org")):
            key = "|".join([h.get("White", ""), h.get("Black", ""), h.get("Date", ""), " ".join(moves)])
            game_id = "sha:" + hashlib.blake2b(key.encode(), digest_size=10).hexdigest()
        if game_id in self.games:
            return False
        opp = "black" if owner == "white" else "white"
        score = {"1-0": "white", "0-1": "black", "1/2-1/2": "draw"}.get(h.get("Result", "*"))
        result = "unknown" if score is None else "draw" if score == "draw" else "win" if score == owner else "loss"
        tc = h.get("TimeControl", "-")
        self.games[game_id] = GameEntry(
            source=source,
            owner_color=owner,
            date=h.get("UTCDate") or h.get("Date", "????.??.??"),
            time_control=tc,
            time_class=_time_class(tc),
            result=result,
            plies=len(moves),
            opponent=h.get(opp.capitalize(), "?"),
            opponent_elo=_int(h.get(f"{opp.capitalize()}Elo")),
            owner_elo=_int(h.get(f"{owner.capitalize()}Elo")),
        )
        h["Source"], h["GameId"], h["OwnerColor"] = source, game_id, owner
        self.root.mkdir(parents=True, exist_ok=True)
        with self.pgn_path.open("a") as f:
            print(game.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=True)), file=f)
            print(file=f)
        return True

    def add_many(self, pgn_texts: Iterable[str], source: str) -> int:
        return sum(self.add_pgn(text, source) for text in pgn_texts)

    # ----------------------------------------------------------------------------------------------

    def sync_chesscom(self, user: str, fetch: Fetcher = http_get, now: datetime | None = None) -> int:
        """Fetch new chess.com games. Months before the current one are final and fetched only once."""
        user = user.lower()
        self.aliases.add(user)
        state = self.sync.setdefault(f"chess.com:{user}", {"done_months": []})
        current = (now or datetime.now(UTC)).strftime("%Y/%m")
        archives = json.loads(fetch(CHESSCOM_ARCHIVES.format(user=user), {}))["archives"]
        added = 0
        for url in archives:
            month = "/".join(url.rstrip("/").split("/")[-2:])
            if month in state["done_months"]:
                continue
            games = json.loads(fetch(url, {}))["games"]
            pgns = [g["pgn"] for g in games if g.get("rules") == "chess" and g.get("pgn")]
            n = self.add_many(pgns, "chess.com")
            added += n
            log.info("chess.com %s: %d games, %d new", month, len(games), n)
            if month < current:
                state["done_months"].append(month)
            self.save()  # progress survives an interrupted sync
        return added

    def sync_lichess(self, user: str, fetch: Fetcher = http_get) -> int:
        """Fetch Lichess games played after the newest one already collected."""
        user = user.lower()
        self.aliases.add(user)
        state = self.sync.setdefault(f"lichess:{user}", {"since_ms": 0})
        text = fetch(LICHESS_GAMES.format(user=user, since=state["since_ms"]), {"Accept": "application/x-chess-pgn"})
        added, newest = 0, state["since_ms"]
        for headers, pgn in iter_games(io.StringIO(text.decode())):
            added += self.add_pgn(pgn, "lichess")
            stamp = _utc_ms(headers.get("UTCDate"), headers.get("UTCTime"))
            newest = max(newest, stamp + 1000 if stamp else 0)
        state["since_ms"] = newest
        self.save()
        return added

    def import_files(self, paths: Iterable[Path | str]) -> int:
        added = 0
        for path in paths:
            with open(path) as f:
                added += self.add_many((pgn for _, pgn in iter_games(f)), "import")
        self.save()
        return added

    # ----------------------------------------------------------------------------------------------

    def report(self) -> dict:
        games = list(self.games.values())
        by_class = Counter(g.time_class for g in games)
        results = {tc: dict(Counter(g.result for g in games if g.time_class == tc)) for tc in sorted(by_class)}
        return {
            "games": len(games),
            "owner_moves": sum((g.plies + (g.owner_color == "white")) // 2 for g in games),
            "by_source": dict(Counter(g.source for g in games)),
            "by_time_class": dict(by_class),
            "results_by_time_class": results,
            "first": min((g.date for g in games), default=None),
            "last": max((g.date for g in games), default=None),
        }


def format_report(report: dict) -> str:
    lines = [
        f"{report['games']} games, {report['owner_moves']} of the owner's moves ({report['first']} to {report['last']})",
        "sources: " + ", ".join(f"{k} {v}" for k, v in report["by_source"].items()),
        f"{'time class':<12}{'games':>6}{'W':>5}{'D':>5}{'L':>5}",
    ]
    for tc, n in sorted(report["by_time_class"].items(), key=lambda kv: -kv[1]):
        r = report["results_by_time_class"][tc]
        lines.append(f"{tc:<12}{n:>6}{r.get('win', 0):>5}{r.get('draw', 0):>5}{r.get('loss', 0):>5}")
    return "\n".join(lines)


def _time_class(tc: str) -> str:
    """Lichess speed classes; chess.com writes increment-free controls as '600' and daily games as '1/86400'."""
    if "/" in tc:
        return "daily"
    return time_class(tc if "+" in tc else f"{tc}+0") or "unknown"


def _int(value: str | None) -> int | None:
    try:
        return int(value) if value else None
    except ValueError:
        return None


def _utc_ms(date: str | None, clock: str | None) -> int:
    try:
        stamp = datetime.strptime(f"{date} {clock}", "%Y.%m.%d %H:%M:%S").replace(tzinfo=UTC)
    except (TypeError, ValueError):
        return 0
    return int(stamp.timestamp() * 1000)
