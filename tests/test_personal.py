import json
from datetime import UTC, datetime

from chess_rl.cli import main
from chess_rl.data.personal import Collection


def pgn(white: str, black: str, moves: str, result: str = "1-0", link: str | None = None, tc: str = "600") -> str:
    headers = {
        "Event": "Live Chess",
        "Site": "Chess.com",
        "Date": "2026.11.01",
        "White": white,
        "Black": black,
        "Result": result,
        "WhiteElo": "500",
        "BlackElo": "480",
        "TimeControl": tc,
        "UTCDate": "2026.11.01",
        "UTCTime": "10:00:00",
    }
    if link:
        headers["Link"] = link
    head = "\n".join(f'[{k} "{v}"]' for k, v in headers.items())
    return f"{head}\n\n{moves} {result}\n"


class FakeChessCom:
    def __init__(self) -> None:
        self.months = {
            "2026/10": [pgn("Owner", "a", "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7#", link="https://chess.com/game/1")],
            "2026/11": [pgn("b", "OWNER", "1. d4 d5", "0-1", link="https://chess.com/game/2", tc="180+2")],
        }
        self.calls: list[str] = []

    def __call__(self, url: str, headers: dict[str, str]) -> bytes:
        self.calls.append(url)
        if url.endswith("/archives"):
            return json.dumps(
                {"archives": [f"https://api.chess.com/pub/player/owner/games/{m}" for m in self.months]}
            ).encode()
        month = "/".join(url.split("/")[-2:])
        return json.dumps({"games": [{"rules": "chess", "pgn": p} for p in self.months[month]]}).encode()


NOW = datetime(2026, 11, 20, tzinfo=UTC)


def test_chesscom_sync_is_incremental_and_deduplicated(tmp_path):
    fake = FakeChessCom()
    col = Collection.open(tmp_path)
    assert col.sync_chesscom("Owner", fetch=fake, now=NOW) == 2
    assert len(fake.calls) == 3

    # reopen from disk: the finished month is skipped, the current one is re-fetched but adds nothing
    fake.calls.clear()
    col = Collection.open(tmp_path)
    assert col.sync_chesscom("owner", fetch=fake, now=NOW) == 0
    assert [u.split("/")[-1] for u in fake.calls] == ["archives", "11"]

    # a new game in the current month is picked up
    fake.months["2026/11"].append(pgn("Owner", "c", "1. c4 e5", "1/2-1/2", link="https://chess.com/game/3"))
    assert col.sync_chesscom("owner", fetch=fake, now=NOW) == 1
    assert col.pgn_path.read_text().count("[GameId") == 3

    games = Collection.open(tmp_path).games
    assert games["https://chess.com/game/1"].owner_color == "white"
    assert games["https://chess.com/game/1"].result == "win"
    assert games["https://chess.com/game/2"].owner_color == "black"
    assert games["https://chess.com/game/2"].result == "win"
    assert games["https://chess.com/game/2"].time_class == "blitz"
    assert games["https://chess.com/game/3"].result == "draw"
    assert games["https://chess.com/game/1"].time_class == "rapid"  # '600' has no '+inc'


def test_lichess_sync_moves_the_cursor(tmp_path):
    seen: list[str] = []
    body = pgn("owner", "x", "1. e4 c5", "0-1", link=None).replace("Chess.com", "https://lichess.org/abcd1234")

    def fetch(url: str, headers: dict[str, str]) -> bytes:
        seen.append(url)
        return body.encode()

    col = Collection.open(tmp_path)
    assert col.sync_lichess("Owner", fetch=fetch) == 1
    assert "since=0" in seen[0]
    assert col.sync_lichess("owner", fetch=fetch) == 0  # same game again: no duplicate
    assert "since=0" not in seen[1]
    assert col.games["https://lichess.org/abcd1234"].result == "loss"


def test_import_needs_the_owner_and_skips_duplicates(tmp_path):
    games = tmp_path / "bot.pgn"
    games.write_text(
        pgn("Om Rode", "chess-rl", "1. e4 e5 2. Nf3 Nc6", "0-1", tc="300+0")
        + "\n"
        + pgn("someone", "else", "1. d4 d5")  # owner not playing: ignored
        + "\n"
        + pgn("chess-rl", "Om Rode", "1. e4 e5")
    )
    col = Collection.open(tmp_path / "col")
    col.aliases.add("om rode")
    assert col.import_files([games]) == 2
    assert col.import_files([games]) == 0
    report = col.report()
    assert report["games"] == 2 and report["by_source"] == {"import": 2}
    assert report["owner_moves"] == 2 + 1  # 4 plies as White, 2 plies as Black


def test_collect_cli(tmp_path, capsys):
    games = tmp_path / "bot.pgn"
    games.write_text(pgn("Om Rode", "chess-rl", "1. e4 e5", "0-1", tc="300+0"))
    root = str(tmp_path / "col")
    assert main(["collect", "--root", root, "import", str(games)]) == 2  # no alias known yet
    assert main(["collect", "--root", root, "import", str(games), "--alias", "Om Rode"]) == 0
    assert main(["collect", "--root", root, "report", "--json"]) == 0
    out = capsys.readouterr().out
    assert json.loads(out[out.index("{") :])["by_time_class"] == {"blitz": 1}
