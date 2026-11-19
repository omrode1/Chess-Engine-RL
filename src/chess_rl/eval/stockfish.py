"""Pinned Stockfish: install a verified official release and locate it.

Strength targets (M4–M8) are measured against Stockfish's UCI_Elo limiter, so the engine version is
pinned. Every machine and CI use the same official binary, checked against its SHA-256.

    uv run chess-rl stockfish install     # ~80 MB download, no sudo
    uv run chess-rl stockfish path

Lookup order: $STOCKFISH_PATH, then the pinned install, then `stockfish` on PATH (a warning is
logged if its version differs from the pin).
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

VERSION = "sf_19"
VERSION_NAME = "Stockfish 19"
URL = f"https://github.com/official-stockfish/Stockfish/releases/download/{VERSION}/stockfish-linux-x86-64-universal.tar.gz"
SHA256 = "9defc0d4e55d49c65a6d042f3e571a39fcea499ade6dbe741b53b8c65e03611f"
_MEMBER = "stockfish/stockfish-linux-x86-64-universal"


class StockfishNotFound(RuntimeError):
    pass


def data_home() -> Path:
    return Path(os.environ.get("CHESS_RL_HOME", Path.home() / ".local" / "share" / "chess-rl"))


def pinned_path() -> Path:
    return data_home() / "stockfish" / VERSION / "stockfish"


def engine_name(path: str | Path) -> str | None:
    """'Stockfish 19' etc. from the engine's UCI `id name` line, or None if it doesn't answer."""
    try:
        out = subprocess.run([str(path)], input="uci\nquit\n", capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in out.splitlines():
        if line.startswith("id name "):
            return line.removeprefix("id name ").strip()
    return None


def find_stockfish() -> Path:
    env = os.environ.get("STOCKFISH_PATH")
    if env:
        if not Path(env).is_file():
            raise StockfishNotFound(f"STOCKFISH_PATH={env} does not exist")
        return Path(env)
    if pinned_path().is_file():
        return pinned_path()
    on_path = shutil.which("stockfish") or (shutil.which("stockfish", path="/usr/games"))
    if on_path:
        name = engine_name(on_path)
        if name != VERSION_NAME:
            log.warning(
                "using %s at %s; results are pinned to %s (run `chess-rl stockfish install`)",
                name,
                on_path,
                VERSION_NAME,
            )
        return Path(on_path)
    raise StockfishNotFound("Stockfish not found. Run `uv run chess-rl stockfish install` or set STOCKFISH_PATH.")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def install(force: bool = False) -> Path:
    """Download the pinned release, verify its SHA-256, and install the binary. Idempotent."""
    target = pinned_path()
    if target.is_file() and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "stockfish.tar.gz"
        log.info("downloading %s", URL)
        urllib.request.urlretrieve(URL, archive)
        digest = _sha256(archive)
        if digest != SHA256:
            raise RuntimeError(f"checksum mismatch for {URL}: got {digest}, expected {SHA256}")
        with tarfile.open(archive) as tar:
            member = tar.getmember(_MEMBER)
            src = tar.extractfile(member)
            if src is None:
                raise RuntimeError(f"{_MEMBER} missing from archive")
            partial = target.with_suffix(".partial")
            with src, partial.open("wb") as out:
                shutil.copyfileobj(src, out)
    partial.chmod(partial.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    partial.replace(target)
    name = engine_name(target)
    if name != VERSION_NAME:
        raise RuntimeError(f"installed binary reports {name!r}, expected {VERSION_NAME!r}")
    return target
