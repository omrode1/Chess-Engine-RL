import json

import pytest

from chess_rl.cli import main


def test_help_lists_subcommands(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "info" in out and "match" in out and "baseline" not in out


def test_info(capsys):
    assert main(["info"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["device"] in {"cpu", "cuda"} and info["versions"]["chess_rl"]
