import json

import pytest

from chess_rl.cli import main
from chess_rl.runlog import read_runs


def test_help_lists_subcommands(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "info" in out and "baseline" in out


def test_info(capsys):
    assert main(["info"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["device"] in {"cpu", "cuda"} and info["versions"]["chess_rl"]


def test_baseline_eval_random_records_run(tmp_path, capsys):
    from tests.test_training_smoke import _model

    model_path = tmp_path / "m"
    _model().save(model_path)
    out_json = tmp_path / "eval.json"
    code = main(
        ["baseline", "eval-random", "--model", str(model_path), "--games", "2", "--max-plies", "20"]
        + ["--json-out", str(out_json)]
    )
    assert code == 0
    results = json.loads(out_json.read_text())
    assert results["agent_vs_random"]["games"] == 2
    (rec,) = read_runs()
    assert rec["kind"] == "baseline-eval-random" and rec["status"] == "ok"
    assert rec["outputs"]["agent_vs_random"]["games"] == 2


@pytest.mark.slow
def test_baseline_train_via_cli(tmp_path):
    code = main(
        ["baseline", "train", "--timesteps", "300", "--num-envs", "1", "--no-curriculum"]
        + ["--model-dir", str(tmp_path / "m"), "--log-dir", str(tmp_path / "l"), "--set", "seed=1"]
    )
    assert code == 0
    assert (tmp_path / "m" / "chess_tactical_final.zip").exists()
    (rec,) = read_runs()
    assert rec["kind"] == "baseline-train" and rec["seed"] == 1 and rec["config"]["use_curriculum"] is False
