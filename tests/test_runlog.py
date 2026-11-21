import json
import socket

import pytest

from chess_rl.config import DataBuildConfig
from chess_rl.runlog import read_runs, track_run


def test_successful_run_is_recorded(_runs_file):
    with track_run("unit", DataBuildConfig(workers=3), seed=5) as run:
        run.outputs["score"] = 0.5
    (rec,) = read_runs()
    assert rec["kind"] == "unit" and rec["status"] == "ok" and rec["seed"] == 5
    assert rec["config"]["workers"] == 3 and rec["outputs"] == {"score": 0.5}
    assert rec["gpu_hours"] == 0.0 and rec["wall_seconds"] >= 0
    assert rec["git"]["sha"] and len(rec["git"]["sha"]) == 40
    assert {"cpu_count", "ram_gb", "gpu"} <= rec["hardware"].keys()
    assert rec["versions"]["chess_rl"]


def test_failed_run_is_recorded_and_reraised(_runs_file):
    with pytest.raises(RuntimeError), track_run("unit", {"a": 1}):
        raise RuntimeError("x")
    assert read_runs()[-1]["status"] == "failed"


def test_interrupted_run_is_recorded(_runs_file):
    with pytest.raises(KeyboardInterrupt), track_run("unit", {}):
        raise KeyboardInterrupt
    assert read_runs()[-1]["status"] == "interrupted"


def test_cuda_runs_count_gpu_hours(_runs_file):
    with track_run("unit", {}, device="cuda"):
        pass
    assert read_runs()[-1]["gpu_hours"] >= 0 and read_runs()[-1]["device"] == "cuda"


def test_record_has_no_hostname(_runs_file):
    with track_run("unit", {}):
        pass
    assert socket.gethostname() not in _runs_file.read_text()
    json.loads(_runs_file.read_text().splitlines()[-1])  # one valid JSON object per line


def test_results_dir_does_not_make_tree_dirty(tmp_path):
    import subprocess

    from chess_rl.system import git_info

    repo = tmp_path / "repo"
    (repo / "results").mkdir(parents=True)
    (repo / "code.py").write_text("x = 1\n")
    (repo / "results" / "runs.jsonl").write_text("{}\n")
    env = ["-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", *env, "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", *env, "-C", str(repo), "commit", "-qm", "init"], check=True)
    assert git_info(repo)["dirty"] is False
    (repo / "results" / "runs.jsonl").write_text("{}\n{}\n")  # a run appended a record
    assert git_info(repo)["dirty"] is False
    (repo / "code.py").write_text("x = 2\n")  # code changed
    assert git_info(repo)["dirty"] is True


def test_sigterm_is_recorded_as_terminated(tmp_path):
    import os
    import signal

    from chess_rl.runlog import Terminated

    runs = tmp_path / "runs.jsonl"
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(Terminated), track_run("t", {}, runs_file=runs):
        os.kill(os.getpid(), signal.SIGTERM)
    assert read_runs(runs)[0]["status"] == "terminated"
    assert signal.getsignal(signal.SIGTERM) is before  # the previous handler is restored
