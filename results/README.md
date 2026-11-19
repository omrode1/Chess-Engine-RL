# Results

Tracked experiment records.

- `runs.jsonl`: one JSON object per training/eval run, written by `chess_rl.runlog.track_run` (git commit, config, seed, hardware, wall-clock, GPU-hours, outputs). Never edited by hand.
- `elo.csv`: one row per `chess-rl match` (players, W/D/L, score, Elo with 95% CI, LOS, settings, git commit).
- `tactics.csv`: one row per `chess-rl tactics run` (player, suite, solved/total, git commit).
