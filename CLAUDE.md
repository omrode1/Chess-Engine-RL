# CLAUDE.md

## Git and GitHub rules (always follow)

- Commit and push as the personal account **omrode1** (`omrode.34@gmail.com`). This is set in the repo-local git config. Never commit with another identity.
- **No AI attribution.** No `Co-Authored-By: Claude` trailer, no "Generated with Claude Code" line, no session links in commits, PR descriptions, issues, or release notes.
- Work on a branch, never directly on `main`. Open a PR that links its issue (`Closes #N`). Merge only after CI passes.
- Use Conventional Commits: `feat:`, `fix:`, `chore:`, `test:`, `docs:`, `refactor:`, `ci:`.
- Do not commit generated artifacts: `__pycache__/`, model checkpoints, TensorBoard logs, CSVs, plots, PGNs, or datasets. Model weights go to GitHub Release assets.

## Commands

- Install: `uv sync --extra cu130` (laptop GPU) or `uv sync --extra cpu`.
- CLI: `uv run chess-rl --help`. Package code lives in `src/chess_rl/`; frozen v0 code in `baseline/` (do not extend it).
- Tests: `env -u PYTHONPATH uv run pytest`. The owner's shell sources ROS 2 Jazzy, whose `PYTHONPATH` leaks ROS pytest plugins into the venv and breaks collection.
- Run `gh` with `-R omrode1/Chess-Engine-RL` from outside the repo (the sandboxed snap `gh` can't read `/etc/gitconfig`). Pass bodies on stdin (`--body-file -`); snap `gh` can't see `/tmp`.
- Stacked PRs: never `--delete-branch` when merging a PR that another PR uses as its base. Deleting the branch auto-closes the dependent PR.

## Project

- Goal: an engine built to beat one specific person (the owner, chess.com rapid ~1200) on laptop-class compute. It models him (Om-sim) and exploits that model in search and RL. The headline question: how much less compute does modelling the opponent need? The project is for a portfolio aimed at both technical and broad audiences.
- Roadmap: GitHub milestones M0–M8 and their issues. Only model players who opt in (the owner by default).
- v0 baseline: `baseline/train_tactical_agent.py` (canonical v0) and `baseline/train_agent.py` (older). Both are frozen and removed in M5 (#27).
- Keep the opening book (`utils/opening_book.py`) and the curriculum positions (`baseline/chess_environment.py`) as they are; #15 moves them into `chess_rl`.
- Every training/eval run goes through `chess_rl.runlog.track_run`, which appends to the tracked `results/runs.jsonl`. Tests redirect it with `CHESS_RL_RUNS_FILE`.
- Target hardware: an RTX 3050 laptop GPU (6 GB), 20 CPU cores, 15 GB RAM. Size defaults (batch size, number of environments, network size) to fit this machine.
