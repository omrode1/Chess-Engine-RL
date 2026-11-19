# CLAUDE.md

## Git and GitHub rules (always follow)

- Commit and push as the personal account **omrode1** (`omrode.34@gmail.com`). This is set in the repo-local git config. Never commit with another identity.
- **No AI attribution.** No `Co-Authored-By: Claude` trailer, no "Generated with Claude Code" line, no session links in commits, PR descriptions, issues, or release notes.
- Work on a branch, never directly on `main`. Open a PR that links its issue (`Closes #N`). Merge only after CI passes.
- Use Conventional Commits: `feat:`, `fix:`, `chore:`, `test:`, `docs:`, `refactor:`, `ci:`.
- Do not commit generated artifacts: `__pycache__/`, model checkpoints, TensorBoard logs, CSVs, plots, PGNs, or datasets. Model weights go to GitHub Release assets.

## Project

- Goal: an engine built to beat one specific person (the owner, chess.com rapid ~1200) on laptop-class compute. It models him (Om-sim) and exploits that model in search and RL. The headline question: how much less compute does modelling the opponent need? The project is for a portfolio aimed at both technical and broad audiences.
- Roadmap: GitHub milestones M0–M8 and their issues. Only model players who opt in (the owner by default).
- Canonical training pipeline: `train_tactical_agent.py`. `train_agent.py` is the older pipeline.
- Keep the opening book (`utils/opening_book.py`) and the curriculum positions (`src/chess_environment.py`).
- Target hardware: an RTX 3050 laptop GPU (6 GB), 20 CPU cores, 15 GB RAM. Size defaults (batch size, number of environments, network size) to fit this machine.
