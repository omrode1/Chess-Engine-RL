# Chess-Engine-RL

**A chess engine trained on a laptop GPU, built to beat one specific person: me.**

> **Status (Nov 2026): M5 done.** Network + search plays far above the 2025 baseline (see [Current engine](#current-engine-m5-release-v030)); next: modelling the opponent (M6). The original 2025 DQN agent is preserved as baseline [v0.1.0](https://github.com/omrode1/Chess-Engine-RL/releases/tag/v0.1.0). It scores **49% against a random-move player**, so it has no measurable chess skill. Why it couldn't learn: [`docs/postmortem-v0.md`](docs/postmortem-v0.md).

## The idea

Most engines play the objectively best move and assume a perfect opponent. This project builds an engine that wins against **one particular human** by modelling them:

1. **Learn chess from human games**: a policy/value network pretrained on Lichess games, sized for an RTX 3050 (6 GB).
2. **Search**: MCTS on top of the network, so it stops hanging pieces.
3. **Model the opponent ("Om-sim")**: a few-shot model of my own moves and mistakes, built from my games.
4. **Exploit the model**: opponent-aware search and RL against the simulated opponent.

Headline question: **how much less compute does an engine need to beat a specific human when it models that human?** Every run's compute is logged so the answer can be measured.

Roadmap and progress: [milestones](https://github.com/omrode1/Chess-Engine-RL/milestones).

## Install

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). GPU training needs an NVIDIA driver supporting CUDA 13.0+.

```bash
git clone https://github.com/omrode1/Chess-Engine-RL.git
cd Chess-Engine-RL
uv sync --extra cu130   # NVIDIA GPU
uv sync --extra cpu     # or CPU only

uv run chess-rl stockfish install   # pinned Stockfish 19 for evaluation (~80 MB, no sudo)
uv run chess-rl info                # versions, hardware, device, Stockfish
```

## Usage

Each run appends a record to `results/runs.jsonl`: git commit, config, seed, hardware, wall-clock time, and GPU-hours.

The v0 DQN pipeline was removed in M5 (#27). Its code and model live on in git history and release [v0.1.0](https://github.com/omrode1/Chess-Engine-RL/releases/tag/v0.1.0); its results stay in the tables below.

## Current engine (M5, release v0.3.0)

The M4 network plus Monte Carlo tree search. With 1,600 search nodes per move (about 0.4 s on the laptop GPU) it scores
**97.5% against Stockfish 19 @ UCI_Elo 1500 and 60% @ 2700**, and solves 48% of Win At Chess (the raw network: 16.7%).
Full report, including what these Stockfish levels do and don't mean: [`results/m5/`](results/m5/README.md).

```bash
uv run chess-rl uci --player "mcts:path=runs/m4/m4-final.pt,book=true"   # play it in a chess GUI (uses the GUI's clock)
uv run chess-rl watch live "mcts:path=runs/m4/m4-final.pt,nodes=1600" "stockfish:elo=2300,movetime=0.1"
uv run chess-rl match "mcts:path=runs/m4/m4-final.pt" "stockfish:elo=1500,movetime=0.1" --tc 3+2 --games 20
```

Get `m4-final.pt` from release [v0.2.0](https://github.com/omrode1/Chess-Engine-RL/releases/tag/v0.2.0) (the network) into `runs/m4/`.

## Modelling the opponent (M6, release v0.4.0)

A second network predicts the moves *people at the owner's level* play, and which of them are blunders.
It is conditioned on both players' ratings and the clock, and trained on 30M positions from 600–1600 Lichess games.
On unseen games it matches **50.8% of human moves vs 48.4% for Maia-2** (blunder AUC 0.82). Fitted to
the owner's 10 games (Om-sim), it does not yet beat the rating-calibrated base model, which needs more games. Full report:
[`results/m6/`](results/m6/README.md).

```bash
uv run chess-rl collect sync --chesscom <user>        # the owner's games, stored locally only
uv run chess-rl omsim label && uv run chess-rl omsim eval
uv run chess-rl omsim report                          # HTML weakness report from real games
```

## Train the network (M4)

```bash
uv run chess-rl data build --config configs/data/strength.yaml   # ~9 min: 30M positions from Lichess (both >= 1800)
uv run chess-rl train sl --config configs/train/m4.yaml           # ~3.5 h on an RTX 3050; --resume runs/m4/latest.pt
# progress lines go to stderr; validation history to runs/m4/history.jsonl
```

The data is streamed from the [Lichess open database](https://database.lichess.org/) (CC0) and stored compactly (~117 B/position). Training writes `best.pt`/`latest.pt` checkpoints, validation history, and a compute record in `results/runs.jsonl`.

## Evaluation

```bash
uv run chess-rl match greedy "stockfish:elo=1500,movetime=0.1" --games 100 --workers 8
```

Games come in colour-swapped pairs from opening-book starts and run in parallel. Each match prints one line per game, records Elo with a 95% confidence interval in [`results/elo.csv`](results/elo.csv), and saves every game as PGN. Player specs: `random`, `greedy`, `stockfish:elo=…,movetime=…`, `net:path=…`, `mcts:path=…`.

Reference ladder (100 games each; Elo of the first player relative to the second):

| Match | W / D / L | Elo [95% CI] |
|---|---|---|
| greedy vs random | 89 / 11 / 0 | +494 [+411, +641] |
| v0 baseline vs random | 6 / 86 / 8 | −7 [−32, +19] |
| random vs Stockfish 19 @ UCI_Elo 1320 | 0 / 1 / 99 | −920 [−∞, −730] |
| greedy vs Stockfish 19 @ UCI_Elo 1320 | 0 / 2 / 98 | −798 [−∞, −646] |

**M4 result:** the raw network (one call per move, no search) scores **85.5% vs Stockfish @ 1320** and about even with Stockfish @ 1500 (≈1550 on that scale), after 3.3 GPU-hours on the laptop. Full report: [`results/m4/`](results/m4/README.md).

| Raw M4 network vs | W / D / L | Elo [95% CI] |
|---|---|---|
| v0 baseline | 98 / 2 / 0 | +798 [+646, +∞] |
| Stockfish 19 @ 1320 | 84 / 3 / 13 | +308 [+228, +429] |
| Stockfish 19 @ 1500 | 44 / 18 / 38 | +21 [−41, +84] |
| Stockfish 19 @ 1700 | 26 / 25 / 49 | −81 [−145, −23] |

Tactics: the [Win At Chess](src/chess_rl/eval/suites/NOTICE.md) suite (300 positions, one move each) and mate-in-1 / mate-in-2 suites from Lichess puzzles, recorded in [`results/tactics.csv`](results/tactics.csv):

```bash
uv run chess-rl tactics run --player greedy --suite wac   # or mate1, mate2, all
```

| Player | WAC solved |
|---|---|
| random | 14 / 300 (4.7%) |
| greedy (one-ply material) | 41 / 300 (13.7%) |
| v0 baseline | 8 / 300 (2.7%) |
| M4 raw network (no search) | 50 / 300 (16.7%) |
| **M5 network + MCTS, 1,600 nodes** | **145 / 300 (48.3%)** |
| Stockfish 19 @ UCI_Elo 1320, 0.1 s/move | 71 / 300 (23.7%) |
| Stockfish 19 full strength, 0.1 s/move | 266 / 300 (88.7%) |

## Watch games in the terminal

```bash
uv run chess-rl watch list runs/matches/<file>.pgn          # one line per game
uv run chess-rl watch replay runs/matches/<file>.pgn --game 3
uv run chess-rl watch live greedy "stockfish:elo=1320,movetime=0.3"
```

## Play it in a chess GUI (UCI)

Any player can run as a UCI engine:

```bash
uv run chess-rl uci --player greedy
```

To add it to Cute Chess, Arena, or En Croissant, use **command** `<repo>/.venv/bin/chess-rl` with **arguments** `uci --player greedy`. Strength can be changed from the GUI with the `Player` option (any player spec, e.g. `stockfish:elo=1500,movetime=0.2` or `mcts:path=runs/m4/m4-final.pt`). The `mcts` engine manages its own time from the GUI's clock.

## Layout

```
src/chess_rl/      the engine package: board/move encoding, opening book, curriculum positions,
                   network, training, MCTS search, evaluation, UCI, CLI
configs/           YAML run configs
scripts/           one-off data scripts (tactics suite generation)
results/           tracked run records and evaluation results
docs/              post-mortem, reports
tests/             pytest suite (CI on every PR)
```

## Development

```bash
uvx pre-commit install   # ruff lint + format on every commit
uv run pytest            # tests (CPU, ~1 min)
```

All work goes through issue → branch → PR → CI. `main` is protected.

## License

[MIT](LICENSE)

## Acknowledgments

[python-chess](https://python-chess.readthedocs.io/), [PyTorch](https://pytorch.org/), [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (v0 baseline, until v0.2.0), and the [Lichess open database](https://database.lichess.org/).
