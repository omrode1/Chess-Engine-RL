# Chess-Engine-RL

**A chess engine trained on a laptop GPU, built to beat one specific person: me.**

> **Status (Nov 2026): being revived and redesigned.** The original 2025 DQN agent is preserved as baseline [v0.1.0](https://github.com/omrode1/Chess-Engine-RL/releases/tag/v0.1.0). It scores **49% against a random-move player**, so it has no measurable chess skill. Why it couldn't learn: [`docs/postmortem-v0.md`](docs/postmortem-v0.md).

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

uv run chess-rl info    # versions, hardware, device
```

## Usage

```bash
# v0 DQN baseline (frozen for comparison)
uv run chess-rl baseline train --config configs/baseline.yaml [--set timesteps=50000]
uv run chess-rl baseline eval-random --model runs/baseline/models/chess_tactical_final --games 100
```

Each run appends a record to `results/runs.jsonl`: git commit, config, seed, hardware, wall-clock time, and GPU-hours. New subcommands arrive with each milestone: data, train, match, uci, play.

## Layout

```
src/chess_rl/      the engine package: config, run logging, CLI (encoding, model, search, ... to come)
baseline/          frozen v0 DQN pipeline (removed in M5)
utils/             opening book (moving into chess_rl in M2)
configs/           YAML run configs
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

[python-chess](https://python-chess.readthedocs.io/), [PyTorch](https://pytorch.org/), [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (v0 baseline), and the [Lichess open database](https://database.lichess.org/).
