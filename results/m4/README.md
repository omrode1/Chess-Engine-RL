# M4 results: supervised pretraining (raw network, no search)

**Exit criterion (#23): the raw network scores ≥50% vs Stockfish 19 @ UCI_Elo 1320. Result: 85.5%, Elo +308 [+228, +429]. ✅**

## Model and data

| | |
|---|---|
| Network | ResNet 10 blocks × 128 filters, squeeze-excitation, 3.60M parameters (`configs/train/m4.yaml`) |
| Data | 29.4M train / 595k val positions from 399,787 Lichess games (Oct 2026, both players ≥1800, blitz/rapid/classical) |
| Objective | cross-entropy on the move played + cross-entropy on the game result (win/draw/loss) |
| Training | 3 epochs, 86,148 steps × 1024, AdamW, warmup + cosine decay, fp16 |
| Checkpoint | `m4-final.pt` (release v0.2.0), sha256 `685248598408d7ea…` |

## Compute (from `results/runs.jsonl`)

| Stage | Hardware | Wall time |
|---|---|---|
| Data build (stream 0.80 GB of the Lichess dump) | 12 CPU processes | 8.8 min |
| Supervised training | RTX 3050 6 GB laptop GPU | **3.34 GPU-hours** |

## Learning curve (validation: 200k held-out positions; `history.jsonl`)

| Step | Positions seen | Policy loss | Top-1 | Top-5 | Value loss | Calibration error |
|---|---|---|---|---|---|---|
| 5,000 | 5.1M | 1.765 | 45.0% | 84.4% | 0.865 | 0.064 |
| 10,000 | 10.2M | 1.621 | 48.3% | 86.9% | 0.845 | 0.032 |
| 20,000 | 20.5M | 1.543 | 50.1% | 88.5% | 0.832 | 0.009 |
| 30,000 | 30.7M | 1.498 | 51.3% | 89.3% | 0.832 | 0.035 |
| 45,000 | 46.1M | 1.460 | 52.1% | 89.8% | 0.824 | 0.016 |
| 60,000 | 61.4M | 1.432 | 52.9% | 90.3% | 0.822 | 0.020 |
| 75,000 | 76.8M | 1.420 | 53.2% | 90.5% | 0.821 | 0.017 |
| 86,148 | 88.2M | 1.417 | **53.3%** | **90.5%** | 0.817 | 0.011 |

Top-1 = how often the network's first choice is the move the ≥1800 player actually played.
Calibration error = mean |predicted − actual score| over 10 bins of predicted score.

## Strength (100 games each, paired openings, `results/elo.csv`)

The raw policy network plays argmax over legal moves: one network call per move, no lookahead.

| vs | W / D / L | Score | Elo [95% CI] |
|---|---|---|---|
| v0 DQN baseline (v0.1.0) | 98 / 2 / 0 | 99.0% | +798 [+646, +∞] |
| greedy (one-ply material) | 99 / 1 / 0 | 99.5% | +920 [+730, +∞] |
| **Stockfish 19 @ UCI_Elo 1320** | **84 / 3 / 13** | **85.5%** | **+308 [+228, +429]** |
| Stockfish 19 @ UCI_Elo 1500 | 44 / 18 / 38 | 53.0% | +21 [−41, +84] |
| Stockfish 19 @ UCI_Elo 1700 | 26 / 25 / 49 | 38.5% | −81 [−145, −23] |

Interpolated, the raw network plays at roughly **Stockfish UCI_Elo ≈ 1550** (0.1 s per Stockfish move).

## Tactics (Win At Chess, 300 positions, `results/tactics.csv`)

| Player | Solved |
|---|---|
| v0 baseline | 8 (2.7%) |
| greedy | 41 (13.7%) |
| **M4 raw network** | **50 (16.7%)** |
| Stockfish @1320 | 71 (23.7%) |
| Stockfish full strength | 266 (88.7%) |

The network plays human-like positional moves well but, with no lookahead, misses concrete tactics.
That is what search (M5) is for, and issue #1 ("hangs pieces") stays open until then.

## Interim check

During training, the step-20k checkpoint (23% of training) already scored 13/3/4 vs Stockfish @1320 in a
20-game check (not recorded in `elo.csv`).

## Files

- `history.jsonl`: every validation
- `summary.json`: final summary
- `train.log`: full training log
