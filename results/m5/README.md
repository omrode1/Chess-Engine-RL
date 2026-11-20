# M5 results: the network plus tree search

**Exit criterion (#26): the engine with search scores ≥55% vs Stockfish 19 @ UCI_Elo 1500. Result: 97.5%, Elo +636 [+497, +∞]. ✅**

## The engine

| | |
|---|---|
| Network | the M4 network, unchanged (10×128 SE ResNet, 3.6M parameters; release v0.2.0) |
| Search | MCTS with PUCT, batched GPU leaf evaluation (virtual loss), WDL values, exact terminals, mate solver, tree reuse, transposition cache (`chess_rl.search`) |
| Budget | 1,600 search nodes per move: about **0.4 s per move** on the RTX 3050 laptop (≈3,500 nodes/s, single game) |
| Extra training | none: all of the gain comes from search |

## Strength ladder (100 games each, paired openings, `results/elo.csv`)

Stockfish plays at a fixed 0.1 s per move, as in every earlier result. The engine searches 1,600 nodes per move.

| vs Stockfish 19 | W / D / L | Score | Elo [95% CI] |
|---|---|---|---|
| UCI_Elo 1500 | 97 / 1 / 2 | **97.5%** | +636 [+497, +∞] |
| UCI_Elo 1900 | 98 / 2 / 0 | 99.0% | +798 [+646, +∞] |
| UCI_Elo 2300 | 84 / 12 / 4 | 90.0% | +382 [+304, +504] |
| UCI_Elo 2700 | 34 / 52 / 14 | 60.0% | +70 [+24, +119] |
| UCI_Elo 3190 (maximum) | 0 / 34 / 66 | 17.0% | −275 [−340, −224] |
| **UCI_Elo 2700, equal time (0.4 s/move)** | 38 / 41 / 21 | 58.5% | +60 [+8, +114] |

The engine thinks about 4× longer per move than Stockfish's 0.1 s. Giving Stockfish the same 0.4 s did not change the
result at UCI_Elo 2700 (60.0% → 58.5%), so the lead is not a thinking-time artefact.

**Reading these numbers.** Stockfish's `UCI_Elo` is calibrated against engine rating lists at much longer time controls,
and its strength limiter plays deliberately imperfect moves. These ladders show where the engine stands *relative to
Stockfish's limiter*. They are not a human or FIDE rating, and the scale is not linear (1900 scored higher than 1500, within
noise; 2700 → 3190 is a big jump). The comparison that matters for this project, the engine against the owner, comes
in M6–M8.

### Search vs no search (same network)

| vs Stockfish 19 | Raw network (M4) | Network + MCTS 1600 (M5) |
|---|---|---|
| UCI_Elo 1500 | 53.0% | 97.5% |
| UCI_Elo 1700 | 38.5% | (not run; 1900 → 99.0%) |
| Approximate level on this scale | ≈1550 | ≈2700–2800 |

## Tactics (`results/tactics.csv`)

| Suite | Raw network | MCTS 800 nodes | MCTS 1600 nodes | Stockfish @1320 |
|---|---|---|---|---|
| mate in 1 (150, new) | 109 (72.7%) | 150 (100%) | **150 (100%)** | |
| mate in 2 (150, new) | 79 (52.7%) | 131 (87.3%) | **133 (88.7%)** | |
| Win At Chess (300) | 50 (16.7%) | 126 (42.0%)* | **145 (48.3%)** | 71 (23.7%) |

\* measured while building #24, not recorded in `tactics.csv`.

The mate suites come from the Lichess puzzle database (CC0); their solution sets are computed exhaustively
(`scripts/make_mate_suites.py`). WAC alone has no mate-in-1 positions.

## Clocks and time management (#25)

100 games at **1 minute + 1 second** per move, time-managed engine vs Stockfish @1500 (0.1 s/move): **+99 =1 −0, no time
losses**. The lowest engine clock in any game was 34.1 s of 60.

## Remaining weakness

Both losses to Stockfish @1500 came from a single tactical blunder each (Stockfish depth 14 on every engine move):

- round 72: 19…b5 (+2.4 → −6.0 pawns)
- round 81: 49. Rd6 throws away a winning ending (+5.0 → −2.5)

At 1,600 nodes the search still misses some tactics (WAC 48%). More nodes, a stronger network (M7) or both should help.

## Compute (from `results/runs.jsonl`)

| Run | Wall time |
|---|---|
| 1+1 clock match (100 games) | 13.8 min |
| Ladder, 6 matches × 100 games, 5 games in parallel | 77.5 min |
| Tactics, MCTS 1600 (all three suites) | 3.8 min |

No training: M5 reuses the M4 network (3.34 GPU-hours).

## Reproduce

```bash
uv run chess-rl match "mcts:path=runs/m4/m4-final.pt,nodes=1600" "stockfish:elo=2300,movetime=0.1" --games 100 --workers 5
uv run chess-rl match "mcts:path=runs/m4/m4-final.pt" "stockfish:elo=1500,movetime=0.1" --tc 1+1 --games 100 --workers 5
uv run chess-rl tactics run --player "mcts:path=runs/m4/m4-final.pt,nodes=1600" --suite all
```

`m4-final.pt` is the network from release v0.2.0. The games are attached to release v0.3.0.
