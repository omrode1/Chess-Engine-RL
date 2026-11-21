# M6 results: modelling the opponent (Om-sim)

M6 builds the opponent model at the heart of the project: a network that predicts the moves people at
the owner's level actually play, including their blunders (#35); a version fitted to the owner's own games
(Om-sim, #36); and a weakness report built from it (#37). Only the owner's games are used. They stay on
this machine (`data/personal/`, git-ignored); only aggregates are committed.

| Issue | Exit criterion | Result |
|---|---|---|
| #34 collector | re-running adds no duplicates; counts reported | ✅ 10 games, 231 of the owner's moves |
| #35 human model | move-match within a few points of Maia-2 at 1100–1300 | ✅ **ahead of Maia-2 by 2.4 points** (2.9 at 1000–1200) |
| #36 Om-sim | beats both baselines on held-out moves, with a CI | ❌ **not met**: 10 games are too few (see below) |
| #37 weakness report | generated from the CLI, checked against real games | ✅ |

## The human move model (#35)

| | |
|---|---|
| Network | the M4 ResNet tower (10×128, SE) with FiLM conditioning on both players' ratings, the mover's clock and a player embedding; policy, game-result and blunder heads (3.8M parameters) |
| Data | 30M positions from 484k Lichess games, October 2026, both players rated 600–1600, blitz/rapid/classical; clocks for every position; blunder labels for 12.9% (games Lichess had analysed) |
| Initialisation | the M4 network (release v0.2.0) |
| Training | 2 epochs (57,408 steps of 1,024), 2.0 GPU-hours for the final segment (see *Compute*); the evaluated and released checkpoint is step 55,000 (best validation loss) |

A *blunder* is a move that loses at least 10 win-percentage points for the player who makes it (Lichess's
win% formula; Lichess itself calls 10 a mistake and 15 a blunder). 13.6% of labelled moves in this band
are blunders.

### Against Maia-2 on unseen games

The test set is every position of the first 3,000 games in the band from **September 2026**. Neither our
model nor Maia-2 trained on that month. Both models score the same 188,791 positions; Maia-2 uses its
blitz model on blitz games and its rapid model otherwise. Intervals are 95% bootstrap over games.

| Model | Move-match (top-1) | 95% CI | NLL |
|---|---|---|---|
| **Human model (ours)** | **50.8%** | 50.5–51.1 | **1.538** |
| Maia-2 (Tang et al., NeurIPS 2024) | 48.4% | 48.1–48.7 | 1.616* |
| M4 strength network (trained on ≥1800 players) | 48.3% | 48.0–48.6 | 1.696 |

Paired difference, ours − Maia-2: **+2.4 points [+2.2, +2.5]** overall and **+2.9 [+2.5, +3.3]** for
movers rated 1000–1200.

| Mover's rating | <800 | 800–1000 | 1000–1200 | 1200–1400 | 1400–1600 |
|---|---|---|---|---|---|
| Ours | 44.2% | 48.5% | 50.3% | 51.3% | 51.9% |
| Maia-2 | 40.5% | 45.4% | 47.5% | 49.1% | 49.9% |
| M4 | 38.7% | 44.2% | 46.9% | 49.0% | 50.4% |

\* Maia-2's package rounds probabilities to 4 decimals, which slightly inflates its NLL.

**Read this carefully.** Our model was trained on games from one month before the test month, in exactly
this rating band; Maia-2 was trained on older Lichess data across all ratings and is a general model. Some
of the gap is likely recency and specialisation (openings and rating pools drift), not a better method.
The fair claim: *for 600–1600 Lichess games in late 2026, this model predicts moves better than Maia-2.*
Note also how a strong network (M4) is a poor predictor of weak players' moves, and worst at the bottom
of the range: the gap between "good move" and "human move" is what M7 will exploit.

**Blunder head:** ROC AUC **0.820** on the 24,535 labelled test positions (base rate 13.6%). Training
curve on the October validation set:

| Step | 5k | 15k | 25k | 35k | 45k | 57.4k |
|---|---|---|---|---|---|---|
| Move-match | 49.4% | 50.0% | 50.3% | 50.7% | 50.9% | 51.1% |
| Blunder AUC | 0.796 | 0.806 | 0.814 | 0.818 | 0.824 | 0.826 |

## Om-sim on the owner's games (#36)

The owner's collection holds **10 games (231 moves)** from chess.com (9) and one game against this
engine. Blunder labels come from Stockfish 19 at depth 16. Three methods, never scored on games they were
fitted on:

- **base@raw**: the human model at the chess.com ratings as given
- **base@offset**: the same model with a chess.com→Lichess rating offset fitted by maximum likelihood on
  the training games. The fitted offset is **+700** (games rated 300–600 on chess.com look like
  1000–1300 on Lichess to the model)
- **Om-sim**: base@offset plus a 64-number player embedding trained on the training games (everything
  else frozen, as in Maia4All)

Maia-2 is scored at the same +700 calibration.

| Method | Forward chaining: 5 games, 131 moves | Time-ordered split: 2 games, 49 moves |
|---|---|---|
| base@raw | 50.4% [42.5, 56.0] | 44.9% [37.5, 52.0] |
| **base@offset** | **53.4%** [45.6, 58.6] | **53.1%** [45.8, 60.0] |
| Om-sim | 50.4% [40.8, 56.1] | 46.9% [37.5, 56.0] |
| Maia-2 @ +700 | 46.6% [36.7, 51.7] | 51.0% [45.8, 56.0] |

Forward chaining fits on games 0..k−1 and predicts game k, for k = 5..9. The time-ordered split holds out
the 2 most recent games.

**The exit criterion is not met.** Om-sim is 3.1 points *behind* base@offset under forward chaining
(paired, [−5.7, −0.9]), though its likelihood is marginally better (NLL 1.477 vs 1.480). With 5 test games
and 131 moves, a game-level bootstrap over so few games understates the uncertainty. These numbers cannot
separate a few points either way. The honest conclusions:

1. **Rating calibration matters** (+3 to +8 points over raw ratings): the population model at the right
   rating is the best predictor of the owner so far, ahead of Maia-2.
2. **Personalisation needs more games.** Maia4All reports gains from about 20 games; the owner has 10, and
   the earliest forward-chaining folds train on 5. The pipeline is built and tested. It needs data:
   `chess-rl collect sync` then `chess-rl omsim eval` re-runs everything.
3. **The owner's blunders are predictable.** The base model's blunder AUC on the owner's held-out moves is
   0.86 (forward chaining) and 0.95 (split), above its 0.82 on the population test set.

## Weakness report (#37)

`chess-rl omsim report` classifies each blunder from Stockfish's lines and renders an HTML report with
boards from the real games (kept local: it contains the games). On the owner's 10 games: 34 blunders,
14.7% of moves (the 600–1600 Lichess rate is 13.6%, from deeper Lichess analysis, so roughly comparable).

| How the blunder happened | Count |
|---|---|
| positional (no material changes hands; often a slip in an already-winning position) | 10 |
| hung the piece that just moved | 7 |
| ignored a threat that was already there | 6 |
| missed a win of material | 5 |
| allowed a fork | 3 |
| allowed a combination (material lost within 8 plies) | 2 |
| allowed mate | 1 |

None came in time trouble. Om-sim (fitted on all 10 games) predicts a 29% blunder risk on the moves that
were blunders vs 8.5% on the rest (in-sample).

The first version of the classifier (single-reply rules) left 14 blunders unexplained. Checking them
against the games showed most lost material two or three moves later, which is why the categories now
follow Stockfish's 8-ply lines.

## Compute (from `results/runs.jsonl`)

| Run | Wall time | GPU-hours |
|---|---|---|
| Data build: 30M positions, 0.52 GB of the October dump streamed | 15.2 min | — |
| Human model, final segment (steps 12,000 → 57,408) | 2.0 h | 1.99 |
| Human model, two segments lost to laptop sleep (reconstructed records) | — | 2.29 |
| Test set (3,000 games) and all evaluations | < 5 min | 0.05 |

The two `killed` segments: the laptop suspended twice during training (idle sleep at 16:18 and overnight),
and the CUDA context did not survive resume (#78). The useful steps in them were steps 0–12,000 (kept via
the checkpoint); the GPU-hours above count wall time while awake, including a hang. Training now runs under
`systemd-inhibit` with a stall watchdog. The M4 network the model starts from cost 3.34 GPU-hours (M4).

## Reproduce

```bash
uv run chess-rl data build --config configs/data/human-band.yaml
uv run chess-rl train human --config configs/train/human.yaml
uv run chess-rl human testset --month 2026-09 --games 3000
uv run chess-rl human eval --test-set data/lichess/human-test-2026-09.jsonl \
    --model human=runs/human/best.pt --model m4=runs/m4/m4-final.pt --preds maia2=runs/human-eval/maia2.jsonl
# Maia-2, from a separate environment with the maia2 package (see the script's docstring):
python scripts/maia2_baseline.py data/lichess/human-test-2026-09.jsonl runs/human-eval/maia2.jsonl

# the owner's games (opt-in, local only)
uv run chess-rl collect sync --chesscom omrode31
uv run chess-rl omsim label && uv run chess-rl omsim eval && uv run chess-rl omsim report
```

Raw numbers: `results/m6/human-model.json`, `results/m6/omsim.json`. The human model is attached to release
v0.4.0. Om-sim (the owner's personal model) is not published.
