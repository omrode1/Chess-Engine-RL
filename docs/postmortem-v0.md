# Post-mortem: why the 2025 DQN chess agent could not learn chess

**Scope:** the v0 pipeline as it stood on 1 April 2025 (commit [`deb8250`][v0]): DQN from Stable-Baselines3, a custom gym environment, hand-shaped rewards, and a 4-stage curriculum.
**Written:** November 2026, during the project revival. Every claim below links to the code, quotes the original training logs, or is reproduced by a test in `tests/`.

[v0]: https://github.com/omrode1/Chess-Engine-RL/tree/deb8250

## TL;DR

The agent never learned chess, and with that design it could not have. The main cause was not the bugs (there were many) but three design decisions:

1. **The action space had no stable meaning.** Action *k* meant "the k-th legal move in python-chess's generation order", which is a different move in every position.
2. **One network played both sides** without knowing whose turn it was, and the rewards never flipped sign between the two players.
3. **Heavily shaped rewards** paid the agent for things other than winning. It learned to farm them by making games longer.

The bugs (contradictory checkmate rewards, a capture detector that fired on every move, an inverted hanging-piece penalty, a curriculum that crashed after stage 1) made a doomed setup worse, and they also hid the design problems behind noisy metrics.

## What the logs showed

The original `training_output.log` (single-stage pipeline, ~1M steps) and `training_curriculum.log` were committed to the repo at the time. They were removed in the history rewrite (#2); the numbers below are copied from them.

**Single-stage run (`train_agent.py`).** The loss diverged by 12 orders of magnitude. Mean reward stayed flat.

| timesteps | exploration ε | mean episode length | mean episode reward | TD loss |
|---:|---:|---:|---:|---:|
| 37,206 | 0.86 | 340 | −234 | 4.35 |
| 182,016 | 0.31 | 364 | −239 | 2.3 × 10¹⁰ |
| 347,580 | 0.05 | 343 | −257 | 1.5 × 10¹³ |
| 702,018 | 0.05 | 360 | −232 | 8.3 × 10¹⁰ |
| 992,628 | 0.05 | 338 | −235 | 3.6 × 10⁹ |

The run then ended with `Training was interrupted: All arrays must be of the same length`. That was a logging bug (below), not a training failure, but the message hid the actual failure: a diverged Q-function.

**Curriculum run (`train_tactical_agent.py`, rewards "rebalanced to be more positive").** After the reward changes, mean episode reward flipped from about −235 to **+205**, while episodes stayed about 350 plies long. The agent earned more reward per game simply by playing more moves (see *Reward farming*).

| timesteps | mean episode length | mean episode reward | TD loss |
|---:|---:|---:|---:|
| 36,666 | 345 | +205 | 10.4 |
| 73,926 | 356 | +215 | 29.8 |

**Games.** The saved PGNs show the same pattern in every game: one side moves a single piece around while the other shuffles a rook back and forth.

```
1. d4 d5 2. Nf3 e6 3. Ne5 Nh6 4. Nd7 Be7 5. Nxb8 Rf8 6. Nc6 Rh8 7. Nxd8 Rf8 8. Nxb7 Rg8
9. Nd6+ Kd8 10. Nxc8 Rf8 11. Nxa7 Rg8 12. Nc6+ Ke8 13. Nxe7 Rf8 14. Nc8 Rg8 ...

1. b4 b5 2. c4 d5 3. h4 g6 4. d3 bxc4 5. Nd2 h6 6. d4 Qd7 7. Nb1 Bb7 8. Kd2 Nc6
9. Rh3 Rd8 10. Rc3 Rb8 11. Re3 Rd8 12. Rh3 Rb8 13. Rc3 Rd8 14. Re3 Rb8 15. Rh3 Rd8 ...
```

In 2025 this was treated as a reward problem: penalties for "oscillation" and "repeated rook moves" were added (commit `b44a7b9`). It is actually what you would expect from design flaw 1. A network that outputs "action index 5" in every position plays whatever move happens to be fifth in python-chess's move-generation order. That is often the same piece moving back and forth.

## Design flaws

### 1. Unstable action space
`action_space = Discrete(218)` ([L59][env59]), and `step(action)` played `legal_moves_list[action]`. The meaning of every output changes from position to position, so a Q-value for "action 5" cannot generalise.

"Action masking" was claimed in the README and in training logs, but the policy simply returned `argmax` over all 218 outputs ([train_agent.py L87][ta87]). A wrapper then wrapped out-of-range indices with `action % len(legal_moves)` ([L105][ta105]), which biases play toward low indices. Nothing was masked.

**The fix:** a fixed move encoding (AlphaZero's 8×8×73 = 4672 moves, relative to the side to move), where index *i* always means the same move geometry, and a real legal-move mask applied before the softmax or argmax (#14).

### 2. Self-play without perspective
A single DQN chose moves for both colours. The observation was 12 piece planes with **no side-to-move, castling, en-passant, or repetition information**, so the network could not tell whose turn it was. Rewards were computed for the side that had just moved. The TD target, however, bootstrapped from the next state, where the *opponent* moves, without negating the value. White's good move and Black's good move were pulling the same Q-values in opposite directions.

**The fix:** encode the board from the side to move's perspective (#13), and use a value target from the side to move's point of view (negamax-style), as in AlphaZero.

### 3. Shaped rewards instead of outcomes
`calculate_reward` combined 15 hand-written components: a +0.2 bonus for any legal move, centre control, development, mobility, pawn structure, captures, defence, hanging pieces, a "safe move" bonus, and more. Many of them paid out on every move ([L243][env243], [L381][env381]). Episodes had no length cap. The result was predictable: the agent maximised return by making games longer. That is exactly what the curriculum log shows: reward ≈ +205 at ≈ 350 plies per game.

**The fix:** the redesigned engine learns from game outcomes only (win / draw / loss). It gets its chess knowledge from human games (supervised pretraining, M4) and from search (M5), not from reward shaping.

## Bugs

Each bug is reproduced by a test. Tests marked `xfail` document v0 behaviour that is intentionally not fixed, because the redesign removes that code.

| Bug | Where | Effect | Status |
|---|---|---|---|
| Checkmate rewards contradict each other | [L257][env257] gives +10 only when **White** mates; [L398][env398] gives +10 only when **Black** mates | Measured reward on the mating move: **+0.5 for White, +2.98 for Black** | `test_mating_move_reward_is_colour_symmetric` (xfail) |
| `is_capture(move)` evaluated after the move was pushed | [L323][env323] | Every move counted as a capture | `test_quiet_move_is_not_rewarded_as_capture` (xfail) |
| Attack maps computed before the move, used after it | [L155-161, 313-375][env155] | Pawn-structure, defence and hanging rewards scored the wrong position | Removed with the redesign |
| Hanging-piece penalty scored the opponent's defended pieces | [L368][env368] | Together with the stale attack maps, quiet opening moves scored about −2 in total | Removed with the redesign |
| `elif is_repetition(3)` could never run | [L301][env301] | Threefold repetition was penalised the same as twofold | Removed with the redesign |
| "Mate in 1" curriculum position has no mate | [L108][env108] | Mislabelled puzzle (the f6 knight blocks the queen) | Label to be fixed (#15) |
| Opening book imported from a nonexistent module | [L41][env41] | `main.py train` and `main.py test` crashed with the default settings | Fixed in #49 |
| `MetricsLogger` built a DataFrame from a list that was never filled | [custom_callbacks.py L241][cb241] | Every run ended with `All arrays must be of the same length`. In the curriculum, `except Exception: break` ([L332-334][tt332]) turned this into "stages 2–4 never run". | Fixed in #50 |
| `model.replay_buffer = None` after loading the stage model | [L311][tt311] | Stage 2 would have crashed (it was never reached) | Fixed in #51 |
| `MaskedDQNPolicy.train` override for a value head | [L123][tt123] | Never called by SB3: the "auxiliary value task" was never trained, and the override broke `policy.eval()` | Fixed in #51 |
| Broad `except Exception` and then saving a model as `final` | [train_agent.py][ta188], [train_tactical_agent.py][tt332] | Crashes were reported as "interrupted" and still produced a "final" model | Fixed in #51 |

## Process lessons

- **No tests and no CI.** Most bugs in the table above would have been caught by a 40-second test suite. The project now has one (#46), and `main` is protected by CI (#47, #5).
- **Metrics that can't tell success from failure.** Mean episode reward under shaped rewards rose while play stayed random. The new project measures strength directly: Elo with confidence intervals against fixed opponents (M3).
- **Fixing symptoms in the reward function.** Rook shuffling was patched with more penalties. The real cause was the action space. Before tuning rewards, ask whether the model can represent the behaviour you want at all.
- **The project log could mislead.** The training scripts printed a list of "Key improvements made" (action masking, value prediction, rebalanced rewards), several of which did not exist in the code.

## What replaces v0

Engine: a fixed 4672-move encoding with real masking, a side-to-move board encoding, a policy/value ResNet pretrained on human games, MCTS search, and self-play RL. Project angle: an engine built to beat one specific person by modelling them (Om-sim) and exploiting that model in search and RL. See the milestones M2–M8.

## Measured baseline: v0.1.0

With all the crash bugs fixed, the canonical pipeline was trained for 300k steps through all four curriculum stages (46 min on an RTX 3050 laptop GPU), then played 100 games against a uniformly random legal-move player. Colours alternated, and games still running at 300 plies were counted as draws.

| Matchup | W / D / L | Score |
|---|---|---|
| v0 agent vs random | 5 / 88 / 7 | **49.0%** (95% CI ≈ 45.6–52.4%) |
| random vs random (reference) | 10 / 85 / 5 | 52.5% |

That is about **−7 ± 24 Elo against random**: no measurable chess skill. During training, mean episode reward was still +193 with games of ~382 plies, the reward-farming signature described above. Weights, games, and logs: [release v0.1.0](https://github.com/omrode1/Chess-Engine-RL/releases/tag/v0.1.0).

Reproduce: `uv run python -m baseline.eval_vs_random --model <v0.1.0 zip> --games 100 --seed 0`.

[env41]: https://github.com/omrode1/Chess-Engine-RL/blob/deb8250/src/chess_environment.py#L41
[env59]: https://github.com/omrode1/Chess-Engine-RL/blob/deb8250/src/chess_environment.py#L59
[env108]: https://github.com/omrode1/Chess-Engine-RL/blob/deb8250/src/chess_environment.py#L108
[env155]: https://github.com/omrode1/Chess-Engine-RL/blob/deb8250/src/chess_environment.py#L155-L161
[env243]: https://github.com/omrode1/Chess-Engine-RL/blob/deb8250/src/chess_environment.py#L243
[env257]: https://github.com/omrode1/Chess-Engine-RL/blob/deb8250/src/chess_environment.py#L257
[env301]: https://github.com/omrode1/Chess-Engine-RL/blob/deb8250/src/chess_environment.py#L301-L304
[env323]: https://github.com/omrode1/Chess-Engine-RL/blob/deb8250/src/chess_environment.py#L323
[env368]: https://github.com/omrode1/Chess-Engine-RL/blob/deb8250/src/chess_environment.py#L364-L377
[env381]: https://github.com/omrode1/Chess-Engine-RL/blob/deb8250/src/chess_environment.py#L381
[env398]: https://github.com/omrode1/Chess-Engine-RL/blob/deb8250/src/chess_environment.py#L398
[ta87]: https://github.com/omrode1/Chess-Engine-RL/blob/deb8250/train_agent.py#L73-L87
[ta105]: https://github.com/omrode1/Chess-Engine-RL/blob/deb8250/train_agent.py#L98-L108
[ta188]: https://github.com/omrode1/Chess-Engine-RL/blob/deb8250/train_agent.py#L181-L189
[tt123]: https://github.com/omrode1/Chess-Engine-RL/blob/deb8250/train_tactical_agent.py#L123-L155
[tt311]: https://github.com/omrode1/Chess-Engine-RL/blob/deb8250/train_tactical_agent.py#L311
[tt332]: https://github.com/omrode1/Chess-Engine-RL/blob/deb8250/train_tactical_agent.py#L332-L334
[cb241]: https://github.com/omrode1/Chess-Engine-RL/blob/deb8250/utils/custom_callbacks.py#L236-L241
