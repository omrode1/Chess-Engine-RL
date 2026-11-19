"""Score a v0 DQN model against a uniformly random legal-move player.

Colours alternate each game. Games that reach --max-plies are adjudicated as draws.
A random-vs-random reference run is included so the agent's score has context.

    uv run python -m scripts.eval_vs_random --model tactical_models/chess_tactical_final --games 100

Run it as a module from the repo root so that `src` is importable.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import chess
import chess.pgn
import numpy as np

from src.chess_environment import ChessEnvWithOpeningBook


@dataclass
class Score:
    wins: int = 0
    draws: int = 0
    losses: int = 0
    adjudicated: int = 0
    plies: list[int] = field(default_factory=list)

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def points(self) -> float:
        return self.wins + 0.5 * self.draws

    def summary(self) -> dict:
        n = self.games
        return {
            **{k: v for k, v in asdict(self).items() if k != "plies"},
            "games": n,
            "score_pct": round(100 * self.points / n, 1) if n else None,
            "mean_plies": round(float(np.mean(self.plies)), 1) if self.plies else None,
        }


def agent_move(model, env: ChessEnvWithOpeningBook) -> int:
    """Same action mapping as the v0 eval scripts: argmax Q, wrapped onto the legal-move list."""
    action, _ = model.predict(env.board_to_tensor(), deterministic=True)
    return int(action) % len(env.legal_moves_list)


def play_game(model, agent_is_white: bool, rng: random.Random, max_plies: int) -> tuple[str, chess.Board, bool]:
    env = ChessEnvWithOpeningBook(use_opening_book=False)
    env.reset()
    while not env.board.is_game_over(claim_draw=True) and env.board.ply() < max_plies:
        agents_turn = env.board.turn == (chess.WHITE if agent_is_white else chess.BLACK)
        if agents_turn and model is not None:
            action = agent_move(model, env)
        else:
            action = rng.randrange(len(env.legal_moves_list))
        env.step(action)
    adjudicated = not env.board.is_game_over(claim_draw=True)
    result = "1/2-1/2" if adjudicated else env.board.result(claim_draw=True)
    return result, env.board, adjudicated


def run(model, games: int, max_plies: int, seed: int, pgn_dir: Path | None, label: str) -> Score:
    rng = random.Random(seed)
    score = Score()
    for g in range(games):
        agent_white = g % 2 == 0
        result, board, adjudicated = play_game(model, agent_white, rng, max_plies)
        agent_result = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[result]
        if not agent_white:
            agent_result = 1.0 - agent_result
        score.wins += agent_result == 1.0
        score.draws += agent_result == 0.5
        score.losses += agent_result == 0.0
        score.adjudicated += adjudicated
        score.plies.append(board.ply())
        if pgn_dir is not None:
            game = chess.pgn.Game.from_board(board)
            game.headers.update(
                {
                    "Event": f"{label} vs random",
                    "White": label if agent_white else "random",
                    "Black": "random" if agent_white else label,
                    "Result": result,
                    "Round": str(g + 1),
                }
            )
            pgn_dir.mkdir(parents=True, exist_ok=True)
            with open(pgn_dir / f"{label}_vs_random.pgn", "a") as f:
                print(game, file=f, end="\n\n")
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Path to a saved SB3 DQN model (.zip)")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pgn-dir", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    from stable_baselines3 import DQN

    model = DQN.load(args.model, device="cpu")
    results = {
        "model": args.model,
        "max_plies": args.max_plies,
        "seed": args.seed,
        "agent_vs_random": run(model, args.games, args.max_plies, args.seed, args.pgn_dir, "agent").summary(),
        "random_vs_random": run(None, args.games, args.max_plies, args.seed, None, "random").summary(),
    }
    print(json.dumps(results, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
