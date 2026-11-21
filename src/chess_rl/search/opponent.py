"""The human move model as an opponent: a move distribution for search (#38) and a player for matches.

    HumanPolicy   batched planes -> move logits of a human at a fixed rating (optionally a specific player,
                  e.g. Om-sim's row), for opponent nodes in exploitative MCTS
    HumanPlayer   samples moves from that distribution: the simulated opponent used in M7/M8 experiments

    human:path=runs/human/best.pt,elo=1200            a 1200 (Lichess) player sampled at temperature 1
    human:path=runs/omsim/om-sim.pt,elo=1200          Om-sim: uses the checkpoint's player row
    keys: elo, opp_elo (default = elo), temperature (default 1; 0 = most likely move), player, device

Ratings are on the Lichess scale the model was trained on. Om-sim's chess.com -> Lichess offset is in its
checkpoint's metadata (`offset`).
"""

from __future__ import annotations

import random

import chess
import numpy as np
import torch

from chess_rl.encoding import encode_board, legal_move_indices
from chess_rl.encoding.moves import index_to_move
from chess_rl.eval.players import Clock
from chess_rl.model.human import HumanNet, load_human
from chess_rl.model.network import prepare_input


class HumanPolicy:
    def __init__(
        self,
        model: HumanNet,
        elo: int = 1200,
        opp_elo: int | None = None,
        player: int | None = None,
        device: str | torch.device = "cpu",
        temperature: float = 1.0,
    ):
        if temperature <= 0:
            raise ValueError("HumanPolicy temperature must be > 0")
        self.temperature = temperature
        self.model = model.eval()
        self.device = torch.device(device)
        self.elo, self.opp_elo, self.player = elo, opp_elo if opp_elo is not None else elo, player
        self.calls = self.positions = 0

    @classmethod
    def load(
        cls,
        path: str,
        elo: int = 1200,
        opp_elo: int | None = None,
        player: int | None = None,
        device="auto",
        temperature: float = 1.0,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model, meta = load_human(path, device)
        if player is None:
            player = meta.get("player")  # Om-sim checkpoints name their player row
        return cls(model, elo, opp_elo, player, device, temperature)

    @torch.inference_mode()
    def __call__(self, planes: np.ndarray, clocks: np.ndarray | None = None) -> np.ndarray:
        """(B, 21, 8, 8) planes (side to move = the human) -> (B, 4672) logits, float32."""
        self.calls += 1
        self.positions += len(planes)
        n = len(planes)
        x = prepare_input(torch.from_numpy(planes).to(self.device))
        elo = torch.full((n,), self.elo, device=self.device)
        them = torch.full((n,), self.opp_elo, device=self.device)
        clock = torch.full((n,), -1, device=self.device) if clocks is None else torch.from_numpy(clocks).to(self.device)
        who = None if self.player is None else torch.full((n,), self.player, device=self.device)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            logits, _, _ = self.model(x, elo, them, clock, who)
        return (logits.float() / self.temperature).cpu().numpy()


class HumanPlayer:
    """Plays like the model's idea of a human at `elo`: samples from its move distribution."""

    def __init__(self, policy: HumanPolicy, temperature: float = 1.0, seed: int | None = None, name: str = "human"):
        self.policy = policy
        self.temperature = temperature
        self.name = name
        self._rng = random.Random(seed)

    def reseed(self, seed: int) -> None:
        self._rng.seed(seed)

    def new_game(self) -> None:
        pass

    def distribution(self, board: chess.Board, clock: Clock | None = None) -> tuple[list[chess.Move], np.ndarray]:
        seconds = -1
        if clock is not None:
            seconds = int(clock.white if board.turn == chess.WHITE else clock.black)
        logits = self.policy(encode_board(board)[None], np.array([seconds]))[0]
        idx = legal_move_indices(board)
        z = logits[idx].astype(np.float64)
        p = np.exp(z - z.max())
        return [index_to_move(int(i), board) for i in idx], p / p.sum()

    def choose(self, board: chess.Board, clock: Clock | None = None) -> chess.Move:
        moves, p = self.distribution(board, clock)
        if self.temperature <= 0:
            return moves[int(np.argmax(p))]
        w = p ** (1.0 / self.temperature)
        return self._rng.choices(moves, weights=w.tolist())[0]

    def close(self) -> None:
        pass
