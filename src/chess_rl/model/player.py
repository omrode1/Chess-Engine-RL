"""The raw network as a Player: picks from the policy over legal moves, no search.

net:path=runs/m4/best.pt                 argmax of the policy (default)
net:path=...,temperature=1.0,device=cpu  sample from the policy
"""

from __future__ import annotations

import random

import chess
import torch

from chess_rl.encoding import encode_board, legal_move_indices
from chess_rl.encoding.moves import index_to_move
from chess_rl.eval.players import Clock
from chess_rl.model.network import ChessNet, load_checkpoint, prepare_input, to_device


class NetPlayer:
    def __init__(
        self,
        path: str | None = None,
        model: ChessNet | None = None,
        device: str = "auto",
        temperature: float = 0.0,
        seed: int | None = None,
        name: str = "net",
        threads: int = 1,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cpu":
            # One thread per player: matches run many players in parallel processes, and torch's default
            # (all cores each) oversubscribes the CPU badly.
            torch.set_num_threads(threads)
        if model is None:
            if path is None:
                raise ValueError("NetPlayer needs a checkpoint path or a model")
            model, _ = load_checkpoint(path, self.device)
        self.model = to_device(model, self.device).eval()
        self.temperature = temperature
        self.name = name
        self._rng = random.Random(seed)

    def reseed(self, seed: int) -> None:
        self._rng.seed(seed)

    def new_game(self) -> None:
        pass

    @torch.no_grad()
    def policy(self, board: chess.Board) -> tuple[list[chess.Move], torch.Tensor, float]:
        """(legal moves, their probabilities, expected score for the side to move)."""
        x = prepare_input(torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device))
        logits, value = self.model(x)
        idx = legal_move_indices(board)
        legal_logits = logits[0, torch.from_numpy(idx).to(self.device)].float()
        probs = torch.softmax(legal_logits, dim=0).cpu()
        wdl = torch.softmax(value[0].float(), dim=0)
        moves = [index_to_move(int(i), board) for i in idx]
        return moves, probs, float(wdl[2] + 0.5 * wdl[1])

    def choose(self, board: chess.Board, clock: Clock | None = None) -> chess.Move:
        moves, probs, _ = self.policy(board)
        if self.temperature <= 0:
            return moves[int(torch.argmax(probs))]
        weights = probs.pow(1.0 / self.temperature).tolist()
        return self._rng.choices(moves, weights=weights)[0]

    def close(self) -> None:
        pass
