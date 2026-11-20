"""The network plus MCTS as a Player (M5).

mcts:path=runs/m4/m4-final.pt,nodes=800         fixed number of search nodes per move
mcts:path=...,movetime=1.0                      fixed time per move (seconds)
mcts:path=...,nodes=800,device=cpu,batch=16     other keys: c_puct, fpu, policy_temp, threads
"""

from __future__ import annotations

import numpy as np
import torch

from chess_rl.eval.players import Clock
from chess_rl.model.network import ChessNet, load_checkpoint, prepare_input, to_device
from chess_rl.search.mcts import MCTS, SearchConfig, SearchResult


class NetEvaluator:
    """Batched network calls for MCTS: planes -> (policy logits, W - L values), both numpy float32."""

    def __init__(self, model: ChessNet, device: torch.device):
        self.model = model
        self.device = device
        self.calls = 0
        self.positions = 0

    @torch.inference_mode()
    def __call__(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        self.positions += len(planes)
        x = prepare_input(torch.from_numpy(planes).to(self.device, non_blocking=True))
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            logits, value = self.model(x)
        wdl = torch.softmax(value.float(), dim=-1)
        v = wdl[:, 2] - wdl[:, 0]
        return logits.float().cpu().numpy(), v.cpu().numpy()


class MCTSPlayer:
    def __init__(
        self,
        path: str | None = None,
        model: ChessNet | None = None,
        nodes: int | None = None,
        movetime: float | None = None,
        config: SearchConfig = SearchConfig(),
        device: str = "auto",
        name: str = "mcts",
        threads: int = 1,
    ):
        if nodes is None and movetime is None:
            nodes = 800
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cpu":
            torch.set_num_threads(threads)  # see NetPlayer: matches run many players in parallel
        if model is None:
            if path is None:
                raise ValueError("MCTSPlayer needs a checkpoint path or a model")
            model, _ = load_checkpoint(path, self.device)
        self.evaluator = NetEvaluator(to_device(model, self.device).eval(), self.device)
        self.search = MCTS(self.evaluator, config)
        self.nodes, self.movetime = nodes, movetime
        self.name = name
        self.last: SearchResult | None = None

    def new_game(self) -> None:
        self.search.reset()

    def choose(self, board, clock: Clock | None = None):
        self.last = self.search.search(board, nodes=self.nodes, movetime=self.movetime)
        return self.last.move

    def close(self) -> None:
        pass
