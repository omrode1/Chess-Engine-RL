"""The network plus MCTS as a Player (M5).

mcts:path=runs/m4/m4-final.pt                   with a clock: time-managed; without: 800 nodes per move
mcts:path=...,nodes=800                         fixed number of search nodes per move
mcts:path=...,movetime=1.0                      fixed time per move (seconds)
mcts:path=...,book=true                         play opening-book moves while in book (for human games)
mcts:path=...,nodes=800,device=cpu,batch=16     other keys: c_puct, fpu, policy_temp, threads, overhead

A fixed budget (nodes / movetime) is still capped by the clock's hard limit when a clock is given.
"""

from __future__ import annotations

import random

import chess
import numpy as np
import torch

from chess_rl.encoding import NUM_PLANES
from chess_rl.eval.players import Clock
from chess_rl.model.network import ChessNet, load_checkpoint, prepare_input, to_device
from chess_rl.openings import OpeningBook
from chess_rl.search.mcts import MCTS, SearchConfig, SearchResult
from chess_rl.search.timeman import TimeManager


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
    DEFAULT_NODES = 800

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
        book: bool = False,
        time_manager: TimeManager = TimeManager(),
        seed: int | None = None,
    ):
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
        # warm up (CUDA context, cuDNN kernel choice) now, not on the first move's clock
        for batch in {1, config.batch_size}:
            self.evaluator(np.zeros((batch, NUM_PLANES, 8, 8), np.float32))
        self.search = MCTS(self.evaluator, config)
        self.nodes, self.movetime = nodes, movetime
        self.time_manager = time_manager
        self.book = OpeningBook() if book else None
        self._rng = random.Random(seed)
        self.name = name
        self.last: SearchResult | None = None
        self.last_budget: tuple[float, float] | None = None

    def reseed(self, seed: int) -> None:
        self._rng.seed(seed)

    def new_game(self) -> None:
        self.search.reset()

    def book_move(self, board: chess.Board) -> chess.Move | None:
        if self.book is None:
            return None
        entries = [(m, w) for m, w in self.book.book.get(board.board_fen(), []) if board.is_legal(m)]
        if not entries:
            return None
        return self._rng.choices([m for m, _ in entries], weights=[w for _, w in entries])[0]

    def choose(self, board: chess.Board, clock: Clock | None = None) -> chess.Move:
        move = self.book_move(board)
        if move is not None:
            self.last, self.last_budget = None, None
            return move
        nodes, movetime, stop = self.nodes, self.movetime, None
        self.last_budget = None
        if clock is not None:
            soft, hard = self.time_manager.budget(board, clock)
            self.last_budget = (soft, hard)
            if nodes is None and movetime is None:
                stop = self.time_manager.stopper(soft, hard)
            else:  # fixed budget, but never flag
                movetime = min(movetime, hard) if movetime is not None else hard
        elif nodes is None and movetime is None:
            nodes = self.DEFAULT_NODES
        self.last = self.search.search(board, nodes=nodes, movetime=movetime, should_stop=stop)
        return self.last.move

    def close(self) -> None:
        pass
