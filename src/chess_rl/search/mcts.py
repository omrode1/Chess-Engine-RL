"""Monte Carlo tree search with network priors (PUCT), batched leaf evaluation and a small mate solver.

One search step walks from the root to a leaf, always taking the child with the highest

    Q(child) + c_puct * P(child) * sqrt(N(parent)) / (1 + N(child))

where P is the network's prior for the move, N counts visits, and Q is the mean result for the side
choosing the move (-1 loss .. +1 win). Unvisited children get a pessimistic "first play urgency" Q:
the parent's Q minus `fpu_reduction` * sqrt(prior mass already visited).

Leaves are collected in batches of `batch_size` for one GPU call. While a leaf waits for its
evaluation, every edge on its path carries a virtual loss, so the next walks in the same batch explore
other lines. The network's win/draw/loss output becomes the value W - L for the side to move.

Terminal positions are exact: checkmate = -1 for the side to move; stalemate, insufficient material,
the fifty-move rule and any repetition (twofold, inside the search or of the game history) = 0. Exact
results propagate up as "proven" values: a position with a move into a lost position is won, and a
position whose every move is proven is decided. Proven values override Q, so a forced mate is never
lost again to averaging, and a search with a proven root stops early.

The tree is kept between moves: if the new position follows the previous root by a few moves inside
the tree, that subtree becomes the new root. A transposition cache keeps network outputs by position.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

import chess
import numpy as np

from chess_rl.encoding import encode_board, move_to_index

# (B, 21, 8, 8) planes -> ((B, 4672) policy logits, (B,) values W - L for the side to move)
Evaluator = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class SearchConfig:
    c_puct: float = 1.75
    fpu_reduction: float = 0.3
    policy_temp: float = 1.0  # softmax temperature on the network's move logits
    batch_size: int = 32
    max_collisions: int = 8  # stop filling a batch after this many walks end on a leaf already waiting
    cache_size: int = 200_000  # positions; 0 disables the cache


class Node:
    __slots__ = ("moves", "P", "N", "W", "PQ", "children", "n_sum", "visits", "v_sum", "proven", "pending", "key")

    def __init__(self) -> None:
        self.moves: list[chess.Move] | None = None  # None until expanded
        self.P = self.N = self.W = self.PQ = None  # per-child arrays (PQ: proven Q, NaN if unknown)
        self.children: list[Node | None] = []
        self.n_sum = 0  # sum of N, including virtual visits
        self.visits = 0  # completed visits of this node
        self.v_sum = 0.0  # sum of values, side to move here
        self.proven: float | None = None  # exact value for the side to move here
        self.pending = False  # waiting for the network
        self.key: tuple | None = None  # transposition-cache key

    @property
    def q(self) -> float:
        return self.v_sum / self.visits if self.visits else 0.0

    def expand(self, moves: list[chess.Move], priors: np.ndarray) -> None:
        k = len(moves)
        self.moves = moves
        self.P = priors.astype(np.float32)
        self.N = np.zeros(k, dtype=np.float32)
        self.W = np.zeros(k, dtype=np.float64)
        self.PQ = np.full(k, np.nan, dtype=np.float64)
        self.children = [None] * k


@dataclass
class SearchResult:
    move: chess.Move
    nodes: int  # completed walks this search (leaf evaluations + terminal hits)
    seconds: float
    visits: dict[chess.Move, int]
    q: float  # value of the chosen move for the side to move, -1..1
    pv: list[chess.Move] = field(default_factory=list)
    proven: float | None = None  # exact root value if solved
    reused: int = 0  # visits carried over from the previous search

    @property
    def nps(self) -> float:
        return self.nodes / self.seconds if self.seconds > 0 else 0.0

    @property
    def score(self) -> float:
        """Expected score 0..1 for the side to move."""
        return (self.q + 1) / 2


def _priors(logits: np.ndarray, indices: np.ndarray, temp: float) -> np.ndarray:
    z = logits[indices].astype(np.float64) / temp
    z = np.exp(z - z.max())
    return z / z.sum()


def _terminal_value(board: chess.Board, moves: list[chess.Move]) -> float | None:
    if not moves:
        return -1.0 if board.is_check() else 0.0
    if board.halfmove_clock >= 100 or board.is_insufficient_material():
        return 0.0
    if board.halfmove_clock >= 4 and board.is_repetition(2):
        return 0.0
    return None


class MCTS:
    def __init__(self, evaluator: Evaluator, config: SearchConfig = SearchConfig()):
        self.evaluate = evaluator
        self.config = config
        self._root: Node | None = None
        self._root_stack: list[chess.Move] = []
        self._root_fen = ""
        self._cache: OrderedDict[tuple, tuple[np.ndarray, float]] = OrderedDict()

    def reset(self) -> None:
        self._root = None
        self._root_stack = []
        self._root_fen = ""
        self._cache.clear()

    # ---- tree reuse ----------------------------------------------------------------------------

    def _take_root(self, board: chess.Board) -> Node:
        old, stack = self._root, board.move_stack
        if old is not None and len(stack) >= len(self._root_stack):
            prefix = len(self._root_stack)
            if stack[:prefix] == self._root_stack and self._start_fen(board, prefix) == self._root_fen:
                node: Node | None = old
                for move in stack[prefix:]:
                    if node is None or node.moves is None or move not in node.moves:
                        node = None
                        break
                    node = node.children[node.moves.index(move)]
                if node is not None and node.proven is None:
                    return node
        return Node()

    @staticmethod
    def _start_fen(board: chess.Board, plies: int) -> str:
        b = board.copy(stack=True)
        while len(b.move_stack) > plies:
            b.pop()
        return b.fen()

    # ---- search --------------------------------------------------------------------------------

    def search(
        self,
        board: chess.Board,
        nodes: int | None = None,
        movetime: float | None = None,
        should_stop: Callable[[MCTS, Node, int, float], bool] | None = None,
    ) -> SearchResult:
        """Search `board` (not modified) until `nodes` walks or `movetime` seconds, whichever first.

        `should_stop(search, root, nodes_done, elapsed)` is polled after each batch (time management).
        """
        if nodes is None and movetime is None and should_stop is None:
            raise ValueError("give nodes, movetime or should_stop")
        moves = list(board.legal_moves)
        if not moves:
            raise ValueError("no legal moves")
        start = time.perf_counter()
        root = self._take_root(board)
        reused = root.visits
        self._root, self._root_stack, self._root_fen = root, list(board.move_stack), board.fen()
        work = board.copy(stack=True)
        if root.moves is None:
            logits, values = self.evaluate(encode_board(work)[None])
            root.expand(moves, _priors(logits[0], self._indices(moves, work.turn), self.config.policy_temp))
            root.visits, root.v_sum = 1, float(values[0])
        done = 0
        if len(moves) == 1:  # forced: no point searching
            return self._result(root, done, time.perf_counter() - start, reused)
        while True:
            done += self._step(root, work)
            elapsed = time.perf_counter() - start
            if root.proven is not None:  # solved: more search cannot change the result
                break
            if nodes is not None and done >= nodes:
                break
            if movetime is not None and elapsed >= movetime:
                break
            if should_stop is not None and should_stop(self, root, done, elapsed):
                break
        return self._result(root, done, time.perf_counter() - start, reused)

    @staticmethod
    def _indices(moves: list[chess.Move], turn: chess.Color) -> np.ndarray:
        return np.fromiter((move_to_index(m, turn) for m in moves), dtype=np.int64, count=len(moves))

    def _step(self, root: Node, board: chess.Board) -> int:
        """Collect up to batch_size leaves, evaluate them in one call, back up. Returns walks completed."""
        cfg = self.config
        batch: list[tuple[Node, list[tuple[Node, int]], list[chess.Move], np.ndarray]] = []
        planes: list[np.ndarray] = []
        completed = collisions = 0
        while len(batch) + completed < cfg.batch_size and collisions <= cfg.max_collisions:
            node, path = root, []
            while node.moves is not None and node.proven is None:
                i = self._select(node)
                node.N[i] += 1
                node.W[i] -= 1  # virtual loss
                node.n_sum += 1
                path.append((node, i))
                board.push(node.moves[i])
                child = node.children[i]
                if child is None:
                    child = node.children[i] = Node()
                node = child
            if node.proven is not None:  # terminal (or solved) position: exact value, no network needed
                self._backup(path, node, node.proven)
                completed += 1
            elif node.pending:
                self._revert(path)
                collisions += 1
            else:
                moves = list(board.legal_moves)
                term = _terminal_value(board, moves)
                if term is not None:
                    node.proven = term
                    self._backup(path, node, term)
                    completed += 1
                else:
                    key = (board._transposition_key(), board.halfmove_clock)
                    hit = self._cache.get(key) if cfg.cache_size else None
                    if hit is not None:
                        self._cache.move_to_end(key)
                        node.expand(moves, hit[0])
                        self._backup(path, node, hit[1])
                        completed += 1
                    else:
                        node.pending, node.key = True, key
                        batch.append((node, path, moves, self._indices(moves, board.turn)))
                        planes.append(encode_board(board))
            for _ in path:
                board.pop()
            if root.proven is not None:
                break
        if batch:
            logits, values = self.evaluate(np.stack(planes))
            for j, (node, path, moves, idx) in enumerate(batch):
                priors, value = _priors(logits[j], idx, cfg.policy_temp), float(values[j])
                node.pending = False
                node.expand(moves, priors)
                if cfg.cache_size:
                    self._cache[node.key] = (priors, value)
                    if len(self._cache) > cfg.cache_size:
                        self._cache.popitem(last=False)
                self._backup(path, node, value)
            completed += len(batch)
        return completed

    def _select(self, node: Node) -> int:
        cfg = self.config
        visited = node.N > 0
        fpu = node.q - cfg.fpu_reduction * math.sqrt(float(node.P[visited].sum()))
        q = np.where(visited, node.W / np.maximum(node.N, 1), fpu)
        q = np.where(np.isnan(node.PQ), q, node.PQ)
        u = cfg.c_puct * math.sqrt(max(node.n_sum, 1)) * node.P / (1 + node.N)
        return int(np.argmax(q + u))

    @staticmethod
    def _revert(path: list[tuple[Node, int]]) -> None:
        for node, i in path:
            node.N[i] -= 1
            node.W[i] += 1
            node.n_sum -= 1

    @staticmethod
    def _backup(path: list[tuple[Node, int]], leaf: Node, value: float) -> None:
        """`value` is for the side to move at `leaf`; each step up flips the perspective."""
        leaf.visits += 1
        leaf.v_sum += value
        child = leaf
        for node, i in reversed(path):
            value = -value
            node.W[i] += 1 + value  # +1 removes the virtual loss
            if child.proven is not None and np.isnan(node.PQ[i]):
                node.PQ[i] = -child.proven
                if node.proven is None:
                    if node.PQ[i] == 1.0:
                        node.proven = 1.0
                    elif not np.isnan(node.PQ).any():
                        node.proven = float(node.PQ.max())
            if node.proven is not None:
                value = node.proven
            node.visits += 1
            node.v_sum += value
            child = node

    # ---- results -------------------------------------------------------------------------------

    @staticmethod
    def _proven_choice(root: Node) -> int | None:
        """Index of a move that wins by force (if the root is a proven win)."""
        if root.proven != 1.0:
            return None
        wins = np.flatnonzero(root.PQ == 1.0)
        if not len(wins):
            return None
        # among forced wins, prefer the one searched most (usually the shortest mate)
        return int(wins[np.argmax(root.N[wins])])

    def _result(self, root: Node, done: int, seconds: float, reused: int) -> SearchResult:
        best = self._proven_choice(root)
        if best is None:
            # most visits; proven losses last; ties by Q
            n = root.N.astype(np.float64).copy()
            n[root.PQ == -1.0] = -1
            q = np.where(root.N > 0, root.W / np.maximum(root.N, 1), -2.0)
            best = int(np.lexsort((q, n))[-1])
        pv, node = [root.moves[best]], root.children[best]
        while node is not None and node.moves is not None and node.N.max() > 0:
            i = int(np.argmax(node.N))
            pv.append(node.moves[i])
            node = node.children[i]
        q = float(root.PQ[best]) if not np.isnan(root.PQ[best]) else float(root.W[best] / max(root.N[best], 1))
        return SearchResult(
            move=root.moves[best],
            nodes=done,
            seconds=seconds,
            visits={m: int(n) for m, n in zip(root.moves, root.N, strict=True)},
            q=q,
            pv=pv,
            proven=root.proven,
            reused=reused,
        )
