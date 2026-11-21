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

Exploitative search (#38): with an `opponent` model (planes -> move logits of the human being played),
nodes where the opponent is to move get human priors H. There, a `caution` share of the visits is
adversarial (PUCT, as above) and the rest follow H: each such visit goes to the move whose share of
these visits lags H the most. Values back up as averages, so an opponent node's value tends to
caution * (a strong reply) + (1 - caution) * (the average human reply). caution = 1 is standard search.
Proven results stay exact, so a forced loss the search has found is never gambled on, and every opponent
node is checked for a mate in one first (with caution = 0 nothing else is adversarial). Deeper forced lines
are found only through the adversarial share: keep caution above 0.

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
    caution: float = 1.0  # with an opponent model: share of adversarial visits at opponent nodes (1 = standard)


class Node:
    __slots__ = (
        "moves",
        "P",
        "N",
        "W",
        "PQ",
        "children",
        "n_sum",
        "visits",
        "v_sum",
        "proven",
        "pending",
        "key",
        "H",
        "NH",
        "adv",
    )

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
        self.H: np.ndarray | None = None  # human priors (opponent nodes in exploitative search)
        self.NH: np.ndarray | None = None  # visits that followed H
        self.adv = 0  # visits that were adversarial

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

    def set_human(self, priors: np.ndarray) -> None:
        self.H = priors.astype(np.float32)
        self.NH = np.zeros(len(priors), dtype=np.float32)


PUCT, ADVERSARIAL, HUMAN = 0, 1, 2  # how a child was selected (undone exactly on a collision)

# (B, 21, 8, 8) planes, side to move = the opponent -> (B, 4672) move logits of the human being played
OpponentModel = Callable[[np.ndarray], np.ndarray]


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


def _has_mate_in_one(board: chess.Board, moves: list[chess.Move]) -> bool:
    for move in moves:
        if board.gives_check(move):
            board.push(move)
            mate = board.is_checkmate()
            board.pop()
            if mate:
                return True
    return False


class MCTS:
    def __init__(
        self, evaluator: Evaluator, config: SearchConfig = SearchConfig(), opponent: OpponentModel | None = None
    ):
        self.evaluate = evaluator
        self.config = config
        self.opponent = opponent if config.caution < 1.0 else None
        self._us: chess.Color = chess.WHITE
        self._root: Node | None = None
        self._root_stack: list[chess.Move] = []
        self._root_fen = ""
        self._cache: OrderedDict[tuple, tuple[np.ndarray, float, np.ndarray | None]] = OrderedDict()

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
        if board.turn != self._us:  # playing the other colour now: human priors were for the other side
            self._us = board.turn
            self.reset()
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
            limit = self.config.batch_size if nodes is None else max(1, min(self.config.batch_size, nodes - done))
            done += self._step(root, work, limit)
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

    def _step(self, root: Node, board: chess.Board, limit: int) -> int:
        """Collect up to `limit` (<= batch_size) leaves, evaluate them in one call, back up. Returns walks completed."""
        cfg = self.config
        batch: list[tuple[Node, list[tuple[Node, int]], list[chess.Move], np.ndarray]] = []
        planes: list[np.ndarray] = []
        human_rows: list[int] = []  # batch entries where the opponent is to move
        completed = collisions = 0
        while len(batch) + completed < limit and collisions <= cfg.max_collisions:
            node, path, kinds = root, [], []
            while node.moves is not None and node.proven is None:
                i, kind = self._select(node, root=node is root)
                node.N[i] += 1
                node.W[i] -= 1  # virtual loss
                node.n_sum += 1
                path.append((node, i))
                kinds.append(kind)
                board.push(node.moves[i])
                child = node.children[i]
                if child is None:
                    child = node.children[i] = Node()
                node = child
            if node.proven is not None:  # terminal (or solved) position: exact value, no network needed
                self._backup(path, node, node.proven)
                completed += 1
            elif node.pending:
                self._revert(path, kinds)
                collisions += 1
            else:
                moves = list(board.legal_moves)
                term = _terminal_value(board, moves)
                if term is None and self._needs_human(board) and _has_mate_in_one(board, moves):
                    term = 1.0  # the opponent mates next move: never assume the human misses it
                if term is not None:
                    node.proven = term
                    self._backup(path, node, term)
                    completed += 1
                else:
                    key = (board._transposition_key(), board.halfmove_clock)
                    hit = self._cache.get(key) if cfg.cache_size else None
                    if hit is not None and (hit[2] is not None or not self._needs_human(board)):
                        self._cache.move_to_end(key)
                        node.expand(moves, hit[0])
                        if hit[2] is not None:
                            node.set_human(hit[2])
                        self._backup(path, node, hit[1])
                        completed += 1
                    else:
                        node.pending, node.key = True, key
                        batch.append((node, path, moves, self._indices(moves, board.turn)))
                        planes.append(encode_board(board))
                        if self._needs_human(board):
                            human_rows.append(len(batch) - 1)
            for _ in path:
                board.pop()
            if root.proven is not None:
                break
        if batch:
            stacked = np.stack(planes)
            logits, values = self.evaluate(stacked)
            human: dict[int, np.ndarray] = {}
            if human_rows and self.opponent is not None:
                human_logits = self.opponent(stacked[human_rows])
                human = {j: human_logits[k] for k, j in enumerate(human_rows)}
            for j, (node, path, moves, idx) in enumerate(batch):
                priors, value = _priors(logits[j], idx, cfg.policy_temp), float(values[j])
                node.pending = False
                node.expand(moves, priors)
                h = _priors(human[j], idx, 1.0) if j in human else None
                if h is not None:
                    node.set_human(h)
                if cfg.cache_size:
                    self._cache[node.key] = (priors, value, h)
                    if len(self._cache) > cfg.cache_size:
                        self._cache.popitem(last=False)
                self._backup(path, node, value)
            completed += len(batch)
        return completed

    def _needs_human(self, board: chess.Board) -> bool:
        return self.opponent is not None and board.turn != self._us

    def _select(self, node: Node, root: bool = False) -> tuple[int, int]:
        cfg = self.config
        if node.H is not None and node.adv >= cfg.caution * node.n_sum:
            # follow the human: the move whose share of human-following visits lags its prior the most
            i = int(np.argmax(node.H - node.NH / (node.NH.sum() + 1)))
            node.NH[i] += 1
            return i, HUMAN
        visited = node.N > 0
        fpu = node.q - cfg.fpu_reduction * math.sqrt(float(node.P[visited].sum()))
        q = np.where(visited, node.W / np.maximum(node.N, 1), fpu)
        q = np.where(np.isnan(node.PQ), q, node.PQ)
        u = cfg.c_puct * math.sqrt(max(node.n_sum, 1)) * node.P / (1 + node.N)
        if root:  # at the root a proven move's value is exact: no exploration bonus (interior search unchanged)
            u = np.where(np.isnan(node.PQ), u, 0.0)
        if node.H is not None:
            node.adv += 1
            return int(np.argmax(q + u)), ADVERSARIAL
        return int(np.argmax(q + u)), PUCT

    @staticmethod
    def _revert(path: list[tuple[Node, int]], kinds: list[int]) -> None:
        for (node, i), kind in zip(path, kinds, strict=True):
            node.N[i] -= 1
            node.W[i] += 1
            node.n_sum -= 1
            if kind == HUMAN:
                node.NH[i] -= 1
            elif kind == ADVERSARIAL:
                node.adv -= 1

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
            # Most visits among searched moves (ties by Q), but proven moves compete on value: a proven
            # draw or loss never replaces a searched move that scores better (at small budgets a high-prior
            # stalemate can collect the most visits), and a proven draw is taken when it beats the best
            # searched move (a saving repetition).
            q = np.where(root.N > 0, root.W / np.maximum(root.N, 1), -2.0)
            searched = np.isnan(root.PQ) & (root.N > 0)
            proven = ~np.isnan(root.PQ)
            best = -1
            if searched.any():
                n = np.where(searched, root.N.astype(np.float64), -1.0)
                best = int(np.lexsort((q, n))[-1])
            if proven.any():
                p = int(np.argmax(np.where(proven, root.PQ, -np.inf)))
                if best < 0 or root.PQ[p] > q[best]:
                    best = p
            if best < 0:  # nothing searched or proven (cannot happen after one walk); fall back to the prior
                best = int(np.argmax(root.P))
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
