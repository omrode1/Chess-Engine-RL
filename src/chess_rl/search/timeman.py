"""Time management: how long to think about one move, given the clock.

    soft  the target: remaining / moves_to_go + most of the increment (moves_to_go shrinks from 45 as
          the game goes on, down to 15 with an increment or 30 in sudden death)
    hard  never more than this: half the remaining time, a quarter of it plus most of the increment,
          or 3x soft, whichever is least

The search stops at `soft`, or earlier when the most-visited move cannot be overtaken by the
second one in the time left (at the current speed), or when the position is solved.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess

from chess_rl.eval.players import Clock


@dataclass(frozen=True)
class TimeManager:
    move_overhead: float = 0.1  # seconds lost per move outside the search (process, GUI, network)
    min_moves_to_go: float = 15  # with an increment
    min_moves_to_go_sudden_death: float = 30  # without one, the clock never refills: plan for a long game
    max_moves_to_go: float = 45
    inc_share: float = 0.8

    def budget(self, board: chess.Board, clock: Clock) -> tuple[float, float]:
        """(soft, hard) thinking time in seconds for the side to move."""
        mine = clock.white if board.turn == chess.WHITE else clock.black
        inc = clock.white_inc if board.turn == chess.WHITE else clock.black_inc
        usable = max(mine - self.move_overhead, 0.0)
        floor_mtg = self.min_moves_to_go if inc > 0 else self.min_moves_to_go_sudden_death
        moves_to_go = max(floor_mtg, self.max_moves_to_go - board.fullmove_number / 2)
        soft = usable / moves_to_go + self.inc_share * inc
        hard = min(0.5 * usable, 0.25 * usable + self.inc_share * inc, 3 * soft)
        soft = min(soft, hard)
        floor = min(0.02, usable / 4)
        return max(soft, floor), max(hard, floor)

    def stopper(self, soft: float, hard: float):
        """A should_stop callback for MCTS.search."""

        def should_stop(search, root, nodes: int, elapsed: float) -> bool:
            if elapsed >= soft or elapsed >= hard:
                return True
            if nodes < 64 or root.N is None or len(root.N) < 2:
                return False
            first, second = sorted(root.N, reverse=True)[:2]
            nodes_left = nodes / elapsed * (soft - elapsed)
            return first - second > nodes_left

        return should_stop
