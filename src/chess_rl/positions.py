"""Curriculum start positions, carried over unchanged from the 2025 v0 environment.

Levels: 0 = standard start (full game), 1 = middlegames, 2 = endgames, 3 = "tactical" positions.
They are used as self-play starting positions (M7) and in the tactics suite (M3). The order within
each level is kept from v0, so seeded random choices pick the same positions as before.

The 2025 labels are kept as descriptions of intent, but the level-3 "tactical" positions contain no
tactics. Stockfish 19 at depth 24 (multipv 5) rates them as ordinary openings: "Fork opportunity"
-110 cp for White with 5 moves within 30 cp of best, "Pin opportunity" -61 cp (3 such moves),
"Discovered attack" +19 cp (5), and the former "Mate in 1" 0 cp (only Ne2). The "Mate in 1" label was
corrected, because Qxf7 is blocked by the knight on f6. For tactics measurement use the WAC suite
(chess_rl.eval.tactics).
"""

from __future__ import annotations

from dataclasses import dataclass

import chess


@dataclass(frozen=True)
class Position:
    label: str
    fen: str

    def board(self) -> chess.Board:
        return chess.Board(self.fen)


LEVEL_NAMES = {0: "full games", 1: "midgames", 2: "endgames", 3: "tactical puzzles"}

CURRICULUM: dict[int, tuple[Position, ...]] = {
    0: (Position("Standard start", chess.STARTING_FEN),),
    1: (
        Position("Open game with development", "r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"),
        Position("Semi-open game", "rnbqkb1r/pp3ppp/2p1pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 6"),
        Position("Closed position", "r1bqk2r/ppp2ppp/2n1pn2/3p4/1bPP4/2NBP3/PP3PPP/R1BQK1NR w KQkq - 0 7"),
        Position("Open with early queen exchange", "rnb1kbnr/pppp1ppp/8/4p3/3P4/8/PPP2PPP/RNBQKBNR w KQkq - 0 4"),
    ),
    2: (
        Position("Rook endgame", "8/5pk1/7p/8/5P2/5K2/7P/4R3 w - - 0 1"),
        Position("Pawn endgame", "8/5p2/5k2/8/5P2/8/5K2/8 w - - 0 1"),
        Position("Queen vs rook endgame", "8/8/5k2/8/8/3Q4/5K2/7r w - - 0 1"),
        Position("Bishop and knight checkmate", "8/8/5k2/8/8/3B4/5K2/5N2 w - - 0 1"),
    ),
    3: (
        Position("Fork opportunity", "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/3P1P2/PPP3PP/RNBQKBNR w KQkq - 0 1"),
        Position("Pin opportunity", "r1bqkb1r/ppp2ppp/2n2n2/3pp3/4P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 0 1"),
        Position("Discovered attack", "r1bqkb1r/ppp2ppp/2n2n2/3p4/3P4/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 1"),
        Position(
            "Queen and bishop aim at f7 (no mate in 1: the f6 knight blocks Qxf7)",
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1",
        ),
    ),
}


def fens(level: int) -> list[str]:
    """FEN strings for a curriculum level, in their original order."""
    return [p.fen for p in CURRICULUM[level]]
