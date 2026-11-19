"""Elo difference with a confidence interval from match results.

Uses the trinomial (win/draw/loss) per-game score variance and a normal approximation on the mean
score, mapped through the logistic Elo curve. LOS (likelihood of superiority) ignores draws:
LOS = Phi((W - L) / sqrt(W + L)).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

Z95 = 1.959963984540054


def score_to_elo(score: float) -> float:
    """Elo difference implied by an expected score in (0, 1). Returns +-inf at 0 or 1."""
    if score <= 0.0:
        return -math.inf
    if score >= 1.0:
        return math.inf
    return -400.0 * math.log10(1.0 / score - 1.0)


def elo_to_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass(frozen=True)
class EloEstimate:
    wins: int
    draws: int
    losses: int
    score: float  # mean score of the first player, 0..1
    elo: float
    elo_low: float  # 95% confidence interval
    elo_high: float
    los: float | None  # probability the first player is stronger; None if no decisive games

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    def as_dict(self) -> dict:
        return {**asdict(self), "games": self.games}

    def __str__(self) -> str:
        los = f"{100 * self.los:.1f}%" if self.los is not None else "n/a"
        return (
            f"+{self.wins} ={self.draws} -{self.losses}  score {100 * self.score:.1f}%  "
            f"Elo {self.elo:+.0f} [{self.elo_low:+.0f}, {self.elo_high:+.0f}]  LOS {los}"
        )


def estimate(wins: int, draws: int, losses: int) -> EloEstimate:
    n = wins + draws + losses
    if n == 0:
        raise ValueError("no games")
    score = (wins + 0.5 * draws) / n
    var = (wins * (1 - score) ** 2 + draws * (0.5 - score) ** 2 + losses * (0 - score) ** 2) / n
    se = math.sqrt(var / n)
    low, high = max(score - Z95 * se, 0.0), min(score + Z95 * se, 1.0)
    decisive = wins + losses
    los = _phi((wins - losses) / math.sqrt(decisive)) if decisive else None
    return EloEstimate(wins, draws, losses, score, score_to_elo(score), score_to_elo(low), score_to_elo(high), los)
