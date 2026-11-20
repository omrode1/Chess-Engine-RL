"""Human move model (M6, #35): predicts the move a human of a given rating plays, and whether it is a blunder.

Maia-style, with Maia-2's idea of conditioning one network on both players' ratings instead of training one
network per rating band. The residual tower is ChessNet's; each block's output is modulated (FiLM: scale and
shift per channel) by a conditioning vector built from

    elo_us, elo_them   100-point rating buckets (Lichess scale), separate embeddings
    clock              the mover's remaining time in log-spaced buckets (+ "unknown")
    player             a player embedding: 0 = the population; Om-sim (#36) learns its own row

Heads:
    policy   (N, 4672) logits over moves (chess_rl.encoding.moves), as ChessNet
    value    (N, 3) game result logits (loss/draw/win) for the side to move, *as humans of these ratings play it*
    blunder  (N,) logit that the move about to be played loses >= 10 win-percentage points

The FiLM layers start at zero, so before training the conditioning is the identity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from chess_rl.encoding.board import NUM_PLANES
from chess_rl.encoding.moves import NUM_MOVE_TYPES, NUM_MOVES
from chess_rl.model.network import SqueezeExcite, to_device

ELO_MIN, ELO_STEP, ELO_BUCKETS = 400, 100, 26  # 400..2900+
CLOCK_EDGES = (10, 20, 30, 60, 120, 180, 300, 600, 1200)  # seconds; bucket 0 = unknown
CLOCK_BUCKETS = len(CLOCK_EDGES) + 2
BLUNDER_POINTS = 10


@dataclass(frozen=True)
class HumanNetConfig:
    blocks: int = 10
    filters: int = 128
    se: bool = True
    se_ratio: int = 4
    value_hidden: int = 128
    cond_dim: int = 64
    players: int = 1  # row 0 = population

    kind: str = "human"


def elo_bucket(elo: torch.Tensor) -> torch.Tensor:
    return torch.clamp((elo.long() - ELO_MIN) // ELO_STEP, 0, ELO_BUCKETS - 1)


def clock_bucket(seconds: torch.Tensor) -> torch.Tensor:
    edges = torch.tensor(CLOCK_EDGES, device=seconds.device)
    b = torch.bucketize(seconds.long(), edges, right=True) + 1
    return torch.where(seconds < 0, torch.zeros_like(b), b)


def clock_bucket_np(seconds: np.ndarray) -> np.ndarray:
    b = np.searchsorted(np.asarray(CLOCK_EDGES), seconds, side="right") + 1
    return np.where(seconds < 0, 0, b)


class FiLMBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int, se: bool, se_ratio: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcite(channels, se_ratio) if se else nn.Identity()
        self.film = nn.Linear(cond_dim, 2 * channels)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        gamma, beta = self.film(cond).chunk(2, dim=1)
        y = y * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return F.relu(x + self.se(y))


class HumanNet(nn.Module):
    def __init__(self, config: HumanNetConfig = HumanNetConfig()):
        super().__init__()
        self.config = config
        c, d = config.filters, config.cond_dim
        self.elo_us = nn.Embedding(ELO_BUCKETS, d)
        self.elo_them = nn.Embedding(ELO_BUCKETS, d)
        self.clock = nn.Embedding(CLOCK_BUCKETS, d)
        self.player = nn.Embedding(config.players, d)
        nn.init.zeros_(self.player.weight)
        self.cond = nn.Sequential(nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.stem = nn.Sequential(nn.Conv2d(NUM_PLANES, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU())
        self.tower = nn.ModuleList(FiLMBlock(c, d, config.se, config.se_ratio) for _ in range(config.blocks))
        self.policy_conv = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU())
        self.policy_out = nn.Conv2d(c, NUM_MOVE_TYPES, 3, padding=1)
        self.value_conv = nn.Sequential(nn.Conv2d(c, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.value_fc = nn.Sequential(nn.Linear(32 * 64 + d, config.value_hidden), nn.ReLU())
        self.value_out = nn.Linear(config.value_hidden, 3)
        self.blunder_out = nn.Linear(config.value_hidden, 1)

    def condition(
        self, elo_us: torch.Tensor, elo_them: torch.Tensor, clock: torch.Tensor, player: torch.Tensor | None = None
    ) -> torch.Tensor:
        e = self.elo_us(elo_bucket(elo_us)) + self.elo_them(elo_bucket(elo_them)) + self.clock(clock_bucket(clock))
        if player is not None:
            e = e + self.player(player.long())
        return self.cond(e)

    def forward(
        self,
        planes: torch.Tensor,
        elo_us: torch.Tensor,
        elo_them: torch.Tensor,
        clock: torch.Tensor,
        player: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cond = self.condition(elo_us, elo_them, clock, player)
        x = self.stem(planes)
        for block in self.tower:
            x = block(x, cond)
        p = self.policy_out(self.policy_conv(x))
        policy = p.permute(0, 2, 3, 1).reshape(-1, NUM_MOVES)
        h = self.value_fc(torch.cat([self.value_conv(x).flatten(1), cond.to(x.dtype)], dim=1))
        return policy, self.value_out(h), self.blunder_out(h).squeeze(1)

    def add_players(self, n: int) -> int:
        """Append n player rows (initialised to the population row, zero); returns the first new index."""
        old = self.player.weight.data
        first = old.shape[0]
        self.player = nn.Embedding(first + n, old.shape[1]).to(old.device)
        self.player.weight.data.zero_()
        self.player.weight.data[:first] = old
        self.config = HumanNetConfig(**{**asdict(self.config), "players": first + n})
        return first


def save_human(path: str | Path, model: HumanNet, **meta: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict(), "meta": meta}, path)


def load_human(path: str | Path, device: str | torch.device = "cpu") -> tuple[HumanNet, dict[str, Any]]:
    data = torch.load(path, map_location=device, weights_only=True)
    config = dict(data["config"])
    if config.get("kind") != "human":
        raise ValueError(f"{path} is not a human-model checkpoint")
    model = to_device(HumanNet(HumanNetConfig(**config)), device)  # type: ignore[arg-type]
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model, data.get("meta", {})
