"""AlphaZero-style policy/value network, sized for an RTX 3050 laptop GPU (6 GB).

input   (N, 21, 8, 8)    board planes (chess_rl.encoding.board)
tower   stem conv3x3 + `blocks` residual blocks of `filters` channels (optional squeeze-excitation)
policy  conv3x3 -> 73 planes -> (N, 4672) logits, index = square * 73 + move_type (chess_rl.encoding.moves)
value   conv1x1 -> FC -> (N, 3) win/draw/loss logits for the side to move
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from chess_rl.encoding.board import NUM_PLANES
from chess_rl.encoding.moves import NUM_MOVE_TYPES, NUM_MOVES


@dataclass(frozen=True)
class NetConfig:
    blocks: int = 10
    filters: int = 128
    se: bool = True  # squeeze-excitation in each block
    se_ratio: int = 4
    value_hidden: int = 128


class SqueezeExcite(nn.Module):
    """Channel gating with scale and bias, as in Lc0's SE blocks."""

    def __init__(self, channels: int, ratio: int):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // ratio)
        self.fc2 = nn.Linear(channels // ratio, 2 * channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3))
        w, b = self.fc2(F.relu(self.fc1(s))).chunk(2, dim=1)
        return x * torch.sigmoid(w)[:, :, None, None] + b[:, :, None, None]


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, se: bool, se_ratio: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcite(channels, se_ratio) if se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.se(self.bn2(self.conv2(y)))
        return F.relu(x + y)


class ChessNet(nn.Module):
    def __init__(self, config: NetConfig = NetConfig()):
        super().__init__()
        self.config = config
        c = config.filters
        self.stem = nn.Sequential(nn.Conv2d(NUM_PLANES, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU())
        self.tower = nn.Sequential(*[ResidualBlock(c, config.se, config.se_ratio) for _ in range(config.blocks)])
        self.policy_conv = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU())
        self.policy_out = nn.Conv2d(c, NUM_MOVE_TYPES, 3, padding=1)
        self.value_conv = nn.Sequential(nn.Conv2d(c, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.value_fc = nn.Sequential(
            nn.Linear(32 * 64, config.value_hidden), nn.ReLU(), nn.Linear(config.value_hidden, 3)
        )

    def forward(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tower(self.stem(planes))
        p = self.policy_out(self.policy_conv(x))  # (N, 73, 8, 8): channel = move type, (row, col) = square
        policy = p.permute(0, 2, 3, 1).reshape(-1, NUM_MOVES)  # square = row * 8 + col, index = square*73 + type
        value = self.value_fc(self.value_conv(x).flatten(1))
        return policy, value


def to_device(model: ChessNet, device: str | torch.device) -> ChessNet:
    """Move to device; on CUDA use channels_last memory format (20-30% faster on the RTX 3050)."""
    model = model.to(device)
    if torch.device(device).type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model


def prepare_input(planes: torch.Tensor) -> torch.Tensor:
    """Match the model's memory format for CUDA inputs."""
    return planes.contiguous(memory_format=torch.channels_last) if planes.is_cuda else planes


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Log-probabilities over legal moves only (mask: bool, same shape as logits)."""
    return torch.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)


def expected_score(value_logits: torch.Tensor) -> torch.Tensor:
    """WDL logits -> expected score in [0, 1] for the side to move (index 0 = loss, 1 = draw, 2 = win)."""
    p = torch.softmax(value_logits.float(), dim=-1)
    return p[..., 2] + 0.5 * p[..., 1]


# ---- checkpoints ---------------------------------------------------------------------------------


def save_checkpoint(path: str | Path, model: ChessNet, **meta: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[ChessNet, dict[str, Any]]:
    data = torch.load(path, map_location=device, weights_only=True)
    model = to_device(ChessNet(NetConfig(**data["config"])), device)
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model, data.get("meta", {})


RELEASE_META_KEYS = ("step", "positions_seen", "best_val")


def export_release(src: str | Path, dst: str | Path) -> dict[str, Any]:
    """Write a checkpoint fit for publishing: weights, architecture, step, positions seen and best validation loss.

    Training checkpoints also hold the optimizer state and the training config, whose data/output paths
    may be absolute local paths. The release file loads with `load_checkpoint` like any other.
    (This replaces a TorchScript export: torch.jit.trace is deprecated, both TorchScript and torch.export
    files embed absolute source paths, and neither ran faster than eager at MCTS batch sizes.)
    """
    data = torch.load(src, map_location="cpu", weights_only=True)
    state = data.get("meta", {}).get("train_state", {})
    meta = {k: state[k] for k in RELEASE_META_KEYS if k in state}
    torch.save({"config": data["config"], "state_dict": data["state_dict"], "meta": meta}, Path(dst))
    return meta
