import time

import chess
import numpy as np
import pytest
import torch

from chess_rl.encoding import NUM_MOVES, NUM_PLANES, encode_board, legal_mask, move_to_index
from chess_rl.eval import RandomPlayer, play_game
from chess_rl.eval.registry import make_player
from chess_rl.model.network import (
    ChessNet,
    NetConfig,
    count_parameters,
    expected_score,
    export_torchscript,
    load_checkpoint,
    masked_log_softmax,
    prepare_input,
    save_checkpoint,
    to_device,
)
from chess_rl.model.player import NetPlayer

SMALL = NetConfig(blocks=2, filters=32)


def test_shapes_and_param_count():
    net = ChessNet(SMALL).eval()
    policy, value = net(torch.zeros(4, NUM_PLANES, 8, 8))
    assert policy.shape == (4, NUM_MOVES) and value.shape == (4, 3)
    assert count_parameters(ChessNet(NetConfig())) > count_parameters(net)


def test_policy_index_layout_matches_move_encoding():
    """A bias on one (move type, square) output must land at index square * 73 + type."""
    net = ChessNet(SMALL).eval()
    with torch.no_grad():
        net.policy_out.weight.zero_()
        net.policy_out.bias.zero_()
        net.policy_out.bias[1] = 10.0  # move type 1: north, distance 2
    board = chess.Board()
    with torch.no_grad():
        logits, _ = net(torch.from_numpy(encode_board(board)).unsqueeze(0))
    e2e4 = move_to_index(chess.Move.from_uci("e2e4"), chess.WHITE)
    assert logits[0, e2e4].item() == pytest.approx(10.0)  # every square gets the type-1 bias, e2's included
    assert logits[0, e2e4 - 1].item() == pytest.approx(0.0)  # type 0 at e2
    assert logits[0, chess.E4 * 73 + 1].item() == pytest.approx(10.0) and logits[0].max().item() == pytest.approx(10.0)


def test_masked_log_softmax_only_legal():
    board = chess.Board()
    mask = torch.from_numpy(legal_mask(board)).unsqueeze(0)
    logp = masked_log_softmax(torch.randn(1, NUM_MOVES), mask)
    assert torch.isinf(logp[~mask]).all()
    assert torch.exp(logp[mask]).sum() == pytest.approx(1.0, abs=1e-5)


def test_expected_score():
    assert expected_score(torch.tensor([[0.0, 0.0, 50.0]])).item() == pytest.approx(1.0)
    assert expected_score(torch.tensor([[0.0, 50.0, 0.0]])).item() == pytest.approx(0.5)


def test_checkpoint_round_trip_and_torchscript(tmp_path):
    net = ChessNet(SMALL).eval()
    save_checkpoint(tmp_path / "m.pt", net, step=7)
    loaded, meta = load_checkpoint(tmp_path / "m.pt")
    x = torch.randn(3, NUM_PLANES, 8, 8)
    with torch.no_grad():
        for a, b in zip(net(x), loaded(x), strict=True):
            torch.testing.assert_close(a, b)
    assert meta == {"step": 7} and loaded.config == SMALL
    export_torchscript(net, tmp_path / "m.ts")
    scripted = torch.jit.load(str(tmp_path / "m.ts"))
    with torch.no_grad():
        torch.testing.assert_close(scripted(x)[0], net(x)[0])


def test_net_player_plays_legal_games(tmp_path):
    save_checkpoint(tmp_path / "m.pt", ChessNet(SMALL), step=0)
    player = make_player(f"net:path={tmp_path / 'm.pt'},device=cpu")
    game = play_game(player, RandomPlayer(0), max_plies=60)
    assert game.termination != "illegal_move"
    sampler = NetPlayer(model=ChessNet(SMALL), device="cpu", temperature=1.0, seed=1)
    moves, probs, score = sampler.policy(chess.Board())
    assert len(moves) == 20 and probs.sum().item() == pytest.approx(1.0, abs=1e-5) and 0 <= score <= 1
    assert sampler.choose(chess.Board()) in chess.Board().legal_moves


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_default_net_inference_speed_on_gpu():
    """#21 done-when: batch-256 forward under 15 ms on the target GPU (fp16 autocast)."""
    torch.backends.cudnn.benchmark = True
    net = to_device(ChessNet(NetConfig()), "cuda").eval()  # the production path: channels_last on CUDA
    x = prepare_input(torch.from_numpy(np.random.rand(256, NUM_PLANES, 8, 8).astype(np.float32)).cuda())
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(5):
            net(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            net(x)
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 20 * 1000
    assert ms < 15, f"{ms:.1f} ms"


def test_cpu_net_player_limits_threads():
    before = torch.get_num_threads()
    try:
        NetPlayer(model=ChessNet(SMALL), device="cpu", threads=2)
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(before)
