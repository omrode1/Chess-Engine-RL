import math

import chess
import numpy as np
import pytest

from chess_rl.encoding import NUM_MOVES
from chess_rl.search.mcts import MCTS, Node, SearchConfig, _terminal_value


def uniform(planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equal priors, value 0: pure search, no chess knowledge."""
    return np.zeros((len(planes), NUM_MOVES), np.float32), np.zeros(len(planes), np.float32)


class MaterialEval:
    """Equal priors; value = tanh(material / 3) for the side to move, read back from the planes."""

    def __init__(self) -> None:
        self.batches: list[int] = []

    def __call__(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.batches.append(len(planes))
        weights = np.array([1, 3, 3, 5, 9, 0], np.float32)
        ours = planes[:, 0:6].sum(axis=(2, 3)) @ weights
        theirs = planes[:, 6:12].sum(axis=(2, 3)) @ weights
        return np.zeros((len(planes), NUM_MOVES), np.float32), np.tanh((ours - theirs) / 3).astype(np.float32)


def walk(node: Node):
    yield node
    for child in node.children:
        if child is not None:
            yield from walk(child)


def test_finds_mate_in_one_without_knowledge():
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")  # Ra8#
    result = MCTS(uniform).search(board, nodes=2000)
    assert result.move == chess.Move.from_uci("a1a8")
    assert result.proven == 1.0 and result.q == 1.0
    assert result.nodes < 2000  # a proven root stops the search early


@pytest.mark.parametrize(
    ("fen", "solution"),
    [  # from the mate2 suite (Lichess puzzles); the last one starts with a quiet king move
        ("4rk2/p4q2/1p3Q1b/8/1p5N/2P1p3/P3P3/2K5 w - - 0 1", "Ng6+"),
        ("3rk2r/2qn2p1/p1Q1p3/3n3p/8/8/PP4PP/5R1K w k - 0 1", "Qxe6+"),
        ("r4r2/2q1Nb2/5Qpk/2n4p/pp5P/8/1PP2PP1/2KR3R w - - 0 1", "Nf5+"),
        ("8/1P6/8/7p/3P4/3k1p1P/4p3/4K3 b - - 0 1", "Ke3"),
    ],
)
def test_proves_mate_in_two(fen, solution):
    board = chess.Board(fen)
    result = MCTS(MaterialEval()).search(board, nodes=6000)
    assert result.proven == 1.0 and board.san(result.move) == solution


def opponent_can_mate(board: chess.Board) -> bool:
    for reply in list(board.legal_moves):
        board.push(reply)
        mate = board.is_checkmate()
        board.pop()
        if mate:
            return True
    return False


def test_does_not_walk_into_mate():
    # Black to move with an open back rank: the rook must not leave the defence of e8 (Re8#)
    board = chess.Board("6k1/5ppp/8/8/8/8/r4PPP/4R1K1 b - - 0 1")
    board.push(MCTS(MaterialEval()).search(board, nodes=3000).move)
    assert not opponent_can_mate(board)


def test_takes_hanging_queen_and_saves_its_own():
    board = chess.Board("4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1")
    assert MCTS(MaterialEval()).search(board, nodes=400).move == chess.Move.from_uci("d2d5")
    board = chess.Board("4k3/8/8/2p1p3/3Q4/8/8/4K3 w - - 0 1")  # Qd4 attacked by two pawns
    board.push(MCTS(MaterialEval()).search(board, nodes=800).move)
    queen = board.pieces(chess.QUEEN, chess.WHITE)
    assert queen and not any(board.is_attacked_by(chess.BLACK, sq) for sq in queen)


def test_statistics_are_consistent_after_search():
    evaluator = MaterialEval()
    search = MCTS(evaluator, SearchConfig(batch_size=16))
    result = search.search(chess.Board(), nodes=1500)
    root = search._root
    assert max(evaluator.batches) <= 16
    assert sum(result.visits.values()) == root.n_sum
    for node in walk(root):
        assert not node.pending
        if node.moves is not None:
            assert node.n_sum == int(node.N.sum())
            assert np.all(np.abs(node.W) <= node.N + 1e-9)  # no virtual loss left behind
        if node.visits:
            assert -1 - 1e-9 <= node.q <= 1 + 1e-9


def test_tree_is_reused_across_moves():
    search = MCTS(MaterialEval())
    board = chess.Board()
    first = search.search(board, nodes=1000)
    board.push(first.move)
    board.push(first.pv[1] if len(first.pv) > 1 else next(iter(board.legal_moves)))
    second = search.search(board, nodes=200)
    assert first.reused == 0 and second.reused > 0
    # an unrelated position starts fresh
    assert search.search(chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"), nodes=50).reused == 0


def test_forced_move_returns_without_search():
    board = chess.Board("7k/6R1/8/8/8/8/8/K7 b - - 0 1")  # the only legal move is Kxg7
    result = MCTS(uniform).search(board, nodes=1000)
    assert result.move == chess.Move.from_uci("h8g7") and result.nodes == 0


def test_terminal_values():
    mated = chess.Board("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1")
    assert _terminal_value(mated, list(mated.legal_moves)) == -1.0
    stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert _terminal_value(stalemate, list(stalemate.legal_moves)) == 0.0
    board = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):  # back to the start: a twofold repetition
        board.push_uci(uci)
    assert _terminal_value(board, list(board.legal_moves)) == 0.0
    assert _terminal_value(chess.Board(), list(chess.Board().legal_moves)) is None


def test_needs_a_limit():
    with pytest.raises(ValueError):
        MCTS(uniform).search(chess.Board())


def test_movetime_limit():
    result = MCTS(uniform).search(chess.Board(), movetime=0.2)
    assert result.seconds < 1.0 and result.nodes > 0 and math.isfinite(result.nps)


def test_mcts_player_from_spec(tmp_path):
    from chess_rl.eval.registry import make_player
    from chess_rl.model.network import ChessNet, NetConfig, save_checkpoint

    path = tmp_path / "tiny.pt"
    save_checkpoint(path, ChessNet(NetConfig(blocks=1, filters=8, se=False)))
    player = make_player(f"mcts:path={path},nodes=64,device=cpu,batch=8,c_puct=2.0")
    board = chess.Board()
    player.new_game()
    for _ in range(4):
        move = player.choose(board)
        assert move in board.legal_moves
        board.push(move)
    assert player.search.config.c_puct == 2.0 and player.last.nodes >= 64
    with pytest.raises(ValueError):
        make_player(f"mcts:path={path},bogus=1")
