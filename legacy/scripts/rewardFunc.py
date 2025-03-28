import chess

# Simple reward function
def get_reward(board, move):
    prev_material = sum([piece_value(p) for p in board.piece_map().values()])
    
    board.push(move)  # Make the move
    new_material = sum([piece_value(p) for p in board.piece_map().values()])
    
    reward = new_material - prev_material  # Material gain/loss

    # Check for checkmate
    if board.is_checkmate():
        reward += 5

    board.pop()  # Undo the move
    return reward

# Piece values for material evaluation
def piece_value(piece):
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
              chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    return values.get(piece.piece_type, 0)

# Test reward function
board = chess.Board()
move = chess.Move.from_uci("e2e4")  # Test move
reward = get_reward(board, move)

print(f"Move: {move}, Reward: {reward}")
