import chess
import numpy as np

# Define piece-to-channel mapping
piece_to_index = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3,
    chess.QUEEN: 4, chess.KING: 5
}

# Function to convert board state to One-Hot Encoding (8x8x12)
def board_to_tensor(board):
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color_offset = 6 if piece.color == chess.BLACK else 0
            piece_index = piece_to_index[piece.piece_type] + color_offset
            row, col = divmod(square, 8)
            tensor[piece_index, row, col] = 1

    return tensor

# Create a new board and make a move
board = chess.Board()
tensor = board_to_tensor(board)

# Print the tensor shape
print("Tensor Shape:", tensor.shape)  # Should be (12, 8, 8)
