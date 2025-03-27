import chess

# Create a new chessboard
board = chess.Board()

# Print the initial board state
print("Initial Board:")
print(board)

# Print FEN representation
print("\nFEN Representation:")
print(board.fen())

# Make a move (e2 to e4)
move = chess.Move.from_uci("e2e4")  # UCI notation
board.push(move)

# Print updated board state
print("\nBoard After e2e4:")
print(board)
