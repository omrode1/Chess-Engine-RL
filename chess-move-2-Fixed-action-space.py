import chess

# Get all possible moves from the starting position
board = chess.Board()
legal_moves = list(board.legal_moves)

# Convert legal moves to UCI notation
uci_moves = [move.uci() for move in legal_moves]

print("Legal Moves from Start Position:")
print(uci_moves)
print("\nTotal Legal Moves:", len(uci_moves))
import chess

# Get all possible moves from the starting position
board = chess.Board()
legal_moves = list(board.legal_moves)

# Convert legal moves to UCI notation
uci_moves = [move.uci() for move in legal_moves]

print("Legal Moves from Start Position:")
print(uci_moves)
print("\nTotal Legal Moves:", len(uci_moves))
