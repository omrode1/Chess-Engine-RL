import chess
import chess.pgn
import random
from typing import Dict, List, Optional, Tuple

class OpeningBook:
    """
    A chess opening book that provides knowledge of standard openings.
    This can be used to guide the agent during the opening phase of the game.
    """
    def __init__(self, use_weighted_selection: bool = True):
        """
        Initialize the opening book with a collection of popular openings.
        
        Args:
            use_weighted_selection: If True, openings will be selected based on weights.
                                   Otherwise, they'll be selected uniformly at random.
        """
        self.use_weighted_selection = use_weighted_selection
        
        # Dictionary to store opening moves
        # Format: fen string (position) -> list of (move, weight) tuples
        self.book: Dict[str, List[Tuple[chess.Move, float]]] = {}
        
        # Populate the opening book with standard openings
        self._populate_opening_book()
    
    def _populate_opening_book(self):
        """Populate the opening book with popular chess openings"""
        # Ruy Lopez / Spanish Opening
        self._add_opening([
            "e2e4", "e7e5",  # 1. e4 e5
            "g1f3", "b8c6",  # 2. Nf3 Nc6
            "f1b5"           # 3. Bb5 (Spanish)
        ], weight=1.0)
        
        # Italian Game
        self._add_opening([
            "e2e4", "e7e5",  # 1. e4 e5
            "g1f3", "b8c6",  # 2. Nf3 Nc6
            "f1c4"           # 3. Bc4 (Italian)
        ], weight=1.0)
        
        # Sicilian Defense
        self._add_opening([
            "e2e4", "c7c5",  # 1. e4 c5
            "g1f3", "d7d6",  # 2. Nf3 d6
            "d2d4", "c5d4",  # 3. d4 cxd4
            "f3d4"           # 4. Nxd4
        ], weight=1.0)
        
        # French Defense
        self._add_opening([
            "e2e4", "e7e6",  # 1. e4 e6
            "d2d4", "d7d5",  # 2. d4 d5
        ], weight=0.8)
        
        # Queen's Gambit
        self._add_opening([
            "d2d4", "d7d5",  # 1. d4 d5
            "c2c4"           # 2. c4 (Queen's Gambit)
        ], weight=0.9)
        
        # King's Indian Defense
        self._add_opening([
            "d2d4", "g8f6",  # 1. d4 Nf6
            "c2c4", "g7g6",  # 2. c4 g6
            "b1c3", "f8g7",  # 3. Nc3 Bg7
        ], weight=0.7)
        
        # English Opening
        self._add_opening([
            "c2c4"           # 1. c4
        ], weight=0.6)
        
        # Caro-Kann Defense
        self._add_opening([
            "e2e4", "c7c6",  # 1. e4 c6
            "d2d4", "d7d5",  # 2. d4 d5
        ], weight=0.7)
        
        # Pirc Defense
        self._add_opening([
            "e2e4", "d7d6",  # 1. e4 d6
            "d2d4", "g8f6",  # 2. d4 Nf6
            "b1c3", "g7g6",  # 3. Nc3 g6
        ], weight=0.6)
        
        # King's Pawn Opening (Various)
        self._add_opening([
            "e2e4"           # 1. e4
        ], weight=1.0)
        
        # Queen's Pawn Opening (Various)
        self._add_opening([
            "d2d4"           # 1. d4
        ], weight=0.9)
        
        # Scandinavian Defense
        self._add_opening([
            "e2e4", "d7d5"   # 1. e4 d5
        ], weight=0.5)
        
        # London System
        self._add_opening([
            "d2d4", "d7d5",  # 1. d4 d5
            "c1f4"           # 2. Bf4
        ], weight=0.7)
        
        # Vienna Game
        self._add_opening([
            "e2e4", "e7e5",  # 1. e4 e5
            "b1c3"           # 2. Nc3
        ], weight=0.6)
        
        # Center Game
        self._add_opening([
            "e2e4", "e7e5",  # 1. e4 e5
            "d2d4", "e5d4"   # 2. d4 exd4
        ], weight=0.5)
        
        # Evans Gambit
        self._add_opening([
            "e2e4", "e7e5",  # 1. e4 e5
            "g1f3", "b8c6",  # 2. Nf3 Nc6
            "f1c4", "f8c5",  # 3. Bc4 Bc5
            "b2b4"           # 4. b4 (Evans Gambit)
        ], weight=0.6)
        
        # King's Gambit
        self._add_opening([
            "e2e4", "e7e5",  # 1. e4 e5
            "f2f4"           # 2. f4 (King's Gambit)
        ], weight=0.5)
    
    def _add_opening(self, uci_moves: List[str], weight: float = 1.0):
        """
        Add an opening to the book using a list of UCI move strings.
        
        Args:
            uci_moves: List of moves in UCI format (e.g., "e2e4")
            weight: The weight/probability of selecting this line
        """
        board = chess.Board()
        
        # Apply each move and add the resulting position to the book
        for i, uci in enumerate(uci_moves):
            # Get the position before the move
            fen_before = board.fen().split(' ')[0]  # Just the piece positions
            
            # Convert UCI string to a move object
            move = chess.Move.from_uci(uci)
            
            # Add to opening book if this is a legal move
            if move in board.legal_moves:
                # Initialize the list if needed
                if fen_before not in self.book:
                    self.book[fen_before] = []
                
                # Add the move with its weight
                self.book[fen_before].append((move, weight))
                
                # Apply the move to the board
                board.push(move)
            else:
                # Stop if we encounter an illegal move in the opening
                break
    
    def get_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Get a move from the opening book for the current position.
        
        Args:
            board: The current chess board position
            
        Returns:
            A chess move from the opening book, or None if the position is not in the book
        """
        # Get just the piece positions from FEN (ignore counters, castling, etc.)
        position = board.fen().split(' ')[0]
        
        # Check if the position is in our opening book
        if position in self.book:
            moves = self.book[position]
            
            if self.use_weighted_selection and len(moves) > 1:
                # Weighted random selection
                total_weight = sum(weight for _, weight in moves)
                rand_val = random.uniform(0, total_weight)
                
                cumulative_weight = 0
                for move, weight in moves:
                    cumulative_weight += weight
                    if rand_val <= cumulative_weight:
                        return move
                
                # Fallback - should not reach here if weights are positive
                return moves[0][0]
            else:
                # Simple random selection
                return random.choice([move for move, _ in moves])
        
        # Position not in book
        return None

# Example usage
if __name__ == "__main__":
    # Test the opening book
    book = OpeningBook()
    board = chess.Board()
    
    print("Testing opening book with a new game:")
    
    for _ in range(10):  # Try up to 10 moves from the book
        book_move = book.get_move(board)
        
        if book_move:
            print(f"Position: {board.fen()}")
            print(f"Book suggests: {book_move.uci()}")
            board.push(book_move)
        else:
            print(f"Position not found in opening book: {board.fen()}")
            break
    
    print(f"Final position after opening: {board.fen()}")
    print(f"Number of positions in opening book: {len(book.book)}") 