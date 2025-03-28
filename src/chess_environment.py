import chess
import chess.pgn
import gym
import numpy as np
from gym import spaces
from utils.opening_book import OpeningBook

class ChessEnvWithOpeningBook(gym.Env):
    def __init__(self, use_opening_book=True, opening_book_moves=10, opening_bonus=0.5):
        """
        Chess environment with opening book knowledge
        
        Args:
            use_opening_book: Whether to use the opening book
            opening_book_moves: Maximum number of moves to follow the opening book
            opening_bonus: Reward bonus for following the opening book
        """
        super(ChessEnvWithOpeningBook, self).__init__()
        self.board = chess.Board()

        # Observation Space (12x8x8 board representation)
        self.observation_space = spaces.Box(low=0, high=1, shape=(12, 8, 8), dtype=np.float32)

        # Action Space - maximum reasonable number of moves in any position is ~218
        self.action_space = spaces.Discrete(218)
        
        # Cache for legal moves (updated on reset and step)
        self.legal_moves_list = list(self.board.legal_moves)
        
        # Tracking variables for improved rewards
        self.move_count = 0
        self.material_history = []
        self.position_history = []  # For tracking repeated positions
        self.invalid_moves_count = 0
        
        # Initialize material balance
        self.last_material_balance = self.calculate_material_balance()
        
        # Opening book parameters
        self.use_opening_book = use_opening_book
        self.opening_book_moves = opening_book_moves
        self.opening_bonus = opening_bonus
        self.opening_book = OpeningBook() if use_opening_book else None
        self.followed_book_move = False

    def reset(self):
        """Reset the board to the starting position"""
        self.board.reset()
        self.legal_moves_list = list(self.board.legal_moves)
        self.move_count = 0
        self.material_history = []
        self.position_history = []
        self.invalid_moves_count = 0
        self.last_material_balance = self.calculate_material_balance()
        self.followed_book_move = False
        
        return self.board_to_tensor()

    def step(self, action):
        """Apply an action and return new state, reward, done, and info"""
        # Check if action is valid
        if action < 0 or action >= len(self.legal_moves_list):
            self.invalid_moves_count += 1
            return self.board_to_tensor(), -0.5, False, {"invalid_move": True}
        
        # Track position before move (for repetition detection)
        position_before = self.board.fen().split(' ')[0]
        self.position_history.append(position_before)
            
        # Get the move from our current legal moves
        move = self.legal_moves_list[action]
        
        # Check if this move follows the opening book
        self.followed_book_move = False
        if self.use_opening_book and self.move_count < self.opening_book_moves:
            book_move = self.opening_book.get_move(self.board)
            if book_move and book_move == move:
                self.followed_book_move = True
        
        # Store material before move for reward calculation
        material_before = self.calculate_material_balance()
        
        # Execute the move
        self.board.push(move)
        self.move_count += 1
        
        # Update legal moves for next step
        self.legal_moves_list = list(self.board.legal_moves)
        
        # Evaluate position after move
        material_after = self.calculate_material_balance()
        self.material_history.append(material_after)
        
        # Calculate reward based on different components
        reward = self.calculate_reward(material_before, material_after, move)
        
        # Update last material balance
        self.last_material_balance = material_after
        
        # Check for game termination
        done = self.board.is_game_over()
        
        # Add win/loss/draw rewards at game end
        if done:
            outcome = self.get_outcome_reward()
            reward += outcome
        
        # Create info dictionary with detailed metrics
        info = {
            "material_balance": material_after,
            "material_change": material_after - material_before,
            "move_count": self.move_count,
            "is_check": self.board.is_check(),
            "is_checkmate": self.board.is_checkmate(),
            "is_stalemate": self.board.is_stalemate(),
            "is_insufficient": self.board.is_insufficient_material(),
            "invalid_moves": self.invalid_moves_count,
            "followed_opening_book": self.followed_book_move
        }
        
        return self.board_to_tensor(), reward, done, info

    def get_hint_from_opening_book(self):
        """Get a hint from the opening book if available"""
        if self.use_opening_book and self.move_count < self.opening_book_moves:
            return self.opening_book.get_move(self.board)
        return None

    def board_to_tensor(self):
        """Convert board to 12x8x8 tensor"""
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        piece_to_index = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3,
                          chess.QUEEN: 4, chess.KING: 5}
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                color_offset = 6 if piece.color == chess.BLACK else 0
                piece_index = piece_to_index[piece.piece_type] + color_offset
                row, col = divmod(square, 8)
                tensor[piece_index, row, col] = 1

        return tensor
        
    def calculate_material_balance(self):
        """Calculate material balance on the board"""
        balance = 0
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                value = self.piece_value(piece)
                if piece.color == chess.WHITE:
                    balance += value
                else:
                    balance -= value
        return balance

    def calculate_reward(self, material_before, material_after, move):
        """Calculate reward based on multiple chess objectives with proper scaling"""
        rewards = {}
        
        # 1. Basic valid move reward (small positive to encourage any valid move)
        rewards["valid_move"] = 0.1
        
        # 2. Material change reward (normalized to be between -1 and 1 for most cases)
        material_change = material_after - material_before
        rewards["material"] = material_change * 0.1  # Scale down material to not overwhelm other rewards
        
        # 3. Opening book bonus (if move follows the opening book)
        if self.followed_book_move:
            rewards["opening_book"] = self.opening_bonus
        else:
            rewards["opening_book"] = 0.0
        
        # 4. Checkmate (large reward/penalty)
        if self.board.is_checkmate():
            rewards["checkmate"] = 5.0 if not self.board.turn else -5.0
            
        # 5. Check (small reward)
        rewards["check"] = 0.2 if self.board.is_check() else 0
            
        # 6. Stalemate is slightly negative
        if self.board.is_stalemate():
            rewards["stalemate"] = -1.0
            
        # 7. Center control reward (scaled down)
        center_control = 0
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        for square in center_squares:
            piece = self.board.piece_at(square)
            if piece is not None and piece.color == (not self.board.turn):  # Check if our piece
                center_control += 0.05
        rewards["center_control"] = center_control
        
        # 8. Development reward (early game)
        development = 0
        if self.move_count < 10:  # Early game
            # Knights and bishops developed
            for square in [chess.B1, chess.G1, chess.C1, chess.F1]:  # White starting positions
                piece = self.board.piece_at(square)
                if piece is None:  # Piece has moved
                    development += 0.05
                    
            # Castling (big early bonus)
            if isinstance(move, chess.Move):
                if move.from_square == chess.E1 and move.to_square in [chess.C1, chess.G1]:
                    development += 0.5  # Castling is a significant achievement
                elif move.from_square == chess.E8 and move.to_square in [chess.C8, chess.G8]:
                    development += 0.5
        rewards["development"] = development
                
        # 9. Repetition penalty (scaled appropriately)
        repetition = 0
        if self.board.is_repetition(2):
            repetition = -0.3
        elif self.board.is_repetition(3):
            repetition = -0.8
        rewards["repetition"] = repetition
        
        # 10. Piece activity - reward for having more moves available
        mobility = len(self.legal_moves_list) / 30.0  # Normalize by dividing by typical move count
        rewards["mobility"] = min(0.5, mobility * 0.1)  # Cap at 0.5 and scale down
        
        # 11. Pawn structure - reward for protected pawns
        pawn_structure = 0
        pawn_squares = [sq for sq in chess.SQUARES if self.board.piece_at(sq) and 
                       self.board.piece_at(sq).piece_type == chess.PAWN]
        for sq in pawn_squares:
            piece = self.board.piece_at(sq)
            if piece and piece.color == (not self.board.turn):  # Our pawns
                # Check if pawn is protected
                if self.board.is_attacked_by(not self.board.turn, sq):
                    pawn_structure += 0.02
        rewards["pawn_structure"] = pawn_structure
                
        # Combine all rewards with appropriate scaling
        total_reward = sum(rewards.values())
        
        # Add debugging info about rewards (optional)
        # print(f"Rewards: {rewards}, Total: {total_reward}")
        
        return total_reward

    def get_outcome_reward(self):
        """Get reward based on game outcome"""
        if not self.board.is_game_over():
            return 0
            
        # Checkmate
        if self.board.is_checkmate():
            return 10.0 if self.board.turn else -10.0  # Win if it's opponent's turn (we checkmated them)
            
        # Draw outcomes (slightly negative to discourage draws)
        return -1.0

    def piece_value(self, piece):
        """Material value of pieces"""
        values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                  chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        return values.get(piece.piece_type, 0)

# Test the environment
if __name__ == "__main__":
    env = ChessEnvWithOpeningBook()
    state = env.reset()
    
    print("Initial Observation Shape:", state.shape)  # Should be (12, 8, 8)
    print("Available Actions:", env.action_space.n)  # Should be 218
    
    # Test with some moves
    done = False
    total_reward = 0
    
    while not done and env.move_count < 20:  # Play up to 20 moves
        # Get a hint from the opening book
        book_move = env.get_hint_from_opening_book()
        
        if book_move:
            # Find the action index for this book move
            try:
                action = env.legal_moves_list.index(book_move)
                print(f"Move {env.move_count + 1}: Following opening book with {book_move.uci()}")
            except ValueError:
                # If book move not in legal_moves_list (shouldn't happen), choose randomly
                action = np.random.randint(len(env.legal_moves_list))
                print(f"Move {env.move_count + 1}: Random move (book move not found)")
        else:
            # Choose a random legal move
            action = np.random.randint(len(env.legal_moves_list))
            print(f"Move {env.move_count + 1}: Random move (no book move)")
        
        # Apply the move
        state, reward, done, info = env.step(action)
        total_reward += reward
        
        # Print information
        print(f"  Board: {env.board}")
        print(f"  Reward: {reward:.2f}, Total: {total_reward:.2f}")
        print(f"  Followed opening book: {info['followed_opening_book']}")
        print()
    
    print(f"Game finished after {env.move_count} moves")
    print(f"Final reward: {total_reward:.2f}")
    print(f"Final board state: {env.board}")
    if env.board.is_game_over():
        if env.board.is_checkmate():
            print("Checkmate!")
        elif env.board.is_stalemate():
            print("Stalemate!")
        else:
            print("Draw!") 