import chess
import chess.pgn
import gym
import numpy as np
from gym import spaces
from utils.opening_book import OpeningBook
import random

class ChessEnvWithOpeningBook(gym.Env):
    """
    Chess environment with opening book and curriculum learning capabilities.
    """
    def __init__(self, use_opening_book=True, opening_book_moves=10, opening_bonus=0.5, 
                 curriculum_level=0, start_position=None):
        """
        Chess environment with opening book knowledge
        
        Args:
            use_opening_book: Whether to use the opening book
            opening_book_moves: Maximum number of moves to follow the opening book
            opening_bonus: Reward bonus for following the opening book
            curriculum_level: Level of difficulty for curriculum learning
            start_position: Optional FEN string for custom start position
        """
        super().__init__()
        
        # Curriculum learning parameters
        self.curriculum_level = curriculum_level  # 0=full game, 1=midgame, 2=tactical endgame, 3=puzzle
        self.start_position = start_position  # Optional FEN string for custom start position
        
        # Set up the board based on curriculum level
        self.setup_curriculum()
        
        # Opening book settings
        self.use_opening_book = use_opening_book and self.curriculum_level == 0  # Only use opening book in full games
        self.opening_book_moves = opening_book_moves
        self.opening_bonus = opening_bonus
        self.followed_book_move = False
        
        if self.use_opening_book:
            from src.chess_opening_book import OpeningBook
            self.opening_book = OpeningBook()
        
        # State tracking for self-play and rewards
        self.legal_moves_list = list(self.board.legal_moves)
        self.move_count = 0
        self.position_history = []
        self.invalid_moves_count = 0
        self.last_material_balance = self.calculate_material_balance()
        self.material_history = [self.last_material_balance]
        
        # Set up the spaces
        # 12 piece types (6 white, 6 black) on an 8x8 board
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(12, 8, 8), dtype=np.float32
        )
        
        # Action space is the maximum number of legal moves in chess (~218)
        self.action_space = spaces.Discrete(218)
        
        # Caches for performance
        self._white_attacks = set()
        self._black_attacks = set()
    
    def setup_curriculum(self):
        """Set up board based on curriculum level"""
        if self.start_position:
            # Use custom FEN position if provided
            self.board = chess.Board(self.start_position)
        elif self.curriculum_level == 0:
            # Full game - start from standard position
            self.board = chess.Board()
        elif self.curriculum_level == 1:
            # Midgame positions - start from common midgame positions
            midgame_positions = [
                # Open game with development
                "r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
                # Semi-open game
                "rnbqkb1r/pp3ppp/2p1pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 6",
                # Closed position
                "r1bqk2r/ppp2ppp/2n1pn2/3p4/1bPP4/2NBP3/PP3PPP/R1BQK1NR w KQkq - 0 7",
                # Open with early queen exchange
                "rnb1kbnr/pppp1ppp/8/4p3/3P4/8/PPP2PPP/RNBQKBNR w KQkq - 0 4",
            ]
            self.board = chess.Board(random.choice(midgame_positions))
        elif self.curriculum_level == 2:
            # Endgame positions - fewer pieces, clearer objectives
            endgame_positions = [
                # Rook endgame
                "8/5pk1/7p/8/5P2/5K2/7P/4R3 w - - 0 1",
                # Pawn endgame
                "8/5p2/5k2/8/5P2/8/5K2/8 w - - 0 1",
                # Queen vs rook endgame
                "8/8/5k2/8/8/3Q4/5K2/7r w - - 0 1",
                # Bishop and knight checkmate
                "8/8/5k2/8/8/3B4/5K2/5N2 w - - 0 1"
            ]
            self.board = chess.Board(random.choice(endgame_positions))
        elif self.curriculum_level == 3:
            # Tactical puzzles - positions with forced tactical solutions
            tactical_positions = [
                # Fork opportunity
                "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/3P1P2/PPP3PP/RNBQKBNR w KQkq - 0 1",
                # Pin opportunity
                "r1bqkb1r/ppp2ppp/2n2n2/3pp3/4P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 0 1",
                # Discovered attack
                "r1bqkb1r/ppp2ppp/2n2n2/3p4/3P4/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 1",
                # Mate in 1
                "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1"
            ]
            self.board = chess.Board(random.choice(tactical_positions))
        
        # Update legal moves list after setting up the board
        self.legal_moves_list = list(self.board.legal_moves)
    
    def reset(self):
        """Reset the environment to start a new game, using curriculum if specified"""
        self.setup_curriculum()
        
        self.move_count = 0
        self.position_history = []
        self.invalid_moves_count = 0
        self.last_material_balance = self.calculate_material_balance()
        self.material_history = [self.last_material_balance]
        
        self.legal_moves_list = list(self.board.legal_moves)
        
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
        
        # Cache which squares are attacked by each side for faster reward calculation
        self._white_attacks = set()
        self._black_attacks = set()
        for square in chess.SQUARES:
            if self.board.is_attacked_by(chess.WHITE, square):
                self._white_attacks.add(square)
            if self.board.is_attacked_by(chess.BLACK, square):
                self._black_attacks.add(square)
        
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
        rewards["valid_move"] = 0.2  # Increased from 0.1
        
        # 2. Material change reward (normalized to be between -1 and 1 for most cases)
        material_change = material_after - material_before
        rewards["material"] = material_change * 0.2  # Doubled from 0.1 to amplify material gains
        
        # 3. Opening book bonus (if move follows the opening book)
        if self.followed_book_move:
            rewards["opening_book"] = self.opening_bonus * 1.5  # Increased to encourage book moves
        else:
            rewards["opening_book"] = 0.0
        
        # 4. Checkmate (large reward/penalty)
        if self.board.is_checkmate():
            rewards["checkmate"] = 10.0 if not self.board.turn else -5.0  # Doubled checkmate reward
            
        # 5. Check (small reward)
        rewards["check"] = 0.3 if self.board.is_check() else 0  # Increased from 0.2
            
        # 6. Stalemate is slightly negative
        if self.board.is_stalemate():
            rewards["stalemate"] = -0.5  # Reduced from -1.0
            
        # 7. Center control reward (scaled up)
        center_control = 0
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        current_color = not self.board.turn  # Our color (just moved)
        
        for square in center_squares:
            piece = self.board.piece_at(square)
            if piece is not None and piece.color == current_color:
                center_control += 0.1  # Doubled from 0.05
        rewards["center_control"] = center_control
        
        # 8. Development reward (early game)
        development = 0
        if self.move_count < 10:  # Early game
            # Knights and bishops developed
            if current_color == chess.WHITE:
                development_squares = [chess.B1, chess.G1, chess.C1, chess.F1]  # White starting positions
            else:
                development_squares = [chess.B8, chess.G8, chess.C8, chess.F8]  # Black starting positions
                
            for square in development_squares:
                piece = self.board.piece_at(square)
                if piece is None:  # Piece has moved
                    development += 0.1  # Doubled from 0.05
                    
            # Castling (big early bonus)
            if isinstance(move, chess.Move):
                if move.from_square == chess.E1 and move.to_square in [chess.C1, chess.G1]:
                    development += 1.0  # Doubled from 0.5
                elif move.from_square == chess.E8 and move.to_square in [chess.C8, chess.G8]:
                    development += 1.0  # Doubled from 0.5
        rewards["development"] = development
                
        # 9. Repetition penalty (scaled appropriately)
        repetition = 0
        if self.board.is_repetition(2):
            repetition = -0.2  # Reduced from -0.3
        elif self.board.is_repetition(3):
            repetition = -0.4  # Reduced from -0.8
        rewards["repetition"] = repetition
        
        # 10. Piece activity - reward for having more moves available
        mobility = len(self.legal_moves_list) / 30.0  # Normalize by dividing by typical move count
        rewards["mobility"] = min(0.8, mobility * 0.2)  # Doubled cap and scaling
        
        # 11. Pawn structure - reward for protected pawns
        pawn_structure = 0
        current_attacks = self._white_attacks if current_color == chess.WHITE else self._black_attacks
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece and piece.color == current_color and piece.piece_type == chess.PAWN:
                if square in current_attacks:  # Pawn is protected
                    pawn_structure += 0.04  # Doubled from 0.02
        rewards["pawn_structure"] = pawn_structure

        # 12. Tactical capture - reward for capturing undefended pieces
        if self.board.is_capture(move):  # This move captured a piece
            captured_square = move.to_square
            captured_piece_type = self.board.piece_type_at(captured_square)
            
            # We need to temporarily undo the move to check if the captured piece was defended
            self.board.pop()
            opponent_color = not current_color
            opponent_attacks = self._black_attacks if current_color == chess.WHITE else self._white_attacks
            was_defended = move.to_square in opponent_attacks
            
            # Get the captured piece value
            captured_piece = self.board.piece_at(move.to_square)
            captured_value = self.piece_value(captured_piece) if captured_piece else 0
            
            # Redo the move
            self.board.push(move)
            
            if not was_defended and captured_value > 0:
                # Extra reward for capturing undefended piece
                rewards["tactical_capture"] = captured_value * 0.6  # Doubled from 0.3
            else:
                # Still give a small reward for any capture
                rewards["capture"] = captured_value * 0.2  # New reward for any capture
                
        # 13. Defense reward - reward for defending threatened pieces
        defense_reward = 0
        opponent_color = self.board.turn  # Opponent's color (about to move)
        opponent_attacks = self._black_attacks if opponent_color == chess.BLACK else self._white_attacks
        our_attacks = self._white_attacks if current_color == chess.WHITE else self._black_attacks
        
        # Find our pieces that are under attack by opponent
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece and piece.color == current_color and square in opponent_attacks:
                # This piece is threatened
                if square in our_attacks:
                    # And it's defended by us
                    defense_reward += self.piece_value(piece) * 0.2  # Doubled from 0.1
        
        rewards["defense"] = defense_reward
        
        # 14. Hanging pieces penalty - penalize having hanging pieces (but reduce penalty)
        hanging_penalty = 0
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:  # Our piece (about to move)
                in_opponent_attacks = square in (self._white_attacks if current_color == chess.BLACK else self._black_attacks)
                in_our_attacks = square in (self._black_attacks if current_color == chess.BLACK else self._white_attacks)
                
                if in_opponent_attacks and not in_our_attacks:
                    # It's hanging! Apply penalty based on piece value
                    piece_value = self.piece_value(piece)
                    hanging_penalty -= piece_value * 0.1  # Reduced from 0.2
        
        rewards["hanging_penalty"] = hanging_penalty
        
        # 15. Add small positive rewards for safe moves (no new pieces hanging)
        if hanging_penalty == 0:
            rewards["safe_move"] = 0.1  # New reward for not hanging pieces
        
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

    # Function to detect hanging pieces
    def detect_hanging_pieces(self):
        hanging_penalty = 0
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:  # Our piece
                # Check if under attack by opponent
                if self.board.is_attacked_by(not self.board.turn, square):
                    # Check if defended by our pieces
                    if not self.board.is_attacked_by(self.board.turn, square):
                        # It's hanging! Apply penalty based on piece value
                        piece_value = self.piece_value(piece)
                        hanging_penalty -= piece_value * 0.2  # Scale based on value
        return hanging_penalty

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