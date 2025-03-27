import chess
import gym
import numpy as np
from gym import spaces

class ChessEnv(gym.Env):
    def __init__(self):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()

        # Observation Space (12x8x8 board representation)
        self.observation_space = spaces.Box(low=0, high=1, shape=(12, 8, 8), dtype=np.float32)

        # Action Space - maximum reasonable number of moves in any position is ~218
        self.action_space = spaces.Discrete(218)
        
        # Cache for legal moves (updated on reset and step)
        self.legal_moves_list = list(self.board.legal_moves)

    def reset(self):
        """Reset the board to the starting position"""
        self.board.reset()
        self.legal_moves_list = list(self.board.legal_moves)
        return self.board_to_tensor()

    def step(self, action):
        """Apply an action and return new state, reward, done, and info"""
        # Check if action is valid
        if action < 0 or action >= len(self.legal_moves_list):
            return self.board_to_tensor(), -1.0, False, {}
            
        # Get the move from our current legal moves
        move = self.legal_moves_list[action]
        
        # Store material before move for reward calculation
        material_before = self.calculate_material_balance()
        
        # Execute the move
        self.board.push(move)
        
        # Update legal moves for next step
        self.legal_moves_list = list(self.board.legal_moves)
        
        # Evaluate position after move
        material_after = self.calculate_material_balance()
        reward = self.calculate_reward(material_before, material_after)
        done = self.board.is_game_over()
        
        return self.board_to_tensor(), reward, done, {}

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

    def calculate_reward(self, material_before, material_after):
        """Calculate reward based on the current board state"""
        # Material change (positive if we gained material)
        material_change = material_after - material_before
        
        # Check for checkmate
        if self.board.is_checkmate():
            return 10.0 if not self.board.turn else -10.0
            
        # Stalemate is slightly negative
        if self.board.is_stalemate():
            return -2.0
        
        # Center control reward (d4, d5, e4, e5)
        center_control_reward = 0
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        for square in center_squares:
            piece = self.board.piece_at(square)
            if piece is not None:
                center_control_reward += 0.2


        # if self.board.piece_at(chess.E4) is not None:
        #     center_control_reward += 0.1
        # if self.board.piece_at(chess.D4) is not None:
        #     center_control_reward += 0.1
        # if self.board.piece_at(chess.E5) is not None:
        #     center_control_reward += 0.1
        # if self.board.piece_at(chess.D5) is not None:
        #     center_control_reward += 0.1
       

       

        #penalty for early side pawn movement
        early_side_pawn_penalty = 0
        if len(self.board.move_stack) < 10:
            side_pawns = [chess.A4, chess.H4, chess.A3, chess.H3]
            for move in self.board.move_stack:
                if move.to_square in side_pawns:
                    early_side_pawn_penalty -= 0.3

        # Penalty for repetition (using python-chess's built-in detection)
        repetition_penalty = 0
        if len(self.board.move_stack) > 4:  # Need at least a few moves to have repetition
            # Check if the current position has occurred before
            if self.board.is_repetition(2):  # Position has occurred twice
                repetition_penalty = -0.3
            elif self.board.is_repetition(3):  # Position has occurred three times
                repetition_penalty = -0.6  # Stronger penalty for more repetitions
                
        # Penalty for moves that don't make progress (moving back and forth)
        move_oscillation_penalty = 0
        if len(self.board.move_stack) >= 4:
            # Check if the last 4 moves involve the same pieces moving back and forth
            last_moves = self.board.move_stack[-4:]
            if (last_moves[0].from_square == last_moves[2].to_square and 
                last_moves[0].to_square == last_moves[2].from_square and
                last_moves[1].from_square == last_moves[3].to_square and
                last_moves[1].to_square == last_moves[3].from_square):
                move_oscillation_penalty = -1.0  # Penalty for going back and forth
            
        # Reward for castling
        castling_reward = 0
        if len(self.board.move_stack) > 0:
            last_move = self.board.move_stack[-1]
            # Check if king moved two squares (indicates castling)
            if last_move.from_square == chess.E1 and last_move.to_square in [chess.C1, chess.G1]:
                castling_reward = 1.0 # Castling for white
            elif last_move.from_square == chess.E8 and last_move.to_square in [chess.C8, chess.G8]:
                castling_reward = 1.0  # Castling for black

        # Reward early development (knights and bishops)
        development_reward = 0
        if self.board.fullmove_number < 5:  # Early game (first 5 moves)
            for square in [chess.B1, chess.G1, chess.C1, chess.F1]:  # Starting squares for knights and bishops
                piece = self.board.piece_at(square)
                if piece is None:  # Piece has moved from starting position
                    if square in [chess.B1, chess.G1]:  # Knight squares
                        development_reward += 0.2
                    else:  # Bishop squares 
                        development_reward += 0.2

        # Small reward for checks
        check_reward = 0.1 if self.board.is_check() else 0
        
        # Small reward for developing pieces (moves that aren't pawn or king in the opening)
        opening_development_reward = 0
        if len(self.board.move_stack) < 10:
            last_move = self.board.peek()
            from_piece = self.board.piece_at(last_move.from_square)
            if from_piece and from_piece.piece_type != chess.PAWN and from_piece.piece_type != chess.KING:
                opening_development_reward = 0.05
        
        # Penalty for moving the same piece multiple times in the opening
        early_piece_movement_penalty = 0
        if len(self.board.move_stack) < 10:
            # Count how many times each piece type has moved
            piece_moves = {}
            for move in self.board.move_stack[-3:]:  # Look at last 3 moves
                from_sq = move.from_square
                if from_sq in piece_moves:
                    piece_moves[from_sq] += 1
                else:
                    piece_moves[from_sq] = 1
            
            # Penalize moving the same piece multiple times early
            for square, count in piece_moves.items():
                if count > 1:
                    early_piece_movement_penalty -= 0.1 * (count - 1)
                
        # Combine all rewards
        total_reward = (
            material_change + 
            check_reward + 
            development_reward + 
            center_control_reward + 
            castling_reward + 
            opening_development_reward +
            repetition_penalty +
            move_oscillation_penalty +
            early_piece_movement_penalty
        )
                
        return total_reward

    def piece_value(self, piece):
        """Material value of pieces"""
        values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                  chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        return values.get(piece.piece_type, 0)

# Test the environment
env = ChessEnv()
state = env.reset()

print("Initial Observation Shape:", state.shape)  # Should be (12, 8, 8)
print("Available Actions:", env.action_space.n)  # Should be 218
