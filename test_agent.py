import chess
import chess.pgn
import gym
import numpy as np
import time
import argparse
import os
from datetime import datetime
from stable_baselines3 import DQN
from src.chess_environment import ChessEnvWithOpeningBook

def parse_args():
    parser = argparse.ArgumentParser(description='Test a trained chess RL agent with opening book knowledge')
    parser.add_argument('--model', type=str, default='models_with_opening_book/chess_with_opening_book_final',
                        help='Path to the trained model')
    parser.add_argument('--games', type=int, default=5, help='Number of games to play')
    parser.add_argument('--max_moves', type=int, default=50, help='Maximum moves per game')
    parser.add_argument('--save_pgn', action='store_true', help='Save games in PGN format')
    parser.add_argument('--pgn_dir', type=str, default='pgn_games', help='Directory to save PGN files')
    parser.add_argument('--visualize', action='store_true', help='Visualize the board after each move')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between moves when visualizing (seconds)')
    parser.add_argument('--use_opening_book', action='store_true', help='Use opening book knowledge for testing')
    parser.add_argument('--debug', action='store_true', help='Print debug information about actions')
    
    return parser.parse_args()

def save_game_to_pgn(board, filename):
    """Save game to PGN format"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Create PGN exporter
    game = chess.pgn.Game()
    game.headers["Event"] = "Chess RL Agent Test"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["White"] = "RL Agent"
    game.headers["Black"] = "RL Agent"
    
    # Set the result based on game state
    if board.is_checkmate():
        result = "1-0" if board.turn == chess.BLACK else "0-1"
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition():
        result = "1/2-1/2"
    else:
        # Incomplete game
        result = "*"
    
    game.headers["Result"] = result
    
    # Add moves
    move_count = 0
    node = game
    for move in board.move_stack:
        node = node.add_variation(move)
        move_count += 1
    
    # Diagnostic information
    if move_count == 0:
        print(f"WARNING: No moves were recorded for this game! Board FEN: {board.fen()}")
    
    # Write to file
    with open(filename, 'w') as f:
        f.write(str(game))
    
    print(f"Game saved to {filename} with {move_count} moves")

def map_to_legal_action(action, env):
    """Map the model's action to a legal action index"""
    # If action is already a valid index, return it
    if 0 <= action < len(env.legal_moves_list):
        return action
    
    # Otherwise, map it to a valid index using modulo
    if len(env.legal_moves_list) > 0:
        return action % len(env.legal_moves_list)
    
    # Fallback (should never happen as there are always legal moves until game over)
    return 0

def test_model(model_path='models_with_opening_book/chess_with_opening_book_final', 
              num_games=5, max_moves=50, save_pgn=False, pgn_dir='pgn_games',
              visualize=False, delay=0.5, use_opening_book=True, debug=False):
    """
    Test a trained chess RL agent with the given parameters.
    
    Args:
        model_path: Path to the trained model
        num_games: Number of games to play
        max_moves: Maximum moves per game
        save_pgn: Whether to save games in PGN format
        pgn_dir: Directory to save PGN files
        visualize: Whether to visualize the board after each move
        delay: Delay between moves when visualizing (seconds)
        use_opening_book: Whether to use opening book knowledge
        debug: Whether to print debug information about actions
        
    Returns:
        A dictionary with game statistics
    """
    # Create PGN directory if saving games
    if save_pgn:
        os.makedirs(pgn_dir, exist_ok=True)
    
    # Load the model
    model = DQN.load(model_path)
    print(f"Loaded model from {model_path}")
    
    # Statistics
    total_moves = 0
    checkmates = 0
    stalemates = 0
    draws = 0
    incomplete = 0
    opening_book_moves = 0
    total_rewards = 0
    
    # Play games
    for game_num in range(1, num_games + 1):
        print(f"\nGame {game_num}/{num_games}")
        print("=" * 40)
        
        # Create environment
        env = ChessEnvWithOpeningBook(use_opening_book=use_opening_book)
        obs = env.reset()
        
        # Game variables
        done = False
        moves = 0
        game_reward = 0
        game_opening_book_moves = 0
        
        # Play until done or max moves reached
        while not done and moves < max_moves:
            # Get action from model
            raw_action, _ = model.predict(obs, deterministic=True)
            
            # Map to legal action
            action = map_to_legal_action(raw_action, env)
            
            # Debug information
            if debug:
                print(f"Original action: {raw_action}, Mapped to: {action}")
                print(f"Legal moves count: {len(env.legal_moves_list)}")
                if 0 <= action < len(env.legal_moves_list):
                    print(f"Corresponding move: {env.legal_moves_list[action]}")
                    
                # Print first few legal moves
                if len(env.legal_moves_list) > 0:
                    print("Sample legal moves:")
                    for i, move in enumerate(env.legal_moves_list[:5]):
                        print(f"  {i}: {move.uci()}")
            
            # Apply the action
            obs, reward, done, info = env.step(action)
            
            # Check if this move followed the opening book
            if info.get('followed_opening_book', False):
                game_opening_book_moves += 1
            
            moves += 1
            game_reward += reward
            
            # Visualize if requested
            if visualize:
                print(f"Move {moves}:")
                print(env.board)
                print(f"Reward: {reward:.2f}")
                if info.get('followed_opening_book', False):
                    print("Following opening book!")
                print()
                time.sleep(delay)
        
        # Game finished
        total_moves += moves
        total_rewards += game_reward
        opening_book_moves += game_opening_book_moves
        
        # Record outcome
        if env.board.is_checkmate():
            checkmates += 1
            outcome = "Checkmate"
        elif env.board.is_stalemate():
            stalemates += 1
            outcome = "Stalemate"
        elif env.board.is_insufficient_material() or env.board.is_fifty_moves() or env.board.is_repetition():
            draws += 1
            outcome = "Draw"
        else:
            incomplete += 1
            outcome = "Incomplete (max moves reached)"
        
        # Print game summary
        print(f"Game {game_num} finished after {moves} moves")
        print(f"Final position: {env.board.fen()}")
        print(f"Outcome: {outcome}")
        print(f"Opening book moves: {game_opening_book_moves}")
        print(f"Total reward: {game_reward:.2f}")
        print(f"Moves in board stack: {len(env.board.move_stack)}")
        
        # Save to PGN if requested
        if save_pgn:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{pgn_dir}/game_with_opening_book_{game_num}_{timestamp}.pgn"
            save_game_to_pgn(env.board, filename)
    
    # Print overall statistics
    print("\nOverall Statistics:")
    print("=" * 40)
    print(f"Total games: {num_games}")
    print(f"Average moves per game: {total_moves / num_games:.2f}")
    print(f"Average reward per game: {total_rewards / num_games:.2f}")
    print(f"Average opening book moves per game: {opening_book_moves / num_games:.2f}")
    print(f"Checkmates: {checkmates} ({checkmates / num_games * 100:.1f}%)")
    print(f"Stalemates: {stalemates} ({stalemates / num_games * 100:.1f}%)")
    print(f"Draws: {draws} ({draws / num_games * 100:.1f}%)")
    print(f"Incomplete games: {incomplete} ({incomplete / num_games * 100:.1f}%)")
    
    # Return statistics as a dictionary
    return {
        "total_games": num_games,
        "total_moves": total_moves,
        "average_moves": total_moves / num_games,
        "average_reward": total_rewards / num_games,
        "opening_book_moves": opening_book_moves,
        "average_opening_book_moves": opening_book_moves / num_games,
        "checkmates": checkmates,
        "stalemates": stalemates,
        "draws": draws,
        "incomplete": incomplete
    }

def main():
    args = parse_args()
    
    test_model(
        model_path=args.model,
        num_games=args.games,
        max_moves=args.max_moves,
        save_pgn=args.save_pgn,
        pgn_dir=args.pgn_dir,
        visualize=args.visualize,
        delay=args.delay,
        use_opening_book=args.use_opening_book,
        debug=args.debug
    )

if __name__ == "__main__":
    main() 