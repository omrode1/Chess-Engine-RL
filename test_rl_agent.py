import chess
import chess.pgn
import os
import datetime
from stable_baselines3 import DQN
from chess_gym_env import ChessEnv

# Set models directory
models_dir = "models"
# Create a directory for PGN files
pgn_dir = "pgn_games"
os.makedirs(pgn_dir, exist_ok=True)

# Find the latest model
model_path = f"{models_dir}/chess_rl_final.zip"
if not os.path.exists(model_path):
    # Try to find the latest checkpoint if final model doesn't exist
    checkpoints = [f for f in os.listdir(models_dir) if f.startswith("chess_model_") and f.endswith(".zip")]
    if checkpoints:
        latest_checkpoint = sorted(checkpoints)[-1]
        model_path = os.path.join(models_dir, latest_checkpoint)
    else:
        raise FileNotFoundError("No trained model found. Please train a model first.")

print(f"Loading model from: {model_path}")
model = DQN.load(model_path)

# Initialize chess environment
env = ChessEnv()
obs = env.reset()

print("Initial board position:")
print(env.board)
print("\nAction space size:", env.action_space.n)

# Let the RL agent play a complete game or maximum 50 moves
for i in range(50):
    # RL model chooses best move
    action, _ = model.predict(obs, deterministic=True)
    
    # Apply move and get feedback
    obs, reward, done, _ = env.step(action)
    
    # Print move details
    if action < len(env.legal_moves_list):
        move = env.legal_moves_list[action]
        print(f"Move {i+1}: {move.uci()} (Action: {action}), Reward: {reward:.2f}")
    else:
        print(f"Move {i+1}: Invalid action ({action}), Reward: {reward:.2f}")

    print(env.board)
    print("--------------------")
    
    if done:
        print("Game Over!")
        if env.board.is_checkmate():
            print("Checkmate!")
        elif env.board.is_stalemate():
            print("Stalemate!")
        elif env.board.is_insufficient_material():
            print("Insufficient material!")
        elif env.board.is_fifty_moves():
            print("Fifty-move rule!")
        elif env.board.is_repetition():
            print("Threefold repetition!")
        break

print("Final board position:")
print(env.board)

# Save the game as a PGN file
def save_game_as_pgn(board, model_name):
    # Create a game from the board's move stack
    game = chess.pgn.Game()
    
    # Set headers
    game.headers["Event"] = "AI Chess Game"
    game.headers["Site"] = "RL Training"
    game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
    game.headers["Round"] = "1"
    game.headers["White"] = "ChessRL Agent"
    game.headers["Black"] = "ChessRL Agent"
    game.headers["Result"] = board.result()
    
    # Add moves
    node = game
    for move in board.move_stack:
        node = node.add_variation(move)
    
    # Generate a unique filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(model_path).replace(".zip", "")
    filename = f"{pgn_dir}/game_{model_name}_{timestamp}.pgn"
    
    # Write PGN to file
    with open(filename, "w") as pgn_file:
        print(game, file=pgn_file, end="\n\n")
    
    print(f"Game saved as PGN to: {filename}")

# Save the game
save_game_as_pgn(env.board, model_path)
