import gym
import torch
import chess
import os
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from chess_gym_env_improved import ImprovedChessEnv  # Import the improved environment
from custom_callbacks import MetricsLogger
from stable_baselines3.common.callbacks import CallbackList

# Create logs directory
logs_dir = "logs_improved"
models_dir = "models_improved"
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Initialize the improved chess environment
env = ImprovedChessEnv()
print("Environment initialized with improved reward function.")

# Define callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=10000,  # Save model every 10k steps
    save_path=models_dir,
    name_prefix="chess_model_improved"
)

# Add custom metrics logger callback
metrics_logger = MetricsLogger(log_dir=logs_dir)

# Combine callbacks
callbacks = CallbackList([checkpoint_callback, metrics_logger])

# Define RL model (Deep Q-Network) with optimized parameters
model = DQN(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=0.0005,           # Higher learning rate for faster adaptation
    batch_size=128,                 # Larger batch size for more stable learning
    buffer_size=100000,             # Larger buffer for better memory
    exploration_final_eps=0.1,      # Final exploration rate
    exploration_fraction=0.4,       # Explore for the first 40% of training
    learning_starts=5000,           # Start learning after accumulating experiences
    target_update_interval=1000,    # Update target network more frequently
    gamma=0.99,                     # Discount factor
    train_freq=4,                   # Update model every 4 steps
    gradient_steps=1,               # Number of gradient steps per update
    tensorboard_log=logs_dir,
    device="auto"
)

print("Starting training with improved parameters and reward function...")
print("This should result in better learning and more stable progress.")

# Train the model
try:
    model.learn(
        total_timesteps=1000000,  # 1M steps for training
        callback=callbacks,
        log_interval=100          # Print stats every 100 updates
    )
    print("Training completed successfully!")
except Exception as e:
    print(f"Training was interrupted: {e}")

# Save the final model
model.save(f"{models_dir}/chess_rl_improved_final")
print(f"Model saved to {models_dir}/chess_rl_improved_final")

# Print location of training metrics
print(f"Training metrics saved to {logs_dir}/plots/ and {logs_dir}/csv/")
print(f"To view TensorBoard logs, run: tensorboard --logdir={logs_dir}")

print("\nKey improvements made:")
print("1. Balanced reward function with better scaling")
print("2. Added small positive rewards for valid moves")
print("3. Enhanced strategic rewards (development, mobility, pawn structure)")
print("4. Optimized learning parameters (learning rate, batch size, exploration)")
print("5. Added detailed metrics tracking for better analysis") 