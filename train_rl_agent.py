import gym
import torch
import chess
import os
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
# from stable_baselines3.common.monitor import PPO
from chess_gym_env import ChessEnv  # Import your custom environment
# from stable_baselines3.common.monitor import Monitor  # Removing this import
from custom_callbacks import MetricsLogger
from stable_baselines3.common.callbacks import CallbackList

# Create logs directory
logs_dir = "logs"
models_dir = "models"
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Initialize the chess environment
env = ChessEnv()
# env = Monitor(env, logs_dir)  # Removing this line as it's causing compatibility issues

# Define callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=5000,  # Save model every 5k steps
    save_path=models_dir,
    name_prefix="chess_model"
)

# Add custom metrics logger callback
metrics_logger = MetricsLogger(log_dir=logs_dir)

# Combine callbacks
callbacks = CallbackList([checkpoint_callback, metrics_logger])

# Define RL model (Deep Q-Network)
model = DQN(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=0.0003,       # Lower learning rate for stability
    batch_size=64,              # Smaller batch size
    buffer_size=50000,          # Smaller buffer size
    exploration_final_eps=0.1,  # Final exploration rate
    exploration_fraction=0.2,   # Explore for the first 20% of training
    learning_starts=1000,       # Start learning after 1000 steps
    target_update_interval=500, # Update target network more frequently
    tensorboard_log=logs_dir,
    device="auto"
)

# Train the model
print("Training Started...")
model.learn(
    total_timesteps=1000000,  # Fewer timesteps for faster training
    callback=callbacks
)
print("Training Completed!")

# Save the final model
model.save(f"{models_dir}/chess_rl_final")
print("Model Saved!")

# Print location of training metrics
print(f"Training metrics saved to {logs_dir}/plots/ and {logs_dir}/csv/")
print("To view TensorBoard logs, run: tensorboard --logdir={logs_dir}")
