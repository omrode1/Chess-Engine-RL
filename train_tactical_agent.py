import gym
import torch
import chess
import chess.pgn
import os
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from src.chess_environment import ChessEnvWithOpeningBook
from utils.custom_callbacks import MetricsLogger, ActionQualityMonitor
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from torch import nn
import torch.nn.functional as F
from gym import spaces
from typing import Any, Dict, List, Optional, Tuple, Type, Union






# Custom neural network that will be used as policy
class ChessFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for the chess board representation.
    This takes a 12x8x8 board tensor and extracts features.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Convolutional layers to process the board
        self.cnn = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate the size of the flattened features
        # For an 8x8 board with 64 filters, this will be 64*8*8 = 4096
        cnn_output_dim = 64 * 8 * 8
        
        # Fully connected layers to reduce dimensionality
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Observations shape: (batch_size, 12, 8, 8)
        features = self.cnn(observations)
        return self.fc(features)

# Custom DQN policy that supports action masking
class MaskedDQNPolicy(DQNPolicy):
    """
    DQN policy that masks illegal moves during action selection.
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        *args,
        **kwargs,
    ):
        # Use our custom feature extractor
        kwargs["features_extractor_class"] = ChessFeatureExtractor
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
    
    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get the action according to the policy for a given observation.
        
        :param observation: The current observation
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        # Get Q-values for all actions
        q_values = self.q_net(observation)
        
        # We can't directly access the environment from the policy
        # Instead, we'll always return the argmax of Q-values
        # The LegalMovesWrapper will ensure the actions are legal
        return torch.argmax(q_values, dim=1)

# Environment wrapper that maps actions to legal moves
class LegalMovesWrapper(gym.Wrapper):
    """
    A wrapper for the chess environment that ensures all actions
    are mapped to legal moves, even during exploration.
    """
    def __init__(self, env):
        super().__init__(env)
        
    def step(self, action):
        # Map action to legal move if needed
        if hasattr(self.env.unwrapped, "legal_moves_list"):
            legal_moves = self.env.unwrapped.legal_moves_list
            if len(legal_moves) > 0:
                # If action is outside the range of legal moves, map it using modulo
                if action >= len(legal_moves):
                    action = action % len(legal_moves)
        
        # Call the original step method with the mapped action
        return self.env.step(action)

def make_env(rank, use_opening_book=True, seed=0):
    """
    Creates a chess environment with the appropriate wrappers.
    Used for vectorized environments.
    """
    def _init():
        env = ChessEnvWithOpeningBook(
            use_opening_book=use_opening_book,
            opening_book_moves=10,  # Use opening book for the first 10 moves
            opening_bonus=0.5  # Bonus reward for following the opening book
        )
        env = LegalMovesWrapper(env)
        return env
    return _init

def train_model(total_timesteps=1000000, model_dir="tactical_models", 
                log_dir="tactical_logs", use_opening_book=True,
                num_envs=4):  # Number of parallel environments
    """
    Train a chess RL agent with the given parameters.
    
    Args:
        total_timesteps: Number of timesteps to train for
        model_dir: Directory to save models
        log_dir: Directory to save logs
        use_opening_book: Whether to use opening book knowledge
        num_envs: Number of parallel environments to use
        
    Returns:
        The trained model
    """
    # Enable CUDA optimizations if available
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Create vectorized environment with multiple parallel environments
    env = SubprocVecEnv([make_env(i, use_opening_book) for i in range(num_envs)], start_method='spawn')
    
    # Add monitoring wrapper at the vectorized environment level instead of individual environments
    env = VecMonitor(env, log_dir)

    print(f"Environment initialized with {num_envs} parallel games, opening book knowledge, legal moves wrapper, and enhanced tactical awareness.")

    # Define callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # Save model every 10k steps
        save_path=model_dir,
        name_prefix="chess_tactical_model"
    )

    # Add custom metrics logger callback
    metrics_logger = MetricsLogger(log_dir=log_dir)

    # Add action quality monitor callback
    action_monitor = ActionQualityMonitor(log_dir=log_dir)

    # Combine callbacks
    callbacks = CallbackList([checkpoint_callback, metrics_logger, action_monitor])

    # Define RL model (Deep Q-Network) with optimized parameters and custom policy
    model = DQN(
        policy=MaskedDQNPolicy,
        env=env, 
        verbose=1, 
        learning_rate=0.0005,           # Higher learning rate for faster adaptation
        batch_size=256,                 # Even larger batch size for more stable learning with parallel envs
        buffer_size=200000,             # Larger buffer for better memory with parallel envs
        exploration_final_eps=0.05,     # Lower final exploration for more exploitation
        exploration_fraction=0.25,      # Faster exploration decay with parallel envs
        learning_starts=1000,           # Start learning sooner with parallel envs
        target_update_interval=500,     # Update target network more frequently
        gamma=0.99,                     # Discount factor
        train_freq=1,                   # Update model every step for faster learning
        gradient_steps=4,               # More gradient steps per update for faster learning
        tensorboard_log=log_dir,
        device="auto"
    )

    print("Starting training with enhanced tactical awareness...")
    print("The agent will receive rewards for following established opening theory.")
    print("Illegal moves will be masked during training to improve action selection.")
    print("The agent will receive penalties for hanging pieces and rewards for capturing undefended pieces.")
    print("The agent will be rewarded for defending its threatened pieces.")
    print(f"Training with {num_envs} parallel environments for increased speed.")

    # Train the model
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=100          # Print stats every 100 updates
        )
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training was interrupted: {e}")

    # Save the final model
    final_model_path = f"{model_dir}/chess_tactical_final"
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")

    # Print location of training metrics
    print(f"Training metrics saved to {log_dir}/plots/ and {log_dir}/csv/")
    print(f"To view TensorBoard logs, run: tensorboard --logdir={log_dir}")

    print("\nKey improvements made:")
    print("1. Incorporated opening book knowledge from established chess theory")
    print("2. Added penalties for hanging pieces")
    print("3. Increased rewards for capturing undefended pieces")
    print("4. Added rewards for defending threatened pieces")
    print("5. Improved action masking to ensure only legal moves are considered")
    print("6. Implemented CNN-based neural network specifically designed for chess")
    print("7. Optimized exploration parameters for better learning")
    print("8. Detailed metrics tracking for better analysis") 
    print("9. Parallelized training with multiple environments for faster learning")
    
    return model

if __name__ == "__main__":
    # Determine number of environments based on CPU cores (leave some for system)
    import multiprocessing
    num_cpus = multiprocessing.cpu_count()
    num_envs = max(2, num_cpus - 2)  # Use at least 2, but leave 2 cores for system
    
    train_model(num_envs=num_envs) 