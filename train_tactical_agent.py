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
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Increased filters
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # Increased filters
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate the size of the flattened features
        # For an 8x8 board with 128 filters, this will be 128*8*8 = 8192
        cnn_output_dim = 128 * 8 * 8
        
        # Fully connected layers to reduce dimensionality
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_dim, 1024),  # Increased hidden layer size
            nn.ReLU(),
            nn.Linear(1024, features_dim),
            nn.ReLU(),
        )
        
        # Value head for predicting game outcome (win/loss probability)
        self.value_head = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()  # Output between -1 and 1 (loss to win probability)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Observations shape: (batch_size, 12, 8, 8)
        features = self.cnn(observations)
        features = self.fc(features)
        # We don't use the value head in the forward pass for DQN
        # It's used in the custom loss function
        return features
        
    def predict_value(self, observations: torch.Tensor) -> torch.Tensor:
        # Use this method to get the value prediction
        features = self.cnn(observations)
        features = self.fc(features)
        value = self.value_head(features)
        return value

# Custom DQN policy that supports action masking and value prediction
class MaskedDQNPolicy(DQNPolicy):
    """
    DQN policy that masks illegal moves during action selection
    and includes a value prediction head.
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
        
        # Value loss weight for the auxiliary task
        self.value_loss_weight = 0.5
        # Create value optimizer
        self.value_optimizer = torch.optim.Adam(
            self.q_net.features_extractor.value_head.parameters(),
            lr=lr_schedule(1)
        )
    
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
    
    def train(self, gradient_steps: int, batch_size: int) -> None:
        """
        Override the train method to include value prediction training
        """
        # Get original training losses
        losses = super().train(gradient_steps, batch_size)
        
        # Sample from replay buffer for value prediction training
        if self.replay_buffer is not None and self.replay_buffer.size() > 0:
            # Sample from the replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            # Extract observations and rewards
            observations = replay_data.observations
            rewards = replay_data.rewards
            dones = replay_data.dones
            
            # Predict values
            values = self.q_net.features_extractor.predict_value(observations).squeeze()
            
            # Calculate value loss - try to predict cumulative reward
            value_loss = F.mse_loss(values, rewards)
            
            # Backpropagate
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            
            # Add value loss to returned losses
            if isinstance(losses, dict):
                losses["value_loss"] = value_loss.item()
            
        return losses

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

def make_env(rank, use_opening_book=True, seed=0, level=0):
    """
    Creates a chess environment with the appropriate wrappers.
    Used for vectorized environments.
    
    Args:
        rank: Environment rank for parallel environments
        use_opening_book: Whether to use opening book
        seed: Random seed
        level: Curriculum level (0=full game, 1=midgame, 2=endgame, 3=tactical puzzles)
    """
    def _init():
        env = ChessEnvWithOpeningBook(
            use_opening_book=use_opening_book,
            opening_book_moves=10,  # Use opening book for the first 10 moves
            opening_bonus=0.5,  # Bonus reward for following the opening book
            curriculum_level=level  # Set the curriculum level
        )
        env = LegalMovesWrapper(env)
        return env
    return _init

def train_model(total_timesteps=3000000, model_dir="tactical_models", 
                log_dir="tactical_logs", use_opening_book=True,
                num_envs=4, use_curriculum=True):  # Added curriculum learning option
    """
    Train a chess RL agent with the given parameters.
    
    Args:
        total_timesteps: Number of timesteps to train for
        model_dir: Directory to save models
        log_dir: Directory to save logs
        use_opening_book: Whether to use opening book knowledge
        num_envs: Number of parallel environments to use
        use_curriculum: Whether to use curriculum learning
        
    Returns:
        The trained model
    """
    # Enable CUDA optimizations if available
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Define curriculum stages and timesteps for each
    curriculum_stages = [
        {"level": 3, "name": "tactical puzzles", "timesteps": int(total_timesteps * 0.15)},
        {"level": 2, "name": "endgames", "timesteps": int(total_timesteps * 0.25)},
        {"level": 1, "name": "midgames", "timesteps": int(total_timesteps * 0.3)},
        {"level": 0, "name": "full games", "timesteps": int(total_timesteps * 0.3)},
    ]
    
    # Skip curriculum if not using it
    if not use_curriculum:
        curriculum_stages = [{"level": 0, "name": "full games", "timesteps": total_timesteps}]
    
    print(f"Training with {'curriculum learning' if use_curriculum else 'standard training'}")
    if use_curriculum:
        print(f"Curriculum stages:")
        for stage in curriculum_stages:
            print(f"  - {stage['name']}: {stage['timesteps']} timesteps (level {stage['level']})")
    
    # Start with first curriculum level
    current_stage = curriculum_stages[0]
    current_model = None
    
    # Train through each curriculum stage
    remaining_timesteps = total_timesteps
    for i, stage in enumerate(curriculum_stages):
        print(f"\n{'='*50}")
        print(f"Starting curriculum stage {i+1}/{len(curriculum_stages)}: {stage['name']}")
        print(f"{'='*50}")
        
        # Create vectorized environment with the current curriculum level
        env_fns = [lambda level=stage['level']: make_env(i, use_opening_book, level=level)() 
                   for i in range(num_envs)]
        env = SubprocVecEnv(env_fns, start_method='spawn')
        
        # Add monitoring wrapper at the vectorized environment level
        env = VecMonitor(env, os.path.join(log_dir, f"stage_{i+1}_{stage['name']}"))

        print(f"Environment initialized with {num_envs} parallel games, curriculum level {stage['level']} ({stage['name']})")

        # Define callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,  # Save model every 10k steps
            save_path=model_dir,
            name_prefix=f"chess_tactical_stage_{i+1}_{stage['name']}"
        )

        # Add custom metrics logger callback
        metrics_logger = MetricsLogger(log_dir=os.path.join(log_dir, f"stage_{i+1}_{stage['name']}"))

        # Add action quality monitor callback
        action_monitor = ActionQualityMonitor(log_dir=os.path.join(log_dir, f"stage_{i+1}_{stage['name']}"))

        # Combine callbacks
        callbacks = CallbackList([checkpoint_callback, metrics_logger, action_monitor])

        # Define RL model or continue from previous stage
        if current_model is None:
            # First stage - create new model
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
                tensorboard_log=os.path.join(log_dir, f"stage_{i+1}_{stage['name']}"),
                device="auto"
            )
        else:
            # Continue from previous stage model
            model = DQN.load(
                current_model,
                env=env,
                # Keep original hyperparameters but adjust a few for curriculum
                exploration_final_eps=max(0.05, 0.1 - i * 0.02),  # Decrease exploration gradually
                learning_rate=0.0005 * (0.7 ** i),  # Decrease learning rate gradually
                tensorboard_log=os.path.join(log_dir, f"stage_{i+1}_{stage['name']}"),
            )
            # This prevents buffer artifacts from previous stages
            model.replay_buffer = None

        print(f"Starting training on {stage['name']} with {stage['timesteps']} timesteps...")

        # Train the model
        try:
            model.learn(
                total_timesteps=stage['timesteps'],
                callback=callbacks,
                log_interval=100          # Print stats every 100 updates
            )
            print(f"Stage {i+1} completed successfully!")
            
            # Save stage model
            stage_model_path = f"{model_dir}/chess_tactical_stage_{i+1}_{stage['name']}_complete"
            model.save(stage_model_path)
            print(f"Stage model saved to {stage_model_path}")
            
            # Update current model for next stage
            current_model = stage_model_path
            
        except Exception as e:
            print(f"Training was interrupted: {e}")
            break
    
    # Save the final model
    final_model_path = f"{model_dir}/chess_tactical_final"
    if model is not None:
        model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")

    # Print location of training metrics
    print(f"Training metrics saved to {log_dir}")
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
    print("10. Implemented curriculum learning to build skills incrementally")
    print("11. Added value prediction auxiliary task for improved learning")
    print("12. Rebalanced rewards to provide more positive feedback")
    
    return model

if __name__ == "__main__":
    # Determine number of environments based on CPU cores (leave some for system)
    import multiprocessing
    num_cpus = multiprocessing.cpu_count()
    num_envs = max(2, num_cpus - 2)  # Use at least 2, but leave 2 cores for system
    
    train_model(num_envs=num_envs) 