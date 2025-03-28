import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
from collections import Counter, deque

class MetricsLogger(BaseCallback):
    """
    Custom callback for logging chess-specific metrics:
    1. Reward Trends (Episode Rewards)
    2. Move Diversity (Exploration vs. Exploitation)
    3. Move Repetitions (Oscillation Detection)
    """
    def __init__(self, log_dir="logs", verbose=0):
        super(MetricsLogger, self).__init__(verbose)
        self.log_dir = log_dir
        
        # Create directories for different visualizations
        self.plots_dir = os.path.join(log_dir, "plots")
        self.csv_dir = os.path.join(log_dir, "csv")
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)
        
        # Initialize metrics trackers
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_moves = []
        
        # For move diversity tracking
        self.move_counter = Counter()
        self.total_moves = 0
        
        # For oscillation detection (using last few moves)
        self.recent_positions = {}  # Store position -> count
        self.oscillation_counts = []
        
        # Store data for CSV export
        self.metrics_data = {
            'timesteps': [],
            'episode': [],
            'reward': [],
            'length': [],
            'move_diversity': [],
            'oscillation_count': []
        }
        self.episode_count = 0
        
        # For tracking positions within episodes
        self.current_positions = []
        
    def _on_step(self) -> bool:
        # Get info from the most recent step - handle both vectorized and non-vectorized envs
        if isinstance(self.locals.get('infos'), list) and len(self.locals.get('infos', [])) > 0:
            # Vectorized environment
            info = self.locals.get('infos')[0]
            reward = self.locals.get('rewards')[0]
            done = self.locals.get('dones')[0]
            env = self.training_env.envs[0] if hasattr(self.training_env, 'envs') else self.training_env
        else:
            # Non-vectorized environment
            info = self.locals.get('info', {})
            reward = self.locals.get('reward', 0)
            done = self.locals.get('done', False)
            env = self.training_env
        
        # Extract board from environment
        # This will work with the ChessEnv regardless of gym vs gymnasium
        board = env.board if hasattr(env, 'board') else None
        
        # Accumulate reward for the current episode
        self.current_episode_reward += reward
        
        # Track positions for oscillation detection
        if board:
            fen = board.fen().split(' ')[0]  # Just the piece positions
            self.current_positions.append(fen)
            
            # Track moves for diversity analysis
            if len(board.move_stack) > 0:
                last_move = board.move_stack[-1].uci()
                self.current_episode_moves.append(last_move)
                self.move_counter[last_move] += 1
                self.total_moves += 1
        
        # Check if episode is done
        if done:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(len(self.current_episode_moves))
            
            # Calculate move diversity using entropy
            move_diversity = self._calculate_move_diversity()
            
            # Calculate oscillation score
            oscillation_count = self._calculate_oscillations(self.current_positions)
            self.oscillation_counts.append(oscillation_count)
            
            # Record metrics
            self.metrics_data['timesteps'].append(self.num_timesteps)
            self.metrics_data['episode'].append(self.episode_count)
            self.metrics_data['reward'].append(self.current_episode_reward)
            self.metrics_data['length'].append(len(self.current_episode_moves))
            self.metrics_data['move_diversity'].append(move_diversity)
            self.metrics_data['oscillation_count'].append(oscillation_count)
            
            # Reset episode-specific trackers
            self.current_episode_reward = 0
            self.current_episode_moves = []
            self.current_positions = []
            
            # Generate plots every 10 episodes
            if self.episode_count % 10 == 0:
                self._save_metrics_csv()
                self._plot_metrics()
                
        return True
    
    def _calculate_move_diversity(self):
        """Calculate move diversity using entropy"""
        if not self.current_episode_moves:
            return 0
            
        # Count occurrences of each move in the episode
        move_counts = Counter(self.current_episode_moves)
        
        # Calculate probabilities
        probs = [count / len(self.current_episode_moves) for count in move_counts.values()]
        
        # Calculate entropy
        entropy = -sum(p * np.log(p) for p in probs)
        
        # Normalize to [0, 1] scale (0 = all same moves, 1 = all different moves)
        max_entropy = np.log(len(self.current_episode_moves)) if len(self.current_episode_moves) > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return normalized_entropy
    
    def _calculate_oscillations(self, positions):
        """Calculate the number of position repetitions"""
        if not positions:
            return 0
            
        # Count repeated positions
        position_counter = Counter(positions)
        
        # Sum up all repetitions (positions that appear more than once)
        oscillation_count = sum(count - 1 for count in position_counter.values() if count > 1)
        
        return oscillation_count
        
    def _save_metrics_csv(self):
        """Save metrics to CSV file"""
        df = pd.DataFrame(self.metrics_data)
        csv_path = os.path.join(self.csv_dir, f"metrics_{self.num_timesteps}.csv")
        df.to_csv(csv_path, index=False)
        
    def _plot_metrics(self):
        """Generate and save plots"""
        timestep = self.num_timesteps
        
        # Create figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        
        # Plot episode rewards
        axs[0].plot(self.episode_rewards)
        axs[0].set_title('Episode Rewards')
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel('Reward')
        
        # Plot move diversity
        diversity_values = self.metrics_data['move_diversity']
        axs[1].plot(diversity_values)
        axs[1].set_title('Move Diversity (Exploration vs Exploitation)')
        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel('Diversity Score (0-1)')
        
        # Plot oscillation counts
        axs[2].plot(self.oscillation_counts)
        axs[2].set_title('Move Repetitions (Oscillation Detection)')
        axs[2].set_xlabel('Episode')
        axs[2].set_ylabel('Number of Repeated Positions')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f"metrics_plot_{timestep}.png"))
        plt.close() 