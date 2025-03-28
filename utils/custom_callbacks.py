import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
from collections import Counter, deque
import csv
from stable_baselines3.common.results_plotter import ts2xy, load_results
import torch

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
        
        # Set up CSV logger
        self.csv_file = open(os.path.join(self.csv_dir, "training_metrics.csv"), 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Timestep', 'Reward', 'Episode Length', 'Material Balance'])
        
        # Metrics tracking
        self.material_balances = []
        self.timesteps = []
        
        # For averaging metrics
        self.current_episode_length = 0
        self.current_episode_material = []
        
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
        
        # Track material balance if available
        if isinstance(info, dict) and 'material_balance' in info:
            self.current_episode_material.append(info['material_balance'])
        
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
            
            # Calculate average material balance for the episode
            if self.current_episode_material:
                avg_material = np.mean(self.current_episode_material)
                self.material_balances.append(avg_material)
            else:
                self.material_balances.append(0)
            
            # Record metrics
            self.metrics_data['timesteps'].append(self.num_timesteps)
            self.metrics_data['episode'].append(self.episode_count)
            self.metrics_data['reward'].append(self.current_episode_reward)
            self.metrics_data['length'].append(len(self.current_episode_moves))
            self.metrics_data['move_diversity'].append(move_diversity)
            self.metrics_data['oscillation_count'].append(oscillation_count)
            
            # Log to CSV
            self.csv_writer.writerow([
                self.num_timesteps, 
                self.current_episode_reward, 
                len(self.current_episode_moves),
                self.material_balances[-1]
            ])
            self.csv_file.flush()  # Ensure data is written immediately
            
            # Reset episode-specific trackers
            self.current_episode_reward = 0
            self.current_episode_moves = []
            self.current_positions = []
            self.current_episode_length = 0
            self.current_episode_material = []
            
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
        
    def on_training_end(self):
        """Clean up and create final plots on training end"""
        self.csv_file.close()
        
        # Create final plots
        self._plot_metrics()
        
        # Create a DataFrame with all metrics for easier analysis
        df = pd.DataFrame({
            'Timestep': self.timesteps,
            'Reward': self.episode_rewards,
            'Episode_Length': self.episode_lengths,
            'Material_Balance': self.material_balances
        })
        
        # Save DataFrame to CSV
        df.to_csv(os.path.join(self.csv_dir, "all_metrics.csv"), index=False)
        
        # Create a smoothed version of the reward plot
        if len(self.episode_rewards) > 0:
            window_size = min(50, len(self.episode_rewards))
            smoothed_rewards = df['Reward'].rolling(window=window_size).mean()
            
            plt.figure(figsize=(10, 5))
            plt.plot(self.timesteps, self.episode_rewards, alpha=0.3, label='Raw')
            plt.plot(self.timesteps, smoothed_rewards, label=f'Smoothed (window={window_size})')
            plt.xlabel('Timesteps')
            plt.ylabel('Episode Reward')
            plt.title('Training Rewards (Smoothed)')
            plt.legend()
            plt.savefig(os.path.join(self.plots_dir, "rewards_smoothed.png"))
            plt.close()

class ActionQualityMonitor(BaseCallback):
    """
    Monitors the quality of action selection during training.
    Tracks how often the model selects legal actions vs illegal actions.
    """
    def __init__(self, log_dir, verbose=0):
        super(ActionQualityMonitor, self).__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(f"{log_dir}/plots", exist_ok=True)
        os.makedirs(f"{log_dir}/csv", exist_ok=True)
        
        # Set up CSV logger for action quality
        self.csv_file = open(f"{log_dir}/csv/action_quality.csv", 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Timestep', 'Legal Actions', 'Illegal Actions', 'Legal Action Rate'])
        
        # Metrics tracking
        self.legal_actions = 0
        self.illegal_actions = 0
        self.legal_action_rates = []
        self.timesteps_log = []
        self.log_frequency = 1000  # Log every 1000 steps
        
    def _on_step(self) -> bool:
        # Get the environment
        env = self.training_env.envs[0].unwrapped if hasattr(self.training_env, 'envs') else self.training_env.unwrapped
        
        # Check the reward value to determine if the move was valid
        # In our chess environment, invalid moves get -0.5 reward
        reward = self.locals.get('rewards', [0])[0] if isinstance(self.locals.get('rewards'), list) else self.locals.get('reward', 0)
        
        # Check for invalid move based on reward value or info dict
        is_invalid = False
        
        # In our chess env, reward of -0.5 indicates an invalid move
        if reward == -0.5:
            is_invalid = True
        
        # Some envs might also have an 'invalid_move' key in info
        info = self.locals.get('infos', [{}])[0] if isinstance(self.locals.get('infos'), list) else self.locals.get('info', {})
        if isinstance(info, dict) and info.get('invalid_move', False):
            is_invalid = True
        
        # Update counters
        if is_invalid:
            self.illegal_actions += 1
        else:
            self.legal_actions += 1
        
        # Log metrics periodically
        if self.num_timesteps % self.log_frequency == 0 and (self.legal_actions + self.illegal_actions) > 0:
            total_actions = self.legal_actions + self.illegal_actions
            legal_action_rate = self.legal_actions / total_actions if total_actions > 0 else 0
            
            self.legal_action_rates.append(legal_action_rate)
            self.timesteps_log.append(self.num_timesteps)
            
            # Log to CSV
            self.csv_writer.writerow([
                self.num_timesteps,
                self.legal_actions,
                self.illegal_actions,
                legal_action_rate
            ])
            self.csv_file.flush()
            
            # Plot the metrics
            if len(self.timesteps_log) > 1:  # Only plot if we have multiple data points
                self._plot_action_quality()
        
        return True
    
    def _plot_action_quality(self):
        """Generate plot for action quality metrics"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.timesteps_log, self.legal_action_rates)
        plt.xlabel('Timesteps')
        plt.ylabel('Legal Action Rate')
        plt.title('Model Action Quality (Higher is Better)')
        plt.ylim(0, 1)
        plt.savefig(f"{self.log_dir}/plots/action_quality.png")
        plt.close()
    
    def on_training_end(self):
        """Clean up on training end"""
        self.csv_file.close()
        
        # Create final plot
        if len(self.timesteps_log) > 1:  # Only plot if we have multiple data points
            self._plot_action_quality()
        
        # Print statistics
        total_actions = self.legal_actions + self.illegal_actions
        legal_rate = self.legal_actions / total_actions if total_actions > 0 else 0
        
        print("\nAction Selection Quality:")
        print(f"Total legal actions: {self.legal_actions}")
        print(f"Total illegal actions: {self.illegal_actions}")
        print(f"Legal action rate: {legal_rate:.2%}") 