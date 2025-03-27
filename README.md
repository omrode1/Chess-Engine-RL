# Chess Engine Reinforcement Learning

A deep reinforcement learning project that teaches an AI agent to play chess through self-play and reward optimization.

![Chess Board](https://github.com/favicon.ico) <!-- Replace with an actual chess image if you have one -->

## Overview

This project implements a reinforcement learning environment for training chess-playing AI agents using the Deep Q-Network (DQN) algorithm from Stable Baselines3. The agent learns chess strategy entirely through play experience and a carefully designed reward function that encodes chess principles.

## Features

- Custom OpenAI Gym environment for chess
- Deep Q-Learning implementation with PyTorch backend
- Reward system modeling good chess principles:
  - Material advantage
  - Center control
  - Piece development
  - King safety (castling)
  - Check/checkmate rewards
- Penalties for suboptimal play:
  - Position repetition penalties
  - Move oscillation detection
  - Early game mistakes
- Automatic PGN generation for game analysis
- Visualization of the agent's progress

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Chess-Engine-RL.git
cd Chess-Engine-RL
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
Chess-Engine-RL/
├── chess_gym_env.py      # Custom chess environment
├── train_rl_agent.py     # Script to train the RL agent
├── test_rl_agent.py      # Script to test the trained agent
├── models/               # Saved model checkpoints
├── logs/                 # Training logs
└── pgn_games/            # Saved chess games in PGN format
```

## How It Works

### The Chess Environment

The project uses a custom Gym environment (`chess_gym_env.py`) that:

1. Represents the chess board as a 12×8×8 tensor (6 piece types × 2 colors × 8×8 board)
2. Handles the action space as indices into the list of legal moves
3. Manages game state, legal moves, and rewards

### The Reward System

The agent learns through a sophisticated reward system that encourages good chess principles:

- **Material Balance**: Capturing opponent pieces (+) and avoiding captures (-)
- **Position Quality**:
  - Center control: +0.2 for each piece in the center squares (d4, d5, e4, e5)
  - Castling: +1.0 for completing castling
  - Check: +0.1 for putting the opponent in check
- **Development**: 
  - +0.2 for developing knights and bishops in the opening
  - +0.05 for piece development (non-pawn, non-king) in the opening
- **Penalties**:
  - -0.3/-0.6 for position repetition (2/3 times)
  - -1.0 for move oscillation (back and forth)
  - -0.1 per move for moving the same piece repeatedly
  - -0.3 for early side pawn movement
  - -2.0 for causing stalemate
- **Winning/Losing**:
  - +10.0 for checkmate (win)
  - -10.0 for being checkmated (loss)

### Training Process

The agent is trained using the DQN algorithm with the following parameters:

- Learning rate: 0.0001
- Batch size: 64
- Replay buffer size: 50,000
- Exploration strategy: ε-greedy with decay
- Total training steps: 500,000

## Usage

### Training a Chess Agent

```bash
python train_rl_agent.py
```

This will start the training process, periodically saving checkpoints to the `models/` directory and logs to the `logs/` directory.

### Testing a Trained Agent

```bash
python test_rl_agent.py
```

This will load the latest trained model and have it play a game against itself. The game will be saved in PGN format in the `pgn_games/` directory.

### Analyzing Games

The saved PGN files can be analyzed with any chess analysis software or website, such as:
- [Lichess](https://lichess.org/analysis)
- [Chess.com](https://www.chess.com/analysis)
- [SCID](http://scid.sourceforge.net/)

## Technical Details

### Neural Network Architecture

The DQN agent uses a Multi-Layer Perceptron policy with:
- Input: 12×8×8 tensor (board state)
- Hidden layers defined by Stable Baselines3
- Output: Q-values for each possible action

### Observation Space

The observation space is a 12×8×8 tensor where:
- 6 channels for each white piece type (pawn, knight, bishop, rook, queen, king)
- 6 channels for each black piece type
- Each channel is an 8×8 grid where 1 indicates presence of the piece

### Action Space

The action space is a single integer in range(0, 218), which indexes into the list of legal moves for the current position.

## Results

The trained agent demonstrates:
1. Understanding of material value
2. Basic opening principles (center control, development)
3. Tactical awareness (capturing pieces, avoiding captures)
4. Improved play over time

## Future Improvements

- Implement a self-play training loop to enhance learning
- Add Monte Carlo Tree Search for more sophisticated play
- Implement Proximal Policy Optimization (PPO) for better sample efficiency
- Enhance the neural network architecture with a CNN for better spatial understanding
- Add a proper evaluation function based on classic chess engines

## License

[MIT License](LICENSE)

## Acknowledgments

- [python-chess](https://python-chess.readthedocs.io/) for the chess logic
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) for the reinforcement learning algorithms
- [OpenAI Gym](https://www.gymlibrary.dev/) for the environment interface

## Contact

For questions or feedback, please open an issue or contact [your-email@example.com].

---

Happy training! 