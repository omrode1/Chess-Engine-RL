# Chess Engine Reinforcement Learning

A deep reinforcement learning project that teaches an AI agent to play chess through self-play and reward optimization, enhanced with opening book knowledge.

![Chess Board](https://github.com/favicon.ico) <!-- Replace with an actual chess image if you have one -->

## Overview

This project implements a reinforcement learning environment for training chess-playing AI agents using the Deep Q-Network (DQN) algorithm from Stable Baselines3. The agent learns chess strategy through play experience and a carefully designed reward function that encodes chess principles, supplemented by opening book knowledge.

## Features

- Custom OpenAI Gym environment for chess
- Deep Q-Learning implementation with PyTorch backend
- Opening book knowledge to guide early game play
- Convolutional Neural Network specifically designed for chess
- Action masking to ensure legal moves
- Reward system modeling good chess principles:
  - Material advantage
  - Center control
  - Piece development
  - King safety (castling)
  - Opening theory adherence
  - Check/checkmate rewards
- Penalties for suboptimal play:
  - Position repetition penalties
  - Move oscillation detection
  - Early game mistakes
- Automatic PGN generation for game analysis
- Visualization of the agent's progress
- Detailed metrics tracking for performance analysis

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
├── src/                        # Source code
│   ├── __init__.py
│   └── chess_environment.py    # Custom chess environment with opening book
├── utils/                      # Utilities
│   ├── __init__.py
│   ├── opening_book.py         # Chess opening book knowledge
│   └── custom_callbacks.py     # Custom callbacks for metrics and logging
├── main.py                     # Main entry point for the project
├── train_agent.py              # Script to train the RL agent
├── test_agent.py               # Script to test the trained agent
├── models_with_opening_book/   # Saved model checkpoints
├── logs_with_opening_book/     # Training logs
├── pgn_games/                  # Saved chess games in PGN format
└── legacy/                     # Archived files from previous versions
    ├── models/
    └── logs/
```

## How It Works

### The Chess Environment

The project uses a custom Gym environment (`src/chess_environment.py`) that:

1. Represents the chess board as a 12×8×8 tensor (6 piece types × 2 colors × 8×8 board)
2. Handles the action space as indices into the list of legal moves
3. Integrates opening book knowledge for the first 10 moves
4. Manages game state, legal moves, and rewards

### Opening Book Knowledge

The agent can leverage established chess opening theory:

- Popular openings like Ruy Lopez, Sicilian Defense, Queen's Gambit, etc.
- Bonus rewards for following established opening lines
- Weighted selection of opening moves based on popularity
- Configurable depth for opening book guidance

### The Reward System

The agent learns through a sophisticated reward system that encourages good chess principles:

- **Opening Book Adherence**: Bonus reward for following opening theory
- **Material Balance**: Capturing opponent pieces (+) and avoiding captures (-)
- **Position Quality**:
  - Center control: Reward for controlling center squares
  - Castling: Significant bonus for completing castling
  - Check: Reward for putting the opponent in check
- **Development**: 
  - Rewards for developing knights and bishops in the opening
  - Rewards for piece mobility and activity
- **Pawn Structure**: Rewards for protected pawns and good pawn structure
- **Penalties**:
  - Penalties for position repetition
  - Penalties for stalemate
- **Winning/Losing**:
  - +5.0 for checkmate (win)
  - -5.0 for being checkmated (loss)

### Neural Network Architecture

A custom Convolutional Neural Network is used to process the chess board:

- Convolutional layers to capture spatial patterns on the board
- Fully connected layers for strategic decision making
- Custom feature extractor designed specifically for chess positions

### Training Process

The agent is trained using the DQN algorithm with the following parameters:

- Learning rate: 0.0005
- Batch size: 128
- Replay buffer size: 100,000
- Exploration strategy: ε-greedy with decay (final epsilon: 0.05)
- Total training steps: 1,000,000

## Usage

### Command-Line Interface

The project provides a convenient command-line interface:

```bash
# Train an agent
python main.py train [--timesteps TIMESTEPS] [--model-dir MODEL_DIR] [--log-dir LOG_DIR] [--no-opening-book]

# Test an agent
python main.py test [--model MODEL] [--games GAMES] [--max-moves MAX_MOVES] [--save-pgn] [--pgn-dir PGN_DIR] [--visualize] [--delay DELAY] [--no-opening-book]
```

### Training a Chess Agent

```bash
python main.py train
```

This will start the training process, periodically saving checkpoints and logs.

### Testing a Trained Agent

```bash
python main.py test --visualize
```

This will load the trained model and have it play a game, visualizing the board after each move.

### Analyzing Games

The saved PGN files can be analyzed with any chess analysis software or website, such as:
- [Lichess](https://lichess.org/analysis)
- [Chess.com](https://www.chess.com/analysis)
- [SCID](http://scid.sourceforge.net/)

## Results

The trained agent demonstrates:
1. Understanding of established opening theory
2. Sound material value assessment
3. Tactical awareness (capturing pieces, avoiding captures)
4. Strategic understanding of piece development and center control
5. Improved play over time as shown by metrics

## Future Improvements

- Implement a self-play training loop to enhance learning
- Add Monte Carlo Tree Search for more sophisticated play
- Implement Proximal Policy Optimization (PPO) for better sample efficiency
- Enhance the neural network architecture for better spatial understanding
- Create a chess engine interface for playing against humans

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