#!/usr/bin/env python3
"""
Chess Engine with Reinforcement Learning

This is the main entry point for the Chess Engine RL project.
It provides a command-line interface to train or test the agent.
"""

import argparse
import os
import sys

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Chess Engine with Reinforcement Learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the agent')
    train_parser.add_argument('--timesteps', type=int, default=1000000,
                             help='Total training timesteps')
    train_parser.add_argument('--model-dir', type=str, default='models_with_opening_book',
                             help='Directory to save models')
    train_parser.add_argument('--log-dir', type=str, default='logs_with_opening_book',
                             help='Directory to save logs')
    train_parser.add_argument('--no-opening-book', action='store_true',
                             help='Disable opening book knowledge')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test the agent')
    test_parser.add_argument('--model', type=str, 
                            default='models_with_opening_book/chess_with_opening_book_final',
                            help='Path to the trained model')
    test_parser.add_argument('--games', type=int, default=5, 
                            help='Number of games to play')
    test_parser.add_argument('--max-moves', type=int, default=50, 
                            help='Maximum moves per game')
    test_parser.add_argument('--save-pgn', action='store_true', 
                            help='Save games in PGN format')
    test_parser.add_argument('--pgn-dir', type=str, default='pgn_games', 
                            help='Directory to save PGN files')
    test_parser.add_argument('--visualize', action='store_true', 
                            help='Visualize the board after each move')
    test_parser.add_argument('--delay', type=float, default=0.5, 
                            help='Delay between moves when visualizing (seconds)')
    test_parser.add_argument('--no-opening-book', action='store_true',
                            help='Disable opening book knowledge for testing')
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_args()
    
    if args.command == 'train':
        print("Starting training mode...")
        # Import train module here to avoid importing everything at startup
        from train_agent import train_model
        
        train_model(
            total_timesteps=args.timesteps,
            model_dir=args.model_dir,
            log_dir=args.log_dir,
            use_opening_book=not args.no_opening_book
        )
        
    elif args.command == 'test':
        print("Starting testing mode...")   
        # Import test module here to avoid importing everything at startup
        from test_agent import test_model
        
        test_model(
            model_path=args.model,
            num_games=args.games,
            max_moves=args.max_moves,
            save_pgn=args.save_pgn,
            pgn_dir=args.pgn_dir,
            visualize=args.visualize,
            delay=args.delay,
            use_opening_book=not args.no_opening_book
        )
        
    else:
        print("Please specify a command: train or test")
        print("Run with --help for more information")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 