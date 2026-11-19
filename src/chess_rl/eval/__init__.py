"""Evaluation: players, games, matches, Elo."""

from chess_rl.eval.game import GameResult, play_game
from chess_rl.eval.players import Clock, MaterialGreedyPlayer, Player, RandomPlayer, StockfishPlayer

__all__ = ["Clock", "GameResult", "MaterialGreedyPlayer", "Player", "RandomPlayer", "StockfishPlayer", "play_game"]
