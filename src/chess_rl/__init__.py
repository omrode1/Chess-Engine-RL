"""chess_rl: a laptop-trained chess engine built to beat one specific human by modelling them."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("chess-rl")
except PackageNotFoundError:  # running from a source checkout without installing
    __version__ = "0.0.0+unknown"
