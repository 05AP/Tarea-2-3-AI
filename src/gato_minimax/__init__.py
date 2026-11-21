"""Minimax con poda alfa-beta para el gato, con visualización del árbol de búsqueda."""

from gato_minimax.game import Board, Player
from gato_minimax.search import SearchResult, alphabeta, minimax

__all__ = ["Board", "Player", "SearchResult", "alphabeta", "minimax"]
__version__ = "1.0.0"
