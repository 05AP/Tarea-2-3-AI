import itertools

import pytest

from gato_minimax.__main__ import DEFAULT_BOARD
from gato_minimax.game import Board, Player
from gato_minimax.search import Bound, NodeResolved, NodeVisited, alphabeta, minimax


def _positions_with_empty_cells(empty: int) -> list[Board]:
    """Todas las posiciones válidas y no terminales con ``empty`` casillas vacías."""
    filled = 9 - empty
    x_count, o_count = (filled + 1) // 2, filled // 2
    boards = []
    for xs in itertools.combinations(range(9), x_count):
        free = [i for i in range(9) if i not in xs]
        for os in itertools.combinations(free, o_count):
            cells = [Player.X if i in xs else Player.O if i in os else None for i in range(9)]
            board = Board(tuple(cells))
            if not board.is_terminal:
                boards.append(board)
    return boards


def test_empty_board_is_a_draw() -> None:
    assert alphabeta(Board()).value == 0


def test_takes_the_immediate_win() -> None:
    result = alphabeta(Board.from_string("XX./OO./..."))

    assert result.best_move == 2
    assert result.value == 5  # gana X dejando 4 casillas vacías: +(4 + 1)


def test_blocks_the_opponent_threat() -> None:
    result = alphabeta(Board.from_string("XX./O../..."))

    assert result.best_move == 2


@pytest.mark.parametrize("empty", [3, 4, 5])
def test_alphabeta_matches_minimax_with_fewer_nodes(empty: int) -> None:
    for board in _positions_with_empty_cells(empty):
        full, pruned = minimax(board), alphabeta(board)

        assert (pruned.value, pruned.best_move) == (full.value, full.best_move), str(board)
        assert pruned.nodes_visited <= full.nodes_visited


def test_recorded_bounds_are_consistent_with_true_values() -> None:
    result = alphabeta(Board.from_string(DEFAULT_BOARD), record=True)
    boards = {e.node_id: e.board for e in result.events if isinstance(e, NodeVisited)}

    assert result.cutoffs > 0
    for event in result.events:
        if isinstance(event, NodeResolved):
            true_value = minimax(boards[event.node_id]).value
            match event.bound:
                case Bound.EXACT:
                    assert true_value == event.value
                case Bound.LOWER:
                    assert true_value >= event.value
                case Bound.UPPER:
                    assert true_value <= event.value
