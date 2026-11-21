"""Minimax con poda alfa-beta opcional y registro del recorrido.

Ambos algoritmos comparten una sola implementación recursiva. En ``minimax``
la ventana ``(α, β)`` se queda en ``(-∞, +∞)`` y nunca hay corte; en
``alphabeta`` la ventana se estrecha con cada hijo y se corta cuando
``alpha >= beta``. Así, la poda es la única diferencia entre los dos.

Cuando se pide ``record=True``, la búsqueda guarda una secuencia ordenada de
eventos (:data:`SearchEvent`) que describe el proceso interno: qué nodo se
visita, qué valor sube de cada hijo a su padre, cómo cambian α y β, y qué
ramas se podan y por qué. El visualizador reproduce esos eventos paso a paso.

Convenciones:

* X es MAX y O es MIN; los valores siempre están en la perspectiva de X.
* α es el mejor valor que MAX ya tiene asegurado en el camino actual y β el
  mejor que MIN ya tiene asegurado. Si ``α >= β`` en un nodo, el jugador de
  arriba nunca permitirá llegar a él, así que sus hijos restantes se podan.
* Se usa la variante *fail-soft*: el valor que devuelve un nodo cuya ventana
  ``(α, β)`` no lo contiene es una cota (:class:`Bound`), no un valor exacto.
  El valor de la raíz siempre es exacto porque inicia con ``(-∞, +∞)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from gato_minimax.game import Board, Player


class Bound(Enum):
    """Qué tanto se sabe del valor devuelto por un nodo."""

    EXACT = "="
    LOWER = "≥"  # el valor real es al menos este (corte en MAX, falla alta)
    UPPER = "≤"  # el valor real es a lo más este (corte en MIN, falla baja)


@dataclass(frozen=True, slots=True)
class NodeVisited:
    """Se entra a un nodo con la ventana ``(alpha, beta)`` heredada del padre."""

    node_id: int
    parent_id: int | None
    move: int | None
    board: Board
    depth: int
    alpha: float
    beta: float


@dataclass(frozen=True, slots=True)
class ValuePropagated:
    """Un hijo devolvió ``child_value`` y el padre actualizó su mejor valor y su ventana."""

    node_id: int
    child_id: int
    child_value: int
    value: int
    best_move: int
    improved: bool
    alpha: float
    beta: float


@dataclass(frozen=True, slots=True)
class BranchesPruned:
    """Se cumplió ``alpha >= beta`` y las jugadas ``pruned_moves`` no se exploran."""

    node_id: int
    pruned_moves: tuple[int, ...]
    alpha: float
    beta: float


@dataclass(frozen=True, slots=True)
class NodeResolved:
    """El nodo terminó: su valor (exacto o cota) sube al padre."""

    node_id: int
    value: int
    bound: Bound
    best_move: int | None


type SearchEvent = NodeVisited | ValuePropagated | BranchesPruned | NodeResolved


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Resultado de una búsqueda desde la raíz."""

    value: int
    best_move: int | None
    nodes_visited: int
    cutoffs: int
    events: tuple[SearchEvent, ...]


def minimax(board: Board, *, record: bool = False) -> SearchResult:
    """Explora el árbol completo con minimax, sin poda."""
    return _Search(pruning=False, record=record).run(board)


def alphabeta(board: Board, *, record: bool = False) -> SearchResult:
    """Explora el árbol con minimax y poda alfa-beta."""
    return _Search(pruning=True, record=record).run(board)


class _Search:
    """Estado de una sola búsqueda: contadores y eventos registrados."""

    def __init__(self, *, pruning: bool, record: bool) -> None:
        self._pruning = pruning
        self._events: list[SearchEvent] | None = [] if record else None
        self._nodes_visited = 0
        self._cutoffs = 0

    def run(self, board: Board) -> SearchResult:
        value, best_move = self._visit(
            board, depth=0, alpha=-math.inf, beta=math.inf, parent_id=None, move=None
        )
        return SearchResult(
            value=value,
            best_move=best_move,
            nodes_visited=self._nodes_visited,
            cutoffs=self._cutoffs,
            events=tuple(self._events or ()),
        )

    def _emit(self, event: SearchEvent) -> None:
        if self._events is not None:
            self._events.append(event)

    def _visit(
        self,
        board: Board,
        *,
        depth: int,
        alpha: float,
        beta: float,
        parent_id: int | None,
        move: int | None,
    ) -> tuple[int, int | None]:
        node_id = self._nodes_visited
        self._nodes_visited += 1
        self._emit(NodeVisited(node_id, parent_id, move, board, depth, alpha, beta))

        if board.is_terminal:
            value = board.utility()
            self._emit(NodeResolved(node_id, value, Bound.EXACT, None))
            return value, None

        entry_alpha, entry_beta = alpha, beta
        maximizing = board.to_move is Player.X
        best_value: int | None = None
        best_move: int | None = None
        moves = board.legal_moves()

        for index, child_move in enumerate(moves):
            child_id = self._nodes_visited
            child_value, _ = self._visit(
                board.play(child_move),
                depth=depth + 1,
                alpha=alpha,
                beta=beta,
                parent_id=node_id,
                move=child_move,
            )

            improved = (
                best_value is None
                or (maximizing and child_value > best_value)
                or (not maximizing and child_value < best_value)
            )
            if improved:
                best_value, best_move = child_value, child_move
            assert best_value is not None and best_move is not None

            if self._pruning:
                if maximizing:
                    alpha = max(alpha, best_value)
                else:
                    beta = min(beta, best_value)
            self._emit(
                ValuePropagated(
                    node_id, child_id, child_value, best_value, best_move, improved, alpha, beta
                )
            )

            remaining = moves[index + 1 :]
            if alpha >= beta and remaining:
                self._cutoffs += 1
                self._emit(BranchesPruned(node_id, remaining, alpha, beta))
                break

        assert best_value is not None, "un nodo no terminal tiene al menos un hijo"
        bound = _classify(best_value, entry_alpha, entry_beta)
        self._emit(NodeResolved(node_id, best_value, bound, best_move))
        return best_value, best_move


def _classify(value: int, alpha: float, beta: float) -> Bound:
    """Clasifica un valor fail-soft respecto a la ventana con que se entró al nodo."""
    if value <= alpha:
        return Bound.UPPER
    if value >= beta:
        return Bound.LOWER
    return Bound.EXACT
