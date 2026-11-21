"""Árbol de búsqueda reconstruido a partir de los eventos, listo para dibujarse.

Este módulo no depende de la interfaz gráfica: convierte la secuencia de
eventos de :mod:`gato_minimax.search` en

* una estructura de árbol con posiciones fijas (``x`` en unidades de hoja,
  ``depth`` en niveles), calculada una sola vez para que los nodos no se
  muevan durante la animación, y
* "fotografías" (:class:`Snapshot`) del estado de la búsqueda después de
  aplicar los primeros ``step`` eventos.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from gato_minimax.game import Board
from gato_minimax.search import (
    Bound,
    BranchesPruned,
    NodeResolved,
    NodeVisited,
    SearchEvent,
    SearchResult,
    ValuePropagated,
)

ROOT_ID = 0


@dataclass(frozen=True, slots=True)
class TreeNode:
    """Nodo del árbol. ``pruned`` indica una rama que la poda dejó sin explorar."""

    node_id: int
    parent_id: int | None
    move: int | None
    board: Board
    depth: int
    pruned: bool
    children: tuple[int, ...]
    x: float


@dataclass(slots=True)
class NodeState:
    """Lo que se sabe de un nodo explorado en un momento dado de la búsqueda."""

    alpha: float
    beta: float
    value: int | None = None
    best_move: int | None = None
    bound: Bound | None = None

    @property
    def resolved(self) -> bool:
        """``True`` cuando el nodo ya devolvió su valor al padre."""
        return self.bound is not None


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Estado de la búsqueda tras aplicar los primeros ``step`` eventos."""

    step: int
    event: SearchEvent | None
    active_id: int | None
    visible: frozenset[int]
    states: Mapping[int, NodeState]
    nodes_visited: int
    cutoffs: int
    principal_variation: tuple[int, ...] = field(default=())


class SearchTree:
    """Árbol completo de una búsqueda registrada (``record=True``)."""

    def __init__(self, result: SearchResult, *, pruning: bool) -> None:
        if not result.events:
            msg = "La búsqueda no tiene eventos; ejecútala con record=True."
            raise ValueError(msg)
        self.result = result
        self.pruning = pruning
        self.events: tuple[SearchEvent, ...] = result.events
        self._placeholders_by_event: dict[int, tuple[int, ...]] = {}
        self.nodes: Mapping[int, TreeNode] = self._build()
        self.max_depth = max(node.depth for node in self.nodes.values())
        self.leaf_count = sum(1 for node in self.nodes.values() if not node.children)

    @property
    def step_count(self) -> int:
        """Número total de pasos (eventos) de la animación."""
        return len(self.events)

    def snapshot(self, step: int) -> Snapshot:
        """Reproduce los primeros ``step`` eventos y devuelve el estado resultante."""
        step = max(0, min(step, self.step_count))
        states: dict[int, NodeState] = {}
        visible: set[int] = set()
        active_id: int | None = None
        cutoffs = 0

        for index, event in enumerate(self.events[:step]):
            match event:
                case NodeVisited(node_id=node_id, alpha=alpha, beta=beta):
                    states[node_id] = NodeState(alpha, beta)
                    visible.add(node_id)
                    active_id = node_id
                case ValuePropagated(node_id=node_id, value=value, best_move=best_move):
                    state = states[node_id]
                    state.value, state.best_move = value, best_move
                    state.alpha, state.beta = event.alpha, event.beta
                    active_id = node_id
                case BranchesPruned(node_id=node_id):
                    visible.update(self._placeholders_by_event[index])
                    cutoffs += 1
                    active_id = node_id
                case NodeResolved(node_id=node_id, value=value, bound=bound, best_move=best_move):
                    state = states[node_id]
                    state.value, state.best_move, state.bound = value, best_move, bound
                    active_id = node_id

        finished = step == self.step_count
        return Snapshot(
            step=step,
            event=self.events[step - 1] if step else None,
            active_id=active_id,
            visible=frozenset(visible),
            states=states,
            nodes_visited=len(states),
            cutoffs=cutoffs,
            principal_variation=self._principal_variation() if finished else (),
        )

    def child_by_move(self, node_id: int, move: int) -> TreeNode | None:
        """Hijo de ``node_id`` al que se llega con ``move``, si existe."""
        for child_id in self.nodes[node_id].children:
            child = self.nodes[child_id]
            if child.move == move:
                return child
        return None

    def _principal_variation(self) -> tuple[int, ...]:
        """Camino de jugadas óptimas desde la raíz, según los mejores movimientos."""
        best_moves = {
            event.node_id: event.best_move
            for event in self.events
            if isinstance(event, NodeResolved)
        }
        path = [ROOT_ID]
        while (move := best_moves.get(path[-1])) is not None:
            child = self.child_by_move(path[-1], move)
            if child is None:
                break
            path.append(child.node_id)
        return tuple(path)

    def _build(self) -> dict[int, TreeNode]:
        """Crea los nodos (explorados y podados) y les asigna posición horizontal."""
        boards: dict[int, Board] = {}
        parents: dict[int, int | None] = {}
        moves: dict[int, int | None] = {}
        depths: dict[int, int] = {}
        pruned: set[int] = set()
        children: dict[int, list[int]] = {}
        next_placeholder_id = self.result.nodes_visited

        for index, event in enumerate(self.events):
            if isinstance(event, NodeVisited):
                node_id = event.node_id
                boards[node_id], moves[node_id] = event.board, event.move
                parents[node_id], depths[node_id] = event.parent_id, event.depth
                children[node_id] = []
                if event.parent_id is not None:
                    children[event.parent_id].append(node_id)
            elif isinstance(event, BranchesPruned):
                created: list[int] = []
                parent_board = boards[event.node_id]
                for move in event.pruned_moves:
                    node_id = next_placeholder_id
                    next_placeholder_id += 1
                    boards[node_id], moves[node_id] = parent_board.play(move), move
                    parents[node_id], depths[node_id] = event.node_id, depths[event.node_id] + 1
                    children[node_id] = []
                    children[event.node_id].append(node_id)
                    pruned.add(node_id)
                    created.append(node_id)
                self._placeholders_by_event[index] = tuple(created)

        positions = _layout(children)
        return {
            node_id: TreeNode(
                node_id=node_id,
                parent_id=parents[node_id],
                move=moves[node_id],
                board=boards[node_id],
                depth=depths[node_id],
                pruned=node_id in pruned,
                children=tuple(children[node_id]),
                x=positions[node_id],
            )
            for node_id in boards
        }


def _layout(children: Mapping[int, list[int]]) -> dict[int, float]:
    """Asigna ``x`` a cada nodo: las hojas en posiciones consecutivas y cada
    padre centrado sobre su primer y último hijo (recorrido en post-orden)."""
    positions: dict[int, float] = {}
    next_leaf = 0

    def place(node_id: int) -> float:
        nonlocal next_leaf
        kids = children[node_id]
        if not kids:
            positions[node_id] = float(next_leaf)
            next_leaf += 1
        else:
            xs = [place(child_id) for child_id in kids]
            positions[node_id] = (xs[0] + xs[-1]) / 2
        return positions[node_id]

    place(ROOT_ID)
    return positions


def format_value(value: float) -> str:
    """Formatea un valor o una cota de la ventana: ``+3``, ``0``, ``-2``, ``+∞``, ``-∞``."""
    if math.isinf(value):
        return "+∞" if value > 0 else "-∞"
    return f"{int(value):+d}" if value else "0"
