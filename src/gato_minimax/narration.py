"""Descripción en español de cada paso de la búsqueda.

Las casillas se muestran al usuario numeradas del 1 al 9 (por filas), aunque
internamente se usan índices de 0 a 8.
"""

from __future__ import annotations

from gato_minimax.game import Board, Player
from gato_minimax.search import (
    Bound,
    BranchesPruned,
    NodeResolved,
    NodeVisited,
    SearchEvent,
    ValuePropagated,
)
from gato_minimax.tree import ROOT_ID, SearchTree, format_value


def cell_label(move: int) -> str:
    """Número de casilla para el usuario (1 a 9)."""
    return str(move + 1)


def role(board: Board) -> str:
    """Rol del jugador en turno: ``"MAX (X)"`` o ``"MIN (O)"``."""
    return "MAX (X)" if board.to_move is Player.X else "MIN (O)"


def describe(event: SearchEvent | None, tree: SearchTree) -> str:
    """Explica qué ocurre en ``event`` y por qué."""
    match event:
        case None:
            return _describe_start(tree)
        case NodeVisited():
            return _describe_visit(event, tree)
        case ValuePropagated():
            return _describe_propagation(event, tree)
        case BranchesPruned():
            return _describe_pruning(event, tree)
        case NodeResolved():
            return _describe_resolution(event, tree)


def _window(alpha: float, beta: float) -> str:
    return f"α = {format_value(alpha)}, β = {format_value(beta)}"


def _describe_start(tree: SearchTree) -> str:
    root = tree.nodes[ROOT_ID].board
    algorithm = "minimax con poda alfa-beta" if tree.pruning else "minimax (sin poda)"
    goal = "mayor" if root.to_move is Player.X else "menor"
    return (
        f"Se ejecutará {algorithm} desde la posición mostrada. Le toca a {role(root)}, "
        f"que busca la jugada con el {goal} valor. Los valores siempre se miden desde la "
        "perspectiva de X: positivo = gana X, negativo = gana O, 0 = empate."
    )


def _describe_visit(event: NodeVisited, tree: SearchTree) -> str:
    board = event.board
    if event.parent_id is None:
        text = f"Se entra a la raíz. Le toca a {role(board)}."
    else:
        mover = board.to_move.opponent.value
        assert event.move is not None
        text = f"{mover} juega en la casilla {cell_label(event.move)} (profundidad {event.depth})."
        text += (
            " Es un estado terminal." if board.is_terminal else f" Ahora le toca a {role(board)}."
        )
    if tree.pruning:
        text += f" Hereda la ventana {_window(event.alpha, event.beta)}."
    return text


def _describe_propagation(event: ValuePropagated, tree: SearchTree) -> str:
    node = tree.nodes[event.node_id]
    child = tree.nodes[event.child_id]
    assert child.move is not None
    text = (
        f"{role(node.board)} recibe el valor {format_value(event.child_value)} "
        f"de la casilla {cell_label(child.move)}."
    )
    if event.improved:
        text += f" Es su mejor opción hasta ahora → valor del nodo = {format_value(event.value)}."
    else:
        text += f" No mejora su mejor opción actual ({format_value(event.value)})."
    if tree.pruning:
        updated = "α" if node.board.to_move is Player.X else "β"
        text += f" Se actualiza {updated}: {_window(event.alpha, event.beta)}."
    return text


def _describe_pruning(event: BranchesPruned, tree: SearchTree) -> str:
    node = tree.nodes[event.node_id]
    alpha, beta = format_value(event.alpha), format_value(event.beta)
    cells = ", ".join(cell_label(move) for move in event.pruned_moves)
    if node.board.to_move is Player.X:
        reason = (
            f"MAX ya asegura al menos {alpha} aquí, pero MIN, más arriba, ya tiene una "
            f"alternativa que vale {beta} o menos; MIN nunca elegirá este camino."
        )
    else:
        reason = (
            f"MIN ya puede forzar {beta} o menos aquí, pero MAX, más arriba, ya tiene "
            f"asegurado {alpha}; MAX nunca elegirá este camino."
        )
    branches = "rama" if len(event.pruned_moves) == 1 else "ramas"
    return (
        f"¡PODA! α = {alpha} ≥ β = {beta}. {reason} Se descartan sin explorar "
        f"{len(event.pruned_moves)} {branches}: casilla(s) {cells}."
    )


def _describe_resolution(event: NodeResolved, tree: SearchTree) -> str:
    node = tree.nodes[event.node_id]
    board = node.board
    value = format_value(event.value)

    if board.is_terminal:
        winner = board.winner
        if winner is None:
            return "Empate (tablero lleno) → utilidad = 0."
        empty = len(board.empty_cells)
        sign = "+" if winner is Player.X else "-"
        return (
            f"Gana {winner.value} con {empty} casilla(s) vacía(s) → "
            f"utilidad = {sign}({empty} + 1) = {value}."
        )

    assert event.best_move is not None
    best = cell_label(event.best_move)
    if event.node_id == ROOT_ID:
        return (
            f"DECISIÓN: {board.to_move.value} juega en la casilla {best}, con valor {value}. "
            "El camino resaltado es la secuencia de jugadas óptimas de ambos jugadores."
        )
    match event.bound:
        case Bound.EXACT:
            detail = f"valor = {value} (mejor jugada: casilla {best})"
        case Bound.LOWER:
            detail = (
                f"valor ≥ {value} (cota inferior). Ya alcanza β, así que MIN, más arriba, "
                "no elegirá este camino y su valor exacto no hace falta"
            )
        case Bound.UPPER:
            detail = (
                f"valor ≤ {value} (cota superior). No supera α, así que MAX, más arriba, "
                "no elegirá este camino y su valor exacto no hace falta"
            )
    return f"Nodo {role(board)} resuelto: {detail}. Este valor sube a su padre."
