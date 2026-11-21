"""Punto de entrada: ``python -m gato_minimax`` o ``gato-minimax``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Final

from gato_minimax.game import Board

DEFAULT_BOARD: Final = "XO./X.X/O.."
"""Posición de la demo: le toca a O y debe bloquear la amenaza de X en la casilla 5."""

MAX_EMPTY_CELLS: Final = 5
"""Límite para que el árbol completo quepa legible en pantalla."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gato-minimax",
        description="Visualiza paso a paso minimax (con o sin poda alfa-beta) en el gato.",
    )
    parser.add_argument(
        "-t",
        "--tablero",
        default=DEFAULT_BOARD,
        help=(
            "posición inicial, por filas: X, O y '.' para vacía; '/' separa filas "
            f"(por defecto: {DEFAULT_BOARD!r})"
        ),
    )
    parser.add_argument(
        "--sin-poda",
        action="store_true",
        help="usa minimax puro, sin poda alfa-beta, para comparar",
    )
    return parser


def parse_board(parser: argparse.ArgumentParser, text: str) -> Board:
    """Valida la posición recibida por línea de comandos."""
    try:
        board = Board.from_string(text)
    except ValueError as error:
        parser.error(str(error))
    if board.is_terminal:
        parser.error("La posición ya es terminal: no hay jugadas que buscar.")
    if len(board.empty_cells) > MAX_EMPTY_CELLS:
        parser.error(
            f"Para que el árbol sea legible, usa una posición con a lo más "
            f"{MAX_EMPTY_CELLS} casillas vacías."
        )
    return board


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    board = parse_board(parser, args.tablero)

    from gato_minimax.app import run

    run(board, pruning=not args.sin_poda)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
