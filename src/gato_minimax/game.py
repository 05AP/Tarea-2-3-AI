"""Modelo del juego del gato (tic-tac-toe).

Representación elegida:

* **Estado**: tablero inmutable de 9 casillas, indexadas de 0 a 8 por filas::

      0 | 1 | 2
      3 | 4 | 5
      6 | 7 | 8

  Cada casilla contiene ``Player.X``, ``Player.O`` o ``None`` (vacía). El turno
  no se guarda: se deduce del número de fichas, porque X siempre inicia.
* **Acción**: índice de una casilla vacía. Las acciones se generan en orden
  ascendente, lo que hace la búsqueda determinista y reproducible.
* **Utilidad** (solo en estados terminales, desde la perspectiva de X = MAX):
  ``+(casillas vacías + 1)`` si gana X, ``-(casillas vacías + 1)`` si gana O y
  ``0`` en empate. Sumar las casillas vacías premia ganar en menos jugadas y
  perder en más, sin alterar el orden victoria > empate > derrota.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

BOARD_SIZE: Final = 9

WINNING_LINES: Final[tuple[tuple[int, int, int], ...]] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)

_EMPTY_SYMBOLS: Final = frozenset(".-_")
_IGNORED_SYMBOLS: Final = frozenset(" \t\n/|")


class Player(Enum):
    """Jugador del gato. X es MAX y siempre mueve primero; O es MIN."""

    X = "X"
    O = "O"

    @property
    def opponent(self) -> Player:
        """Devuelve el jugador contrario."""
        return Player.O if self is Player.X else Player.X


type Cell = Player | None


@dataclass(frozen=True, slots=True)
class Board:
    """Estado inmutable del juego.

    Raises:
        ValueError: si el tablero no tiene 9 casillas o el número de fichas
            no corresponde a una partida alcanzable (X inicia y se alternan).
    """

    cells: tuple[Cell, ...] = (None,) * BOARD_SIZE

    def __post_init__(self) -> None:
        if len(self.cells) != BOARD_SIZE:
            msg = f"El tablero debe tener {BOARD_SIZE} casillas, tiene {len(self.cells)}."
            raise ValueError(msg)
        x_count = self.cells.count(Player.X)
        o_count = self.cells.count(Player.O)
        if x_count - o_count not in (0, 1):
            msg = f"Conteo de fichas inválido: X={x_count}, O={o_count} (X inicia y se alternan)."
            raise ValueError(msg)

    @classmethod
    def from_string(cls, text: str) -> Board:
        """Construye un tablero a partir de texto, leído por filas.

        Se aceptan ``X``/``O`` (sin distinguir mayúsculas) y ``.``, ``-`` o
        ``_`` para casillas vacías. Espacios, ``/`` y ``|`` se ignoran, de modo
        que ``"XO./.X./..O"`` y ``"XO..X...O"`` son equivalentes.

        Raises:
            ValueError: si hay símbolos desconocidos o el tablero es inválido.
        """
        cells: list[Cell] = []
        for symbol in text.upper():
            if symbol in _IGNORED_SYMBOLS:
                continue
            if symbol in _EMPTY_SYMBOLS:
                cells.append(None)
            elif symbol in (Player.X.value, Player.O.value):
                cells.append(Player(symbol))
            else:
                msg = f"Símbolo no válido en el tablero: {symbol!r}."
                raise ValueError(msg)
        return cls(tuple(cells))

    @property
    def to_move(self) -> Player:
        """Jugador al que le toca mover."""
        return Player.X if self.cells.count(Player.X) == self.cells.count(Player.O) else Player.O

    @property
    def empty_cells(self) -> tuple[int, ...]:
        """Índices de las casillas vacías, en orden ascendente."""
        return tuple(index for index, cell in enumerate(self.cells) if cell is None)

    @property
    def winner(self) -> Player | None:
        """Jugador con tres en línea, o ``None`` si nadie ha ganado."""
        for a, b, c in WINNING_LINES:
            cell = self.cells[a]
            if cell is not None and cell is self.cells[b] is self.cells[c]:
                return cell
        return None

    @property
    def is_terminal(self) -> bool:
        """``True`` si alguien ganó o ya no quedan casillas vacías."""
        return self.winner is not None or None not in self.cells

    def legal_moves(self) -> tuple[int, ...]:
        """Acciones disponibles; vacío si el estado es terminal."""
        return () if self.is_terminal else self.empty_cells

    def play(self, move: int) -> Board:
        """Devuelve el tablero resultante de que el jugador en turno marque ``move``.

        Raises:
            ValueError: si la jugada no es legal en este estado.
        """
        if move not in self.legal_moves():
            msg = f"Jugada ilegal: casilla {move}."
            raise ValueError(msg)
        cells = list(self.cells)
        cells[move] = self.to_move
        return Board(tuple(cells))

    def utility(self) -> int:
        """Valor de un estado terminal desde la perspectiva de X (MAX).

        Raises:
            ValueError: si el estado no es terminal.
        """
        if not self.is_terminal:
            msg = "La utilidad solo está definida para estados terminales."
            raise ValueError(msg)
        winner = self.winner
        if winner is None:
            return 0
        score = len(self.empty_cells) + 1
        return score if winner is Player.X else -score

    def __str__(self) -> str:
        symbols = [cell.value if cell is not None else "." for cell in self.cells]
        return "\n".join("".join(symbols[row : row + 3]) for row in range(0, BOARD_SIZE, 3))
