"""Visualizador paso a paso del árbol de búsqueda (tkinter).

La ventana reproduce los eventos registrados por la búsqueda:

* cada nodo es un mini-tablero; el borde indica quién mueve (azul = MAX/X,
  rojo = MIN/O) y la casilla sombreada es la jugada que llevó a ese estado;
* debajo de cada nodo se muestra su valor (``v = +3``, o ``v ≥ +3`` / ``v ≤ -2``
  si es una cota) y, mientras está activo, su ventana ``α``/``β``;
* las ramas podadas aparecen en gris punteado con una ✕;
* al terminar, el camino óptimo se resalta en verde.
"""

from __future__ import annotations

import sys
import tkinter as tk
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from tkinter import ttk
from typing import Final

from gato_minimax.game import BOARD_SIZE, Board, Player
from gato_minimax.narration import describe, role
from gato_minimax.search import Bound, alphabeta, minimax
from gato_minimax.tree import ROOT_ID, NodeState, SearchTree, Snapshot, TreeNode, format_value

FONT_FAMILY: Final = "Segoe UI"


@dataclass(frozen=True, slots=True)
class Palette:
    """Colores de la interfaz."""

    background: str = "#ffffff"
    grid: str = "#c9ced6"
    text: str = "#1f2328"
    muted: str = "#6e7781"
    max_player: str = "#0969da"
    max_fill: str = "#ddf4ff"
    min_player: str = "#cf222e"
    min_fill: str = "#ffebe9"
    last_move: str = "#fff8c5"
    active: str = "#fb8f44"
    pruned: str = "#afb8c1"
    pruned_fill: str = "#eaeef2"
    principal: str = "#1a7f37"


COLORS: Final = Palette()


@dataclass(frozen=True, slots=True)
class Geometry:
    """Medidas del dibujo del árbol a escala 1 (en píxeles)."""

    cell: float = 16
    leaf_spacing: float = 60
    level_height: float = 132
    label_height: float = 14
    text_height: float = 40
    margin: float = 24
    max_scale: float = 1.5

    @property
    def board(self) -> float:
        return self.cell * 3

    @property
    def node_height(self) -> float:
        return self.label_height + self.board + self.text_height


class TreeCanvas(tk.Canvas):
    """Lienzo que dibuja una :class:`Snapshot` del árbol, escalado a su tamaño."""

    def __init__(self, master: tk.Misc, tree: SearchTree) -> None:
        super().__init__(master, background=COLORS.background, highlightthickness=0)
        self._tree = tree
        self._geometry = Geometry()
        self._snapshot: Snapshot | None = None
        self._scale = 1.0
        self.bind("<Configure>", lambda _event: self._redraw())

    def show(self, snapshot: Snapshot) -> None:
        """Dibuja el estado de la búsqueda indicado."""
        self._snapshot = snapshot
        self._redraw()

    # -- Geometría -------------------------------------------------------------

    def _update_scale(self) -> None:
        geometry = self._geometry
        needed_width = self._tree.leaf_count * geometry.leaf_spacing + 2 * geometry.margin
        needed_height = (
            self._tree.max_depth * geometry.level_height
            + geometry.node_height
            + 2 * geometry.margin
        )
        width, height = max(self.winfo_width(), 1), max(self.winfo_height(), 1)
        self._scale = min(geometry.max_scale, width / needed_width, height / needed_height)

    def _origin(self, node: TreeNode) -> tuple[float, float]:
        """Esquina superior izquierda del tablero de ``node``."""
        g, s = self._geometry, self._scale
        tree_width = self._tree.leaf_count * g.leaf_spacing * s
        offset_x = (self.winfo_width() - tree_width) / 2
        center_x = offset_x + (node.x + 0.5) * g.leaf_spacing * s
        top = (g.margin + node.depth * g.level_height + g.label_height) * s
        return center_x - g.board * s / 2, top

    def _font(self, pixels: float, *, bold: bool = False) -> tuple[str, int, str]:
        """Fuente en píxeles (tamaño negativo en Tk) para que escale igual que el dibujo."""
        size = -max(8, round(pixels * self._scale))
        return (FONT_FAMILY, size, "bold" if bold else "normal")

    # Dibujo

    def _redraw(self) -> None:
        self.delete("all")
        snapshot = self._snapshot
        if snapshot is None:
            return
        self._update_scale()
        principal = set(snapshot.principal_variation)
        for node_id in sorted(snapshot.visible):
            node = self._tree.nodes[node_id]
            if node.parent_id is not None:
                on_path = node_id in principal and node.parent_id in principal
                self._draw_edge(node, on_path=on_path)
        for node_id in sorted(snapshot.visible):
            self._draw_node(self._tree.nodes[node_id], snapshot, principal)

    def _draw_edge(self, node: TreeNode, *, on_path: bool) -> None:
        assert node.parent_id is not None
        g, s = self._geometry, self._scale
        parent = self._tree.nodes[node.parent_id]
        px, py = self._origin(parent)
        cx, cy = self._origin(node)
        start = (px + g.board * s / 2, py + (g.board + g.text_height) * s)
        end = (cx + g.board * s / 2, cy - g.label_height * s)
        if node.pruned:
            self.create_line(*start, *end, fill=COLORS.pruned, width=1.5 * s, dash=(4, 3))
            mid_x, mid_y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
            self.create_text(
                mid_x, mid_y, text="✕", fill=COLORS.min_player, font=self._font(18, bold=True)
            )
        else:
            color, width = (COLORS.principal, 4) if on_path else (COLORS.muted, 1.2)
            self.create_line(*start, *end, fill=color, width=width * s)

    def _draw_node(self, node: TreeNode, snapshot: Snapshot, principal: set[int]) -> None:
        g, s = self._geometry, self._scale
        x, y = self._origin(node)
        size = g.board * s
        state = snapshot.states.get(node.node_id)
        maximizing = node.board.to_move is Player.X
        player_color = COLORS.max_player if maximizing else COLORS.min_player

        if node.node_id == snapshot.active_id:
            pad = 5 * s
            self.create_rectangle(
                x - pad, y - pad, x + size + pad, y + size + pad, fill=COLORS.active, outline=""
            )

        if node.pruned:
            label, label_color, border, border_width = "podada", COLORS.pruned, COLORS.pruned, 1.5
        elif node.board.is_terminal:
            label, label_color, border, border_width = "FIN", COLORS.muted, COLORS.text, 1.5
        else:
            label = "MAX" if maximizing else "MIN"
            label_color, border, border_width = player_color, player_color, 2.5
        if node.node_id in principal:
            border, border_width = COLORS.principal, 4

        self.create_text(
            x + size / 2,
            y - 2 * s,
            text=label,
            anchor="s",
            fill=label_color,
            font=self._font(12, bold=True),
        )
        fill = COLORS.background
        if state is not None and state.resolved and not node.board.is_terminal:
            fill = COLORS.max_fill if maximizing else COLORS.min_fill
        self._draw_board(node, x, y, fill=fill, faded=node.pruned)
        self.create_rectangle(
            x,
            y,
            x + size,
            y + size,
            outline=border,
            width=border_width * s,
            dash=(3, 2) if node.pruned else (),
        )
        if state is not None:
            self._draw_state_text(x + size / 2, y + size + 3 * s, state)

    def _draw_board(self, node: TreeNode, x: float, y: float, *, fill: str, faded: bool) -> None:
        cell = self._geometry.cell * self._scale
        self.create_rectangle(x, y, x + cell * 3, y + cell * 3, fill=fill, outline="")
        if node.move is not None:
            row, col = divmod(node.move, 3)
            self.create_rectangle(
                x + col * cell,
                y + row * cell,
                x + (col + 1) * cell,
                y + (row + 1) * cell,
                fill=COLORS.pruned_fill if faded else COLORS.last_move,
                outline="",
            )
        for i in (1, 2):
            self.create_line(x + i * cell, y, x + i * cell, y + 3 * cell, fill=COLORS.grid)
            self.create_line(x, y + i * cell, x + 3 * cell, y + i * cell, fill=COLORS.grid)
        for index in range(BOARD_SIZE):
            piece = node.board.cells[index]
            if piece is None:
                continue
            row, col = divmod(index, 3)
            color = COLORS.max_player if piece is Player.X else COLORS.min_player
            self.create_text(
                x + (col + 0.5) * cell,
                y + (row + 0.5) * cell,
                text=piece.value,
                fill=COLORS.muted if faded else color,
                font=self._font(14, bold=True),
            )

    def _draw_state_text(self, center_x: float, top: float, state: NodeState) -> None:
        lines: list[tuple[str, str]] = []
        if state.resolved:
            assert state.value is not None and state.bound is not None
            symbol = "=" if state.bound is Bound.EXACT else state.bound.value
            lines.append((f"v {symbol} {format_value(state.value)}", COLORS.text))
        elif state.value is not None:
            lines.append((f"mejor: {format_value(state.value)}", COLORS.muted))
        if self._tree.pruning and not state.resolved:
            lines.append(
                (f"α {format_value(state.alpha)}  β {format_value(state.beta)}", COLORS.muted)
            )
        for offset, (text, color) in enumerate(lines):
            self.create_text(
                center_x,
                top + offset * 16 * self._scale,
                text=text,
                anchor="n",
                fill=color,
                font=self._font(13, bold=offset == 0 and state.resolved),
            )


class BoardPreview(tk.Canvas):
    """Tablero grande con el estado del nodo activo."""

    def __init__(self, master: tk.Misc, *, cell: int) -> None:
        self._cell = cell
        size = cell * 3 + 2
        super().__init__(
            master, width=size, height=size, background=COLORS.background, highlightthickness=0
        )

    def show(self, board: Board, last_move: int | None) -> None:
        self.delete("all")
        cell = self._cell
        if last_move is not None:
            row, col = divmod(last_move, 3)
            self.create_rectangle(
                1 + col * cell,
                1 + row * cell,
                1 + (col + 1) * cell,
                1 + (row + 1) * cell,
                fill=COLORS.last_move,
                outline="",
            )
        for i in (1, 2):
            self.create_line(1 + i * cell, 1, 1 + i * cell, 1 + 3 * cell, fill=COLORS.grid, width=2)
            self.create_line(1, 1 + i * cell, 1 + 3 * cell, 1 + i * cell, fill=COLORS.grid, width=2)
        for index in range(BOARD_SIZE):
            row, col = divmod(index, 3)
            center = (1 + (col + 0.5) * cell, 1 + (row + 0.5) * cell)
            piece = board.cells[index]
            if piece is None:
                self.create_text(
                    *center, text=str(index + 1), fill=COLORS.grid, font=(FONT_FAMILY, 11)
                )
            else:
                color = COLORS.max_player if piece is Player.X else COLORS.min_player
                self.create_text(
                    *center, text=piece.value, fill=color, font=(FONT_FAMILY, 24, "bold")
                )


class VisualizerApp:
    """Ventana principal: árbol, panel explicativo y controles de reproducción."""

    SIDEBAR_WIDTH: Final = 360
    DEFAULT_DELAY_MS: Final = 700

    def __init__(self, root: tk.Tk, tree: SearchTree, *, baseline_nodes: int | None) -> None:
        self._root = root
        self._tree = tree
        self._baseline_nodes = baseline_nodes
        self._step = 0
        self._playing = False
        self._after_id: str | None = None
        self._ui_scale = root.winfo_fpixels("1i") / 96  # 1.0 en pantallas al 100 %

        title = "Minimax con poda alfa-beta" if tree.pruning else "Minimax (sin poda)"
        root.title(f"{title} — Gato")
        root.geometry("1500x860")
        root.minsize(1000, 680)
        with suppress(tk.TclError):  # "zoomed" (maximizada) no existe en todos los sistemas
            root.state("zoomed")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        self._canvas = TreeCanvas(root, tree)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        self._build_sidebar(title)
        self._build_controls()
        self._bind_keys()
        self._render()

    # Construcción de la interfaz

    def _build_sidebar(self, title: str) -> None:
        sidebar_width = round(self.SIDEBAR_WIDTH * self._ui_scale)
        sidebar = ttk.Frame(self._root, padding=16, width=sidebar_width)
        sidebar.grid(row=0, column=1, sticky="ns")
        sidebar.pack_propagate(False)
        wrap = sidebar_width - 40

        ttk.Label(sidebar, text=title, font=(FONT_FAMILY, 15, "bold")).pack(anchor="w")
        self._step_label = ttk.Label(sidebar, font=(FONT_FAMILY, 11))
        self._step_label.pack(anchor="w", pady=(2, 10))

        ttk.Label(sidebar, text="Nodo actual", font=(FONT_FAMILY, 10, "bold")).pack(anchor="w")
        self._node_label = ttk.Label(sidebar, font=(FONT_FAMILY, 10), foreground=COLORS.muted)
        self._node_label.pack(anchor="w")
        self._preview = BoardPreview(sidebar, cell=round(40 * self._ui_scale))
        self._preview.pack(anchor="w", pady=(4, 12))

        ttk.Label(sidebar, text="¿Qué está pasando?", font=(FONT_FAMILY, 10, "bold")).pack(
            anchor="w"
        )
        self._narration = tk.Message(
            sidebar,
            width=wrap,
            padx=0,
            font=(FONT_FAMILY, 11),
            anchor="nw",
            background=self._root["background"],
        )
        self._narration.pack(anchor="w", fill="x")
        ttk.Frame(sidebar, height=1).pack(fill="both", expand=True)

        self._stats_label = ttk.Label(
            sidebar, font=(FONT_FAMILY, 10), justify="left", wraplength=wrap
        )
        self._stats_label.pack(anchor="w", pady=(8, 8))
        self._build_legend(sidebar, wrap)

    def _build_legend(self, parent: tk.Misc, wrap: int) -> None:
        legend = ttk.Frame(parent)
        legend.pack(anchor="w", fill="x")
        ttk.Label(legend, text="Leyenda", font=(FONT_FAMILY, 10, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        for row, (kind, style, text) in enumerate(self._legend_items(), start=1):
            swatch = tk.Canvas(
                legend, width=26, height=16, highlightthickness=0, background=COLORS.background
            )
            match kind:
                case "border":
                    swatch.create_rectangle(4, 2, 22, 14, outline=style, width=2.5)
                case "fill":
                    swatch.create_rectangle(4, 2, 22, 14, fill=style, outline="")
                case "dash":
                    swatch.create_rectangle(4, 2, 22, 14, outline=COLORS.pruned, dash=(3, 2))
                    swatch.create_text(13, 8, text="✕", fill=COLORS.min_player)
                case "line":
                    swatch.create_line(2, 8, 24, 8, fill=style, width=4)
                case _:
                    swatch.create_text(13, 8, text=style, fill=COLORS.text, font=(FONT_FAMILY, -12))
            ui = self._ui_scale
            swatch.configure(width=26 * ui, height=16 * ui)
            swatch.scale("all", 0, 0, ui, ui)
            swatch.grid(row=row, column=0, padx=(0, 6), pady=2, sticky="w")
            ttk.Label(
                legend,
                text=text,
                font=(FONT_FAMILY, 9),
                foreground=COLORS.muted,
                wraplength=wrap - round(32 * self._ui_scale),
                justify="left",
            ).grid(row=row, column=1, sticky="w")

    def _legend_items(self) -> list[tuple[str, str, str]]:
        """Elementos de la leyenda como ``(tipo de muestra, color o símbolo, texto)``."""
        items = [
            ("border", COLORS.max_player, "MAX (X): busca el mayor valor"),
            ("border", COLORS.min_player, "MIN (O): busca el menor valor"),
            ("fill", COLORS.last_move, "Jugada que llevó a ese estado"),
            ("fill", COLORS.active, "Nodo que se procesa en este paso"),
            ("text", "v", "Valor del nodo: + favorece a X, - favorece a O"),
        ]
        if self._tree.pruning:
            items += [
                ("text", "≥ ≤", "Cota: no hizo falta el valor exacto"),
                ("text", "α β", "α: lo mejor que MAX ya asegura · β: lo mejor que MIN ya asegura"),
                ("dash", "", "Rama podada (α ≥ β): no se explora"),
            ]
        items.append(("line", COLORS.principal, "Camino óptimo (al terminar)"))
        return items

    def _build_controls(self) -> None:
        bar = ttk.Frame(self._root, padding=(16, 8))
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        buttons = (
            ("⏮ Inicio", lambda: self._go_to(0)),
            ("◀ Anterior", lambda: self._go_to(self._step - 1)),
            ("Siguiente ▶", lambda: self._go_to(self._step + 1)),
            ("Final ⏭", lambda: self._go_to(self._tree.step_count)),
        )
        for text, command in buttons:
            ttk.Button(
                bar, text=text, command=partial(self._manual, command), takefocus=False
            ).pack(side="left", padx=4)
        self._play_button = ttk.Button(
            bar, text="⏵ Reproducir", command=self._toggle_play, takefocus=False
        )
        self._play_button.pack(side="left", padx=(16, 4))

        ttk.Label(bar, text="Velocidad (ms por paso):").pack(side="left", padx=(24, 4))
        self._delay = tk.IntVar(value=self.DEFAULT_DELAY_MS)
        ttk.Scale(bar, from_=100, to=2000, variable=self._delay, length=200, takefocus=False).pack(
            side="left"
        )
        ttk.Label(
            bar, text="Teclas: ← → pasos · Espacio reproducir · Inicio/Fin", foreground=COLORS.muted
        ).pack(side="right")

    def _bind_keys(self) -> None:
        self._root.bind("<Right>", lambda _e: self._manual(lambda: self._go_to(self._step + 1)))
        self._root.bind("<Left>", lambda _e: self._manual(lambda: self._go_to(self._step - 1)))
        self._root.bind("<Home>", lambda _e: self._manual(lambda: self._go_to(0)))
        self._root.bind(
            "<End>", lambda _e: self._manual(lambda: self._go_to(self._tree.step_count))
        )
        self._root.bind("<space>", lambda _e: self._toggle_play())

    # Reproducción

    def _manual(self, action: Callable[[], None]) -> None:
        """Ejecuta una navegación manual, pausando la reproducción automática."""
        self._pause()
        action()

    def _go_to(self, step: int) -> None:
        self._step = max(0, min(step, self._tree.step_count))
        self._render()

    def _toggle_play(self) -> None:
        if self._playing:
            self._pause()
            return
        if self._step >= self._tree.step_count:
            self._go_to(0)
        self._playing = True
        self._play_button.configure(text="⏸ Pausa")
        self._schedule()

    def _pause(self) -> None:
        self._playing = False
        self._play_button.configure(text="⏵ Reproducir")
        if self._after_id is not None:
            self._root.after_cancel(self._after_id)
            self._after_id = None

    def _schedule(self) -> None:
        self._after_id = self._root.after(int(self._delay.get()), self._tick)

    def _tick(self) -> None:
        self._after_id = None
        self._go_to(self._step + 1)
        if self._step >= self._tree.step_count:
            self._pause()
        elif self._playing:
            self._schedule()

    # Render

    def _render(self) -> None:
        snapshot = self._tree.snapshot(self._step)
        self._canvas.show(snapshot)
        self._step_label.configure(text=f"Paso {snapshot.step} de {self._tree.step_count}")
        self._narration.configure(text=describe(snapshot.event, self._tree))

        node = self._tree.nodes[snapshot.active_id if snapshot.active_id is not None else ROOT_ID]
        self._preview.show(node.board, node.move)
        situation = "estado terminal" if node.board.is_terminal else f"le toca a {role(node.board)}"
        self._node_label.configure(text=f"Profundidad {node.depth} · {situation}")
        self._stats_label.configure(text=self._stats_text(snapshot))

    def _stats_text(self, snapshot: Snapshot) -> str:
        total = self._tree.result.nodes_visited
        lines = [f"Nodos visitados: {snapshot.nodes_visited} de {total}"]
        if self._tree.pruning:
            lines.append(f"Podas realizadas: {snapshot.cutoffs} de {self._tree.result.cutoffs}")
            if self._baseline_nodes is not None:
                saved = 1 - total / self._baseline_nodes
                lines.append(
                    f"Minimax sin poda visitaría {self._baseline_nodes} nodos "
                    f"(la poda ahorra {saved:.0%})"
                )
        return "\n".join(lines)


def run(board: Board, *, pruning: bool) -> None:
    """Ejecuta la búsqueda sobre ``board`` y abre el visualizador."""
    search = alphabeta if pruning else minimax
    result = search(board, record=True)
    baseline = minimax(board).nodes_visited if pruning else None
    _enable_high_dpi()
    root = tk.Tk()
    VisualizerApp(root, SearchTree(result, pruning=pruning), baseline_nodes=baseline)
    root.mainloop()


def _enable_high_dpi() -> None:
    """En Windows, dibuja a la resolución real de la pantalla en lugar de escalar
    la ventana como imagen, lo que deja el texto borroso."""
    if sys.platform == "win32":
        import ctypes

        with suppress(AttributeError, OSError):
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
