# Tarea-2-3-AI
Tarea 2 y 3 de Inteligencia Artificial - Alan Emiliano Palomo Plata - 2048459

**Opción B — Minimax con poda alfa-beta** aplicado al gato (tic-tac-toe), con un visualizador
que reproduce paso a paso el recorrido del árbol de juego: qué nodo se visita, qué valor sube
de cada hijo a su padre, cómo cambian α y β, y qué ramas se podan y por qué.

📹 **Video de la demo:** https://drive.google.com/file/d/1wozfusKoZQmJGX1TtKfFDLhjFiPCpr5e/view?usp=sharing

## Requisitos

- Python 3.12 o superior (incluye `tkinter`; no hay dependencias externas para ejecutar).

## Instalación y uso

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"

gato-minimax                 # alfa-beta sobre la posición de la demo
gato-minimax --sin-poda      # minimax puro, para comparar
gato-minimax -t "XO./X.X/O.."   # cualquier posición con a lo más 5 casillas vacías
```

La posición se escribe por filas: `X`, `O` y `.` para vacía; `/` separa filas.

**Controles:** `Siguiente ▶` / `◀ Anterior` (o flechas ← →) avanzan un paso; `Reproducir`
(o Espacio) anima la búsqueda; `Inicio` / `Final` saltan al principio o al resultado.

### Pruebas

```bash
pytest
```

## Estructura

```
src/gato_minimax/
├── game.py        # Estado del juego, acciones, estados terminales y utilidad
├── search.py      # Minimax y alfa-beta (una sola implementación) + registro de eventos
├── tree.py        # Reconstruye el árbol a partir de los eventos y calcula su disposición
├── narration.py   # Explicación en español de cada paso
├── app.py         # Visualizador (tkinter)
└── __main__.py    # Línea de comandos
tests/
└── test_minimax.py
```

La búsqueda no sabe nada de la interfaz: solo emite eventos (`NodeVisited`,
`ValuePropagated`, `BranchesPruned`, `NodeResolved`). El visualizador los reproduce, así
que lo que se ve en pantalla es exactamente lo que hizo el algoritmo.

## Justificación del diseño

### Juego

El gato es un juego de dos jugadores, de suma cero y con información perfecta. Su árbol
es lo bastante pequeño para explorarse **completo** y, partiendo de una posición a media
partida, cabe en pantalla. Eso permite mostrar todos los valores y todas las podas sin
recortar nada.

### Representación de estados y acciones

- **Estado:** tupla inmutable de 9 casillas (índices 0 a 8 por filas) con `X`, `O` o vacía.
  El turno no se guarda: se deduce del número de fichas, porque X siempre inicia. Al ser
  inmutable, cada nodo del árbol es un estado independiente y no hay que "deshacer" jugadas.
- **Acción:** índice de una casilla vacía. Se generan en orden ascendente, así que la
  búsqueda es determinista y reproducible. En pantalla las casillas se numeran del 1 al 9.

### Función de utilidad

Como el árbol se explora hasta los estados terminales, **no se necesita una función de
evaluación heurística**: desde el tablero vacío, minimax visita 549,946 nodos y alfa-beta
solo 20,866, lo que toma segundos. En los estados terminales, desde la perspectiva de
X (MAX):

| Resultado | Utilidad |
|-----------|----------|
| Gana X    | `+(casillas vacías + 1)` |
| Empate    | `0` |
| Gana O    | `-(casillas vacías + 1)` |

Sumar las casillas vacías conserva el orden victoria > empate > derrota, pero además
hace que se prefiera **ganar lo antes posible y perder lo más tarde posible**. Con un
simple ±1, una victoria inmediata y una tardía valdrían lo mismo.

### Minimax y poda alfa-beta

- MAX (X) elige el hijo de mayor valor y MIN (O) el de menor valor.
- **α** es lo mejor que MAX ya tiene asegurado en el camino actual y **β** lo mejor que
  MIN ya tiene asegurado. Cuando en un nodo se cumple `α ≥ β`, el jugador de arriba nunca
  dejaría que la partida llegue ahí, así que las jugadas restantes se **podan**.
- Ambos algoritmos usan la misma función recursiva. En minimax la ventana se queda en
  `(-∞, +∞)` y nunca hay corte, por lo que la poda es la única diferencia.
- Se usa la variante *fail-soft*: si el valor de un nodo queda fuera de la ventana con la
  que se entró, es una **cota** y se muestra como `v ≥ …` o `v ≤ …`. Ese valor exacto no se
  necesita, porque ese camino no se elegirá. La raíz siempre obtiene su valor exacto.

### Posición de la demo

```
X O .
X . X      Le toca a O (MIN).
O . .
```

X amenaza con ganar en la casilla 5. Minimax encuentra que O debe bloquear ahí (valor −2:
después O gana). Cualquier otra jugada vale +3, porque X gana de inmediato. La posición
produce 6 podas y mantiene el árbol legible.

| Búsqueda desde... | Minimax | Alfa-beta | Ahorro |
|---|---:|---:|---:|
| La posición de la demo | 43 nodos | 25 nodos (6 podas) | 42 % |
| El tablero vacío | 549,946 nodos | 20,866 nodos | 96 % |

Ambos algoritmos devuelven siempre el mismo valor y la misma jugada; las pruebas lo
verifican en todas las posiciones con 3, 4 y 5 casillas vacías.
