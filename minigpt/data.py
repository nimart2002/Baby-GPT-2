"""Generación de datos: recorridos aleatorios por cuadrículas 3x3."""
import random
import string

# Creamos todos los nombres posibles de dos letras: 'aa', 'ab',....

NAMES = [a + b for a in string.ascii_lowercase for b in string.ascii_lowercase]

#Creamos las cuatro direcciones y como cambian las posiciones de la cuadrilla

DIRECTIONS = ['NORTH', 'SOUTH', 'EAST', 'WEST']

MOVES = {
    'NORTH': (-1,0), # Una fila hacia arriba
    'SOUTH': (1,0), # Una fila hacia abajo
    'EAST': (0,1),
    'WEST': (0,-1),

}

GRID_SIZE = 3

def make_grid(rng):
    """Crea una cuadrícula nueva: asigna un nombre al azar a cada casilla."""
    names = rng.sample(NAMES,GRID_SIZE * GRID_SIZE)
    grid = {}
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            grid[(row,col)] = names[row*GRID_SIZE + col]

    return grid


def valid_moves(pos):
    """Direcciones que no se salen de la cuadrícula desde la posición pos."""
    row,col = pos
    moves = []
    for direction in DIRECTIONS:
        d_row, d_col = MOVES[direction]
        new_row, new_col = row + d_row, col + d_col
        if 0<=new_row < GRID_SIZE and 0 <= new_col < GRID_SIZE:
            moves.append(direction)
    
    return moves

def random_walk(grid, n_steps, rng):
    """Paseo aleatorio de n_steps pasos.

    Devuelve:
      words: lista de palabras, p. ej. ["ab", "EAST", "cd", "SOUTH", "ef"]
      path:  lista de posiciones visitadas, p. ej. [(0, 0), (0, 1), (1, 1)]
    """
    pos = (rng.randrange(GRID_SIZE), rng.randrange(GRID_SIZE))
    words = [grid[pos]]
    path = [pos]
    for _ in range(n_steps):
        direction = rng.choice(valid_moves(pos))
        d_row, d_col = MOVES[direction]
        pos = (pos[0] + d_row, pos[1] + d_col)
        words.append(direction)
        words.append(grid[pos])
        path.append(pos)
    return words, path

def print_grid(grid):
    """Dibuja la cuadrícula en la terminal."""
    for row in range(GRID_SIZE):
        print("  ".join(grid[(row, col)] for col in range(GRID_SIZE)))

if __name__ == "__main__":
    rng = random.Random(0)          # semilla fija: siempre el mismo resultado
    grid = make_grid(rng)
    print("Cuadrícula:")
    print_grid(grid)
    words, path = random_walk(grid, n_steps=6, rng=rng)
    print("\nRecorrido:")
    print(" ".join(words))
    print("\nPosiciones:", path)


