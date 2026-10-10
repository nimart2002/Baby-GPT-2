"""Lotes de entrenamiento: convierte recorridos en tensores de PyTorch."""
import random

import torch

from minigpt.data import make_grid, random_walk
from minigpt.tokenizer import BOS

def make_sequence(tokenizer, block_size, rng):
    """Una secuencia de block_size + 1 números: <bos> seguido de un recorrido."""
    n_steps = block_size // 2 
    grid = make_grid(rng)
    words, _ = random_walk(grid, n_steps,rng)
    ids = tokenizer.encode([BOS] + words)
    return ids[: block_size + 1]

def get_batch(tokenizer, batch_size, block_size, rng):
    """Devuelve dos tensores (batch_size, block_size):
         x: la entrada
         y: lo que hay que predecir en cada posición (x desplazado una posición)
    """
    xs, ys = [], []
    for _ in range(batch_size):
        ids = make_sequence(tokenizer, block_size, rng)
        xs.append(ids[:-1])
        ys.append(ids[1:])
    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    return x, y


if __name__ == "__main__":
    from minigpt.tokenizer import build_grid_tokenizer

    tok = build_grid_tokenizer()
    rng = random.Random(0)
    x, y = get_batch(tok, batch_size=4, block_size=16, rng=rng)

    print("x:", x.shape, x.dtype)
    print("y:", y.shape, y.dtype)
    print("\nPrimera fila de x:", x[0].tolist())
    print("Primera fila de y:", y[0].tolist())

    print("\nQué aprende el modelo en cada posición de la primera secuencia:")
    for t in range(8):
        contexto = " ".join(tok.decode(x[0, : t + 1].tolist()))
        objetivo = tok.decode([y[0, t].item()])[0]
        print(f"  t={t}:  {contexto:45s} →  {objetivo}")
