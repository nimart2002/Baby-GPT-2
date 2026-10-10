import random

from data import make_grid, random_walk
from tokenizer import build_grid_tokenizer


def test_vocab_size():
    tok = build_grid_tokenizer()
    assert tok.vocab_size == 1 + 4 + 676


def test_roundtrip():
    """Codificar y decodificar debe devolver exactamente las palabras originales."""
    tok = build_grid_tokenizer()
    rng = random.Random(1)
    for _ in range(100):
        grid = make_grid(rng)
        words, _ = random_walk(grid, n_steps=20, rng=rng)
        assert tok.decode(tok.encode(words)) == words