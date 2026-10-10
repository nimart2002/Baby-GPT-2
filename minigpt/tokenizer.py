"""Tokenizador: convierte palabras en números enteros y viceversa."""

from minigpt.data import DIRECTIONS, NAMES

class Tokenizer:
    def __init__(self, vocab):
        """vocab: lista de tokens. La posición de cada token en la lista es su número."""
        self.vocab = list(vocab)
        self.stoi = {token: i for i, token in enumerate(self.vocab)}
        self.itos = {i: token for i, token in enumerate(self.vocab)}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, words):
        """Lista de palabras -> lista de enteros."""
        return [self.stoi[w] for w in words]
    
    def decode(self, ids):
        return [self.itos[i] for i in ids]

    
BOS = '<bos>'  # token especial: "aquí empieza una secuencia"

def build_grid_tokenizer():
    """Tokenizador para la tarea de las cuadrículas."""
    return Tokenizer([BOS] + DIRECTIONS + NAMES)

if __name__ == "__main__":
    import random
    from minigpt.data import make_grid, random_walk

    tok = build_grid_tokenizer()
    print("Tamaño del vocabulario:", tok.vocab_size)
    print("Primeros tokens:", tok.vocab[:8])

    rng = random.Random(0)
    grid = make_grid(rng)
    words, _ = random_walk(grid, n_steps=6, rng=rng)
    ids = tok.encode([BOS] + words)
    print("\nPalabras:", [BOS] + words)
    print("Números: ", ids)
    print("De vuelta:", tok.decode(ids))