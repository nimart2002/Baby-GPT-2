import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BigramModel(nn.Module):
    """El modelo más simple: predice el siguiente token mirando SOLO el token actual.

    Es una tabla de V x V números: la fila i contiene las puntuaciones (logits)
    de cada posible token siguiente cuando el token actual es i.
    """
    def __init__(self,vocab_size):
        super().__init__()  # El objeto inicializado es a la vez objeto de una clase superior (nn.Module)
        self.table = nn.Embedding(vocab_size,vocab_size)

    def forward(self, idx, targets=None):
        """idx: (B, T) tokens de entrada.  targets: (B, T) tokens a predecir, o None."""
        logits = self.table(idx)                          # (B, T, V)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Alarga la secuencia idx (B, T) con max_new_tokens tokens nuevos."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx)                        # (B, T, V)
            logits = logits[:, -1, :]                    # solo la última posición: (B, V)
            probs = F.softmax(logits, dim=-1)            # logits -> probabilidades
            next_id = torch.multinomial(probs, num_samples=1)   # sortear el siguiente: (B, 1)
            idx = torch.cat([idx, next_id], dim=1)       # añadirlo: (B, T + 1)
        return idx

class Head(nn.Module):
    """Una cabeza de atención causal.

    Cada posición t calcula una media ponderada de los "valores" de las posiciones 0..t.
    El peso de cada posición s depende de cuánto se parece la consulta de t a la clave de s.
    """

    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)   # "¿qué busco?"
        self.key = nn.Linear(n_embd, head_size, bias=False)     # "¿qué contengo?"
        self.value = nn.Linear(n_embd, head_size, bias=False)   # "¿qué ofrezco si me eligen?"
        # Máscara triangular: no es un parámetro (no se entrena), por eso es un "buffer"
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)                                   # (B, T, head_size)
        k = self.key(x)                                     # (B, T, head_size)
        v = self.value(x)                                   # (B, T, head_size)

        scores = q @ k.transpose(-2, -1)                    # (B, T, T): consulta_t · clave_s
        scores = scores / math.sqrt(k.shape[-1])            # escalar para que softmax no se sature
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))   # no mirar el futuro
        weights = F.softmax(scores, dim=-1)                 # cada fila suma 1

        return weights @ v                                  # (B, T, head_size)


class OneHeadModel(nn.Module):
    """Embeddings de token y de posición + una cabeza de atención + capa de salida."""

    def __init__(self, vocab_size, block_size, n_embd=64):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)     # qué token es
        self.pos_emb = nn.Embedding(block_size, n_embd)     # en qué posición está
        self.head = Head(n_embd, n_embd, block_size)
        self.lm_head = nn.Linear(n_embd, vocab_size)        # de vuelta a puntuaciones por token

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)            # 0, 1, ..., T-1
        x = self.tok_emb(idx) + self.pos_emb(pos)           # (B, T, n_embd)
        x = self.head(x)                                    # (B, T, n_embd)
        logits = self.lm_head(x)                            # (B, T, V)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]            # el contexto no puede pasar de block_size
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx