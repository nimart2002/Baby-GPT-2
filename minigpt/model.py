import torch
import torch.nn as nn
import torch.nn.functional as F

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