"""Bucle de entrenamiento."""
import math
import random

import torch

from batches import get_batch


def train(model, tokenizer, steps, batch_size, block_size, lr, seed=0, log_every=100):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    losses = []
    for step in range(1, steps + 1):
        x, y = get_batch(tokenizer, batch_size, block_size, rng)   # 1. datos

        logits, loss = model(x, y)                                  # 2. predicción y error

        optimizer.zero_grad()                                       # 3. borrar gradientes viejos
        loss.backward()                                             # 4. calcular gradientes
        optimizer.step()                                            # 5. ajustar los pesos

        losses.append(loss.item())
        if step == 1 or step % log_every == 0:
            recent = losses[-log_every:]
            print(f"paso {step:5d}   pérdida {sum(recent) / len(recent):.3f}")
    return losses


if __name__ == "__main__":
    from model import BigramModel
    from tokenizer import BOS, build_grid_tokenizer

    tok = build_grid_tokenizer()
    model = BigramModel(tok.vocab_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros: {n_params:,}")

    # Comprobación: antes de entrenar, la pérdida debería rondar ln(V)
    x, y = get_batch(tok, batch_size=32, block_size=32, rng=random.Random(123))
    _, loss = model(x, y)
    print(f"Pérdida inicial: {loss.item():.3f}   (ln V = {math.log(tok.vocab_size):.3f})\n")

    train(model, tok, steps=1000, batch_size=32, block_size=32, lr=1e-1)

    # Generar un recorrido a partir de <bos>
    start = torch.tensor([[tok.stoi[BOS]]])
    out = model.generate(start, max_new_tokens=20)
    print("\nRecorrido generado:")
    print(" ".join(tok.decode(out[0].tolist())))