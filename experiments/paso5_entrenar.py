"""Paso 5: entrenar el modelo con una cabeza de atención y compararlo con el bigrama."""
import math
import random

import torch

from minigpt.batches import get_batch
from minigpt.model import OneHeadModel
from minigpt.tokenizer import BOS, build_grid_tokenizer
from minigpt.train import train

BLOCK_SIZE = 32

tok = build_grid_tokenizer()
model = OneHeadModel(tok.vocab_size, BLOCK_SIZE, n_embd=64)
print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}")

x, y = get_batch(tok, batch_size=32, block_size=BLOCK_SIZE, rng=random.Random(123))
_, loss = model(x, y)
print(f"Pérdida inicial: {loss.item():.3f}   (ln V = {math.log(tok.vocab_size):.3f})")
print("Límite del bigrama: 3.95\n")

train(model, tok, steps=2000, batch_size=32, block_size=BLOCK_SIZE, lr=3e-3)

start = torch.tensor([[tok.stoi[BOS]]])
out = model.generate(start, max_new_tokens=20)
print("\nRecorrido generado:")
print(" ".join(tok.decode(out[0].tolist())))
