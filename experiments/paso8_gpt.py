"""Paso 8: entrenar el GPT definitivo, guardarlo, cargarlo y generar."""
import math
import random

import torch

from minigpt.batches import get_batch
from minigpt.gpt import GPT, GPTConfig
from minigpt.tokenizer import BOS, build_grid_tokenizer
from minigpt.train import train

tok = build_grid_tokenizer()
cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=32,
                n_layer=2, n_head=2, n_embd=64, ff_dim=256)   # configuración del documento del TFM
torch.manual_seed(0)
model = GPT(cfg)
print(f"Parámetros: {model.num_params():,}")

x, y = get_batch(tok, batch_size=32, block_size=cfg.block_size, rng=random.Random(123))
_, loss = model(x, y)
print(f"Pérdida inicial: {loss.item():.3f}   (ln V = {math.log(tok.vocab_size):.3f})\n")

losses = train(model, tok, steps=8000, batch_size=32, block_size=cfg.block_size,
               lr=3e-3, log_every=1000)
print(f"\nPérdida final (media de los últimos 200 pasos): {sum(losses[-200:]) / 200:.3f}")

model.save("checkpoints/gpt_grid.pt")
loaded = GPT.load("checkpoints/gpt_grid.pt")
print("Modelo guardado y cargado de nuevo.")

start = torch.tensor([[tok.stoi[BOS]]])
torch.manual_seed(1)
for temperature in [0.5, 1.0, 1.5]:
    out = loaded.generate(start, max_new_tokens=20, temperature=temperature)
    print(f"\nTemperatura {temperature}:")
    print(" ".join(tok.decode(out[0].tolist())))