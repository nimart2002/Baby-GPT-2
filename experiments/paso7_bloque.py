"""Paso 7: ¿qué aporta cada pieza del bloque? Se añaden una a una."""
import torch

from minigpt.model import BlockModel
from minigpt.tokenizer import build_grid_tokenizer
from minigpt.train import train

BLOCK_SIZE = 32
STEPS = 2000

CONFIGS = [
    # (nombre, n_layer, use_mlp, use_residual, use_layernorm)
    ("1 bloque: solo atención",               1, False, False, False),
    ("1 bloque: + MLP",                       1, True,  False, False),
    ("1 bloque: + MLP + residual",            1, True,  True,  False),
    ("1 bloque: + MLP + residual + LayerNorm", 1, True,  True,  True),
    ("4 bloques: sin residual ni LayerNorm",  4, True,  False, False),
    ("4 bloques: completos",                  4, True,  True,  True),
]

tok = build_grid_tokenizer()
results = []
for name, n_layer, use_mlp, use_res, use_ln in CONFIGS:
    print(f"\n===== {name} =====")
    torch.manual_seed(0)
    model = BlockModel(tok.vocab_size, BLOCK_SIZE, n_embd=64, n_head=4, n_layer=n_layer,
                       use_mlp=use_mlp, use_residual=use_res, use_layernorm=use_ln)
    n_params = sum(p.numel() for p in model.parameters())
    losses = train(model, tok, steps=STEPS, batch_size=32, block_size=BLOCK_SIZE,
                   lr=3e-3, log_every=1000)
    results.append((name, n_params, sum(losses[-200:]) / 200))

print("\nResumen (pérdida media de los últimos 200 pasos):")
for name, n_params, loss in results:
    print(f"  {name:42s} {n_params:>8,} parámetros   pérdida {loss:.3f}")
print("  (bigrama: 4.04; 4 cabezas del paso 6: 2.71)")