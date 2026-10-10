"""Paso 6: ¿ayudan varias cabezas? Mismo tamaño total (n_embd), distinto número de cabezas."""
import torch

from minigpt.model import MultiHeadModel
from minigpt.tokenizer import BOS, build_grid_tokenizer
from minigpt.train import train

BLOCK_SIZE = 32
STEPS = 2000

tok = build_grid_tokenizer()
results = {}
for n_head in [1, 2, 4]:
    print(f"\n===== {n_head} cabeza(s) =====")
    torch.manual_seed(0)
    model = MultiHeadModel(tok.vocab_size, BLOCK_SIZE, n_embd=64, n_head=n_head)
    print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    losses = train(model, tok, steps=STEPS, batch_size=32, block_size=BLOCK_SIZE,
                   lr=3e-3, log_every=500)
    results[n_head] = sum(losses[-200:]) / 200          # media de los últimos 200 pasos

    start = torch.tensor([[tok.stoi[BOS]]])
    out = model.generate(start, max_new_tokens=20)
    print("Generado:", " ".join(tok.decode(out[0].tolist())))

print("\nPérdida final (media de los últimos 200 pasos):")
for n_head, loss in results.items():
    print(f"  {n_head} cabeza(s): {loss:.3f}")
print("  (bigrama: 4.04)")