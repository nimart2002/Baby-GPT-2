"""GPT: el modelo definitivo del proyecto.

Es la misma arquitectura que hemos construido en los pasos 4-7 (model.py),
reorganizada como GPT-2:
  - toda la configuración en un solo objeto (GPTConfig)
  - atención multicabeza calculada de una vez, sin bucle sobre las cabezas
  - weight tying, inicialización de GPT-2 y dropout
  - un solo generate(), con temperatura
  - guardar y cargar modelos entrenados
"""
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128       # longitud máxima de contexto
    n_layer: int = 2            # número de bloques
    n_head: int = 2             # cabezas por bloque
    n_embd: int = 64            # dimensión del flujo residual
    ff_dim: int = 256           # dimensión interna del MLP
    dropout: float = 0.0        # probabilidad de dropout (0 = desactivado)
    bias: bool = True           # sesgos en las capas lineales y LayerNorm


class CausalSelfAttention(nn.Module):
    """Atención multicabeza causal, todas las cabezas a la vez."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_size = cfg.n_embd // cfg.n_head
        # Una sola capa calcula Q, K y V de todas las cabezas: n_embd -> 3·n_embd
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))
        self.last_weights = None     # aquí guardamos los pesos de atención para analizarlos

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)               # cada uno (B, T, C)
        # Separar las cabezas: (B, T, C) -> (B, T, n_head, head_size) -> (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)    # (B, n_head, T, T)
        scores = scores.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        self.last_weights = weights.detach()
        y = self.attn_drop(weights) @ v                     # (B, n_head, T, head_size)

        # Volver a juntar las cabezas: (B, n_head, T, head_size) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, cfg.ff_dim, bias=cfg.bias)
        self.proj = nn.Linear(cfg.ff_dim, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """x = x + Atención(LN(x));  x = x + MLP(LN(x))"""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)     # embedding de tokens
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)     # embedding de posiciones
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight                   # weight tying

        self.apply(self._init_weights)
        # Inicialización de GPT-2: las capas que escriben en el flujo residual empiezan
        # más pequeñas, para que la suma de muchos bloques no crezca demasiado.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, return_hidden=False):
        """idx: (B, T).  targets: (B, T) o None; las posiciones con -100 no cuentan en la pérdida.

        Si return_hidden=True, devuelve también el flujo residual tras cada bloque
        (lista de n_layer + 1 tensores (B, T, n_embd)), para analizar representaciones.
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"secuencia de {T} tokens > block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        hidden = [x]
        for block in self.blocks:
            x = block(x)
            hidden.append(x)
        logits = self.lm_head(self.ln_f(x))                     # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-100)
        if return_hidden:
            return logits, loss, hidden
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, greedy=False):
        """Alarga idx (B, T) con max_new_tokens tokens.
        greedy=True elige siempre el más probable; si no, se sortea con la temperatura dada."""
        was_training = self.training
        self.eval()                                             # desactiva el dropout
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if greedy:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        self.train(was_training)
        return idx

    def save(self, path):
        torch.save({"config": asdict(self.cfg), "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        checkpoint = torch.load(path, map_location=map_location)
        model = cls(GPTConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["state_dict"])
        return model