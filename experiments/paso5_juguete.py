"""Paso 5: la idea de la atención con tensores pequeños que se pueden mirar a mano."""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
T, C = 4, 2                    # 4 posiciones, cada una un vector de 2 números
x = torch.randn(T, C)
print("x (una fila por posición):\n", x)

# 1) Promedio de las posiciones anteriores, con un bucle
avg_loop = torch.zeros(T, C)
for t in range(T):
    avg_loop[t] = x[: t + 1].mean(dim=0)       # media de las filas 0..t
print("\n1) Promedio del pasado con un bucle:\n", avg_loop)

# 2) Lo mismo con una multiplicación de matrices
w = torch.tril(torch.ones(T, T))               # matriz triangular inferior de unos
w = w / w.sum(dim=1, keepdim=True)             # cada fila suma 1
print("\n2) Matriz de pesos:\n", w)
print("   w @ x:\n", w @ x)
print("   ¿Coincide con el bucle?", torch.allclose(w @ x, avg_loop))

# 3) Lo mismo con softmax: -inf en el futuro, 0 en el pasado
scores = torch.zeros(T, T)
scores = scores.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
w3 = F.softmax(scores, dim=1)
print("\n3) Pesos con softmax:\n", w3)
print("   ¿Iguales a los de 2)?", torch.allclose(w3, w))

# 4) Atención de verdad: los pesos dependen del contenido (consulta · clave)
head_size = 2
W_q = torch.randn(C, head_size)
W_k = torch.randn(C, head_size)
W_v = torch.randn(C, head_size)
q, k, v = x @ W_q, x @ W_k, x @ W_v
scores = q @ k.T / head_size ** 0.5            # (T, T): parecido entre cada consulta y cada clave
scores = scores.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
w4 = F.softmax(scores, dim=1)
print("\n4) Pesos de atención (dependen de x):\n", w4)
print("   Suma de cada fila:", w4.sum(dim=1))
print("   Salida = w4 @ v:\n", w4 @ v)