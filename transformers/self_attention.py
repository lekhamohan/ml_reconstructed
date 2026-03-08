"""
============================================================
SECTION 1: QUESTION (Minimal Prompt + Skeleton)
============================================================

Self Attention (Single-Head, Causal) (Reconstructed Practice Problem)

Implement a single-head self-attention module (decoder-style / causal attention).

Given an input tensor `embedded` of shape (B, T, C):
- Project it into Q, K, V using linear layers (no bias)
- Compute scaled dot-product attention scores
- Apply a causal mask so position t cannot attend to positions > t
- Apply softmax to obtain attention weights
- Multiply weights by V to produce output of shape (B, T, attention_dim)

Implement:
- __init__(embedding_dim, attention_dim)
- forward(embedded)

(Stop here if you want to solve without hints.)
"""

import torch
import torch.nn as nn
from torchtyping import TensorType


class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        raise NotImplementedError("Implement __init__()")

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        raise NotImplementedError("Implement forward()")


"""
============================================================
SECTION 2: OPTIONAL HELPERS / HINTS (Read only if needed)
============================================================

⚠️ Optional Help Section

Reconstructed details from expected structure:
- Use three linear layers (bias=False):
    key_layer:   (embedding_dim -> attention_dim)
    query_layer: (embedding_dim -> attention_dim)
    value_layer: (embedding_dim -> attention_dim)

- For embedded shape (B, T, C):
    K, Q, V shapes: (B, T, attention_dim)

- Attention scores:
    scores = (Q @ K^T) / sqrt(attention_dim)
  where K^T transposes the last two dims -> (B, attention_dim, T)

- Causal mask:
    tril = torch.tril(torch.ones(T, T))
    scores = scores.masked_fill(tril == 0, -inf)

- weights = softmax(scores, dim=-1)
- output = weights @ V

Some graders may require rounding output to a fixed decimal places.
"""