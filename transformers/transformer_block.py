"""
============================================================
SECTION 1: QUESTION (Minimal Prompt + Skeleton)
============================================================

Transformer Block (Reconstructed Practice Problem)

Implement a single Transformer block (decoder-style), using:
- Pre-LayerNorm
- Multi-Headed Self-Attention (causal)
- Feedforward network (MLP)
- Residual connections

Given input `embedded` of shape (B, T, model_dim), compute:

1) x = x + MultiHeadAttention(LayerNorm(x))
2) x = x + FeedForward(LayerNorm(x))

Return the resulting tensor (same shape).

Implement:
- TransformerBlock.__init__(model_dim, num_heads)
- TransformerBlock.forward(embedded)

(Stop here if you want to solve without hints.)
"""

import torch
import torch.nn as nn
from torchtyping import TensorType


class TransformerBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
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
- MultiHeadAttention: num_heads heads, each head_size = model_dim // num_heads
- FeedForward (VanillaNeuralNetwork):
    up projection:   Linear(model_dim, model_dim*4)
    activation:      ReLU
    down projection: Linear(model_dim*4, model_dim)
    dropout:         Dropout(p=0.2)

- Layer norms:
    ln1 = LayerNorm(model_dim)
    ln2 = LayerNorm(model_dim)

- Forward uses residual + pre-norm:
    x = x + attn(ln1(x))
    x = x + ff(ln2(x))

Some graders may require rounding output to fixed decimals.
"""