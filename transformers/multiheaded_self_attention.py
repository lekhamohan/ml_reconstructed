"""
============================================================
SECTION 1: QUESTION (Minimal Prompt + Skeleton)
============================================================

Multi-Headed Self Attention (Reconstructed Practice Problem)

Implement a multi-headed causal self-attention module.

Given `embedded` of shape (B, T, embedding_dim):
- Create `num_heads` independent SingleHeadAttention modules
- Each head uses attention_dim_per_head = attention_dim // num_heads
- Concatenate head outputs along the last dimension
- Return output of shape (B, T, attention_dim)

Implement:
- __init__(embedding_dim, attention_dim, num_heads)
- forward(embedded)

(Stop here if you want to solve without hints.)
"""

import torch
import torch.nn as nn
from torchtyping import TensorType


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
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
- head_size = attention_dim // num_heads
- Use nn.ModuleList to store heads.
- Each head is a SingleHeadAttention(embedding_dim, head_size).
- Forward:
    outputs = [head(embedded) for head in heads]
    out = torch.cat(outputs, dim=-1)

Some graders may require deterministic initialization via torch.manual_seed(0)
and rounding outputs to fixed decimals.
"""