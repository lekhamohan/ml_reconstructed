"""
============================================================
SECTION 1: QUESTION (Minimal Prompt + Skeleton)
============================================================

GPT Model (Reconstructed Practice Problem)

Implement a small GPT-like decoder-only Transformer model.

Model components:
- token embedding table: Embedding(vocab_size, model_dim)
- positional embedding table: Embedding(context_length, model_dim)
- stack of `num_blocks` TransformerBlocks
- final LayerNorm
- language modeling head: Linear(model_dim, vocab_size)

Forward input:
- context: (B, T) integer token IDs (T <= context_length)

Forward output:
- probabilities (or logits) over vocabulary for each position:
  shape (B, T, vocab_size)

Implement:
- __init__(vocab_size, context_length, model_dim, num_blocks, num_heads)
- forward(context)

(Stop here if you want to solve without hints.)
"""

import torch
import torch.nn as nn
from torchtyping import TensorType


class GPT(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_blocks: int, num_heads: int):
        super().__init__()
        raise NotImplementedError("Implement __init__()")

    def forward(self, context: TensorType[int]) -> TensorType[float]:
        raise NotImplementedError("Implement forward()")


"""
============================================================
SECTION 2: OPTIONAL HELPERS / HINTS (Read only if needed)
============================================================

⚠️ Optional Help Section

Reconstructed details from expected structure:
- token_embeddings = token_embedding_table(context)              # (B, T, C)
- position_embeddings = position_embedding_table(arange(T))      # (T, C) or broadcastable to (B, T, C)
- x = token_embeddings + position_embeddings
- x = transformer_blocks(x)   # preserves (B, T, C)
- x = layernorm_final(x)
- output = langmod_head(x)    # (B, T, vocab_size)

Some graders may apply softmax over last dim and round decimals.

Common gotcha:
- positional embedding indexing uses sequence positions 0..T-1.
"""