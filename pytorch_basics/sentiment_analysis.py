"""
============================================================
SECTION 1: QUESTION (Minimal Prompt + Skeleton)
============================================================

Sentiment Analysis (Reconstructed Practice Problem)

Implement a simple PyTorch model for sentiment analysis.

The model should:
- take tokenized input sequences (integer token IDs)
- embed tokens into dense vectors
- average embeddings across the sequence dimension
- produce a single output score/probability per example

Implement:
- __init__(vocabulary_size)
- forward(x)

Notes:
- Assume valid inputs.
- `x` contains token IDs.
- Return a tensor of shape (N, 1), where N is batch size.

(Stop here if you want to solve without hints.)
"""

import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocabulary_size, 16)
        self.linear_layer = nn.Linear(16, 1)
        self.activation = nn.Sigmoid()
    
    def forward(self, x: TensorType[int]) -> TensorType[float]:
        embedding = self.embedding_layer(x)
        pooled = embedding.mean(dim=1)
        output = self.linear_layer(pooled)
        return self.activation(output)


"""
============================================================
SECTION 2: OPTIONAL HELPERS / HINTS (Read only if needed)
============================================================

⚠️  Optional Help Section
Read this only if you want hints, clarifications, or helper APIs.

Reconstructed details (may not match original wording exactly):
- The model uses an embedding layer to map token IDs to vectors.
- Embedding dimension is 16.
- After embedding, average across the sequence length dimension (dim=1).
- Then apply a linear layer to map from 16 -> 1.
- Then apply Sigmoid activation.

Suggested architecture:
1) Embedding(vocabulary_size, 16)
2) Mean over sequence dimension (dim=1)
3) Linear(16 -> 1)
4) Sigmoid

Helpful PyTorch classes/functions:
- nn.Module
- nn.Embedding
- nn.Linear
- nn.Sigmoid
- torch.mean

Shape intuition:
- Input x:                (N, sequence_length)         [integer token IDs]
- After embedding:        (N, sequence_length, 16)
- After mean over dim=1:  (N, 16)
- After linear + sigmoid: (N, 1)

Testing tips:
- Instantiate model with a vocabulary size (e.g. 100)
- Pass integer tensor input with shape (batch_size, seq_len)
- Verify output shape is (batch_size, 1)
- Sigmoid outputs should be in [0, 1]
"""
