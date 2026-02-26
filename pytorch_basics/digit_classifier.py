"""
============================================================
SECTION 1: QUESTION (Minimal Prompt + Skeleton)
============================================================

Digit Classifier (Reconstructed Practice Problem)

Implement a simple feedforward neural network for digit classification using PyTorch.

Create a model that:
- accepts flattened digit images as input
- produces scores/probabilities for 10 output classes

Requirements:
- Use two linear layers
- Use ReLU activation after the first linear layer
- Use dropout with p=0.2 after ReLU
- Use Sigmoid activation after the second linear layer

Implement:
- __init__()
- forward(images)

Notes:
- Assume valid input shapes.
- Input images are already flattened.
- Return a tensor of shape (N, 10), where N is batch size.

(Stop here if you want to solve without hints.)
"""

import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("Implement __init__()")

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        raise NotImplementedError("Implement forward()")


"""
============================================================
SECTION 2: OPTIONAL HELPERS / HINTS (Read only if needed)
============================================================

⚠️  Optional Help Section
Read this only if you want hints, clarifications, or helper APIs.

Reconstructed details (may not match original wording exactly):
- The model is a 2-layer MLP for digit classification.
- Input is a flattened image tensor (commonly shape: [N, 784]).
- Hidden layer size is 512.
- Output layer size is 10 (one per digit class).

Suggested architecture:
1) Linear(784 -> 512)
2) ReLU
3) Dropout(p=0.2)
4) Linear(512 -> 10)
5) Sigmoid

Helpful PyTorch classes:
- nn.Module
- nn.Linear
- nn.ReLU
- nn.Dropout
- nn.Sigmoid

Shape intuition:
- Input:  (N, 784)
- After layer1: (N, 512)
- After layer2: (N, 10)

Testing tips:
- Instantiate the model and pass a random tensor with shape (batch_size, 784)
- Verify output shape is (batch_size, 10)
- If using Sigmoid, outputs should be in [0, 1]
"""
