"""
============================================================
SECTION 1: QUESTION (Minimal Prompt + Skeleton)
============================================================

PyTorch Basics (Reconstructed Practice Problem)

Implement the methods in the Solution class using PyTorch.

Tasks:
1) reshape(to_reshape)
   - Reshape a 2D tensor so the output has exactly 2 columns.

2) average(to_avg)
   - Compute the mean across dimension 0.

3) concatenate(cat_one, cat_two)
   - Concatenate two tensors column-wise (dimension 1).

4) get_loss(prediction, target)
   - Compute Mean Squared Error (MSE) loss.

Notes:
- Assume valid inputs.
- Return PyTorch tensors.

(Stop here if you want to solve without hints.)
"""

import torch
import torch.nn
from torchtyping import TensorType


class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        print(to_reshape.shape)  # Debug: Check input shape
        return to_reshape.reshape(-1, 2)


    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        return to_avg.mean(dim=0)

    def concatenate(
        self,
        cat_one: TensorType[float],
        cat_two: TensorType[float],
    ) -> TensorType[float]:
        return torch.cat(
            (cat_one, cat_two), dim=1)

    def get_loss(
        self,
        prediction: TensorType[float],
        target: TensorType[float],
    ) -> TensorType[float]:
        mse_loss = torch.nn.functional.mse_loss(prediction, target)
        return mse_loss


"""
============================================================
SECTION 2: OPTIONAL HELPERS / HINTS (Read only if needed)
============================================================

⚠️  Optional Help Section
Read this only if you want hints, clarifications, or helper APIs.

Reconstructed details (may not match original wording exactly):
- reshape(to_reshape):
  * Input is a 2D tensor.
  * Output must have exactly 2 columns.
  * Number of rows should be inferred automatically.
  * You may assume total number of elements is divisible by 2.

- average(to_avg):
  * Mean should be computed across dimension 0
    (i.e., column-wise average across rows).

- concatenate(cat_one, cat_two):
  * Concatenate along dimension 1 (horizontal concatenation).
  * Assume both tensors have the same number of rows.

- get_loss(prediction, target):
  * Use Mean Squared Error (MSE) between prediction and target.

Helpful PyTorch functions:
- torch.reshape (or tensor.reshape)
- torch.mean (or tensor.mean)
- torch.cat
- torch.nn.functional.mse_loss

Example shape intuition:
- reshape:
    input shape  (2, 4) -> output shape (4, 2)

Suggested approach:
- Implement one method at a time.
- Run tests after each method.
"""
