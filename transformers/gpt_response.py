"""
============================================================
SECTION 1: QUESTION (Minimal Prompt + Skeleton)
============================================================

GPT Response / Text Generation (Reconstructed Practice Problem)

Implement an autoregressive text generation method for a GPT-like model.

You are given:
- `model`: a GPT-like model that takes `context` (token IDs) and returns logits
  of shape (B, T, vocab_size)
- `new_chars`: number of new tokens/chars to generate
- `context`: initial context token IDs, shape (B, T)
- `context_length`: maximum context window size used by the model
- `int_to_char`: dictionary mapping token IDs (int) to characters (str)

At each generation step:
1) If context length exceeds `context_length`, truncate to the last `context_length` tokens.
2) Run the model on the current context to obtain logits.
3) Take the logits at the final time step (the last token position).
4) Convert logits to probabilities (softmax).
5) Sample the next token from the probability distribution.
6) Append the sampled token to the context.

Finally, decode the resulting token sequence into a string using `int_to_char`
and return it.

Implement:
- generate(model, new_chars, context, context_length, int_to_char) -> str

Notes:
- Use sampling (not greedy argmax).
- Assume valid inputs and mappings.
- Return a decoded string.

(Stop here if you want to solve without hints.)
"""

import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution:
    def generate(
        self,
        model,
        new_chars: int,
        context: TensorType[int],
        context_length: int,
        int_to_char: dict,
    ) -> str:
        raise NotImplementedError("Implement generate()")


"""
============================================================
SECTION 2: OPTIONAL HELPERS / HINTS (Read only if needed)
============================================================

⚠️ Optional Help Section

Reconstructed details (may not match original wording exactly):

Expected tensor shapes:
- context: (B, T)
- model(context) returns logits: (B, T, vocab_size)
- last-step logits: logits[:, -1, :] -> (B, vocab_size)
- probs: softmax(last_logits, dim=-1) -> (B, vocab_size)
- sampled next token: (B, 1)
- updated context after concat: (B, T+1)

Key PyTorch utilities:
- nn.functional.softmax(logits, dim=-1)
- torch.multinomial(probs, num_samples=1)   # sampling next token
- torch.cat((context, idx_next), dim=1)

Causal context window:
- If context is longer than context_length, keep only:
    context = context[:, -context_length:]

Determinism:
- Some graders may set a manual seed to make sampling reproducible.

Decoding:
- Convert a list of token IDs to a string using int_to_char:
    ''.join(int_to_char[i] for i in ids)
"""