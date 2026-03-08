"""
============================================================
SECTION 1: QUESTION (Minimal Prompt + Skeleton)
============================================================

GPT Dataset (Reconstructed Practice Problem)

In decoder-only language modeling (GPT-style), we train a model to predict the next token
given a context window of tokens.

Implement a batch loader that:
- takes a raw dataset string
- samples random contiguous windows of length `context_length`
- returns:
  X: the context tokens (each sequence length = context_length)
  Y: the next-token targets (each sequence length = context_length)

Function to implement:
- batch_loader(raw_dataset, context_length, batch_size) -> (X, Y)

Notes:
- X and Y should each be lists of token sequences (list of list of str).
- X[k] and Y[k] correspond to the same sampled window, shifted by one position.
- Use randomness to sample different windows.

(Stop here if you want to solve without hints.)
"""

import torch
from typing import List, Tuple


class Solution:
    def encode(self, token: str, vocab_dict: dict) -> int:
        # Placeholder encoding function (to be implemented)
        return vocab_dict[token]
    
    def decode(self, token_id: int, inv_vocab_dict: dict) -> str:
        return inv_vocab_dict[token_id]
        
    
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        dataset = raw_dataset.split()  # naive tokenization by whitespace
        vocab_dict = { token: idx for idx, token in enumerate(set(dataset)) }
        inv_vocab_dict = {idx: token for token, idx in vocab_dict.items()}
        data = torch.tensor([self.encode(token, vocab_dict) for token in dataset])  # convert tokens to ids
        # Sample random start indices for windows
        
        index = torch.randint(0, len(data)-context_length, (batch_size,))
        X = [data[i: i+context_length].tolist() for i in index]
        Y = [data[i+1: i+1 +context_length].tolist() for i in index]
  
        decoded_X = [[self.decode(token_id, inv_vocab_dict) for token_id in sequence] for sequence in X]
        decoded_Y = [[self.decode(token_id, inv_vocab_dict) for token_id in sequence] for sequence in Y]

        return (decoded_X, decoded_Y)

        
        


"""
============================================================
SECTION 2: OPTIONAL HELPERS / HINTS (Read only if needed)
============================================================

⚠️ Optional Help Section

Reconstructed details from expected structure:
- Build a vocabulary over tokens from raw_dataset split on whitespace.
- Create an encode mapping token->id and a decode mapping id->token.
- Convert the full dataset into a 1D tensor of token ids.
- Sample `batch_size` random start indices i where i+context_length+1 is valid.
- For each start i:
    X = data[i : i+context_length]
    Y = data[i+1 : i+context_length+1]
- Return X and Y decoded back to tokens (strings).
- Deterministic behavior may be enforced via torch.manual_seed(0) in some graders.

Output shapes:
- X: batch_size sequences of length context_length
- Y: batch_size sequences of length context_length
"""