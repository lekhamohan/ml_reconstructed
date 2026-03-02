"""
============================================================
SECTION 1: QUESTION (Minimal Prompt + Skeleton)
============================================================

NLP Intro (Reconstructed Practice Problem)

Implement a simple dataset encoding function for text classification preprocessing.

Given two lists of sentences:
- positive
- negative

Build a vocabulary from all words, assign each unique word an integer ID, and
convert each sentence into a sequence of token IDs.

Return the encoded dataset as a padded PyTorch tensor.

Implement:
- get_dataset(positive, negative)

Notes:
- Assume valid inputs.
- Tokenization can be done by splitting on whitespace.
- Return a tensor with batch-first padding.

(Stop here if you want to solve without hints.)
"""

from typing import List

import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        vocab = set()

        for sentence in positive + negative:
            vocab.update(sentence.split())

        vocab = sorted(vocab)
        word2id = {w: idx for idx, w in enumerate(vocab, start=1)}
         
        def encode(sentence_split: list[str])->TensorType[float]:
            return torch.tensor([word2id[s] for s in sentence_split])
        
        encoded_dataset = []
        for sentence in positive + negative:
            encoded_dataset.append(encode(sentence.split()))
        
            
        return nn.utils.rnn.pad_sequence(encoded_dataset, batch_first=True)
    




        


"""
============================================================
SECTION 2: OPTIONAL HELPERS / HINTS (Read only if needed)
============================================================

⚠️  Optional Help Section
Read this only if you want hints, clarifications, or helper APIs.

Reconstructed details (may not match original wording exactly):
- Build a vocabulary using words from both positive and negative sentences.
- Tokenization is whitespace-based (sentence.split()).
- Sort the vocabulary before assigning IDs (for deterministic encoding).
- Assign integer IDs starting from 1.
- Encode each sentence as a tensor of token IDs.
- Combine all encoded sentences into one padded tensor using batch-first format.

Ordering:
- Encoded outputs should include all positive sentences first,
  then all negative sentences.

Padding:
- Use PyTorch sequence padding utilities.
- Batch-first output means shape should be (num_sentences, max_seq_len).

Helpful PyTorch APIs:
- nn.utils.rnn.pad_sequence
- torch.tensor

Suggested approach:
1) Collect unique words across both lists
2) Sort vocabulary
3) Build word->id mapping (start IDs at 1)
4) Encode each sentence
5) Pad sequences with batch_first=True
"""
