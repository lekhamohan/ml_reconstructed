# PyTorch Library Practice Problems (Reconstructed)

> **Note:** These are **reconstructed practice prompts** based on public solution structures and may not match the original wording exactly.

This folder contains PyTorch-focused machine learning practice problems (in a NeetCode-style format) along with runnable tests.

The goal is to preserve and practice core ML/PyTorch exercises even when the original problem pages are unavailable.

---

## Folder Contents

### Problem Files
- `pytorch_basics.py` — basic tensor operations (`reshape`, `mean`, `cat`, `mse_loss`)
- `digit_classifier.py` — (to be documented/reconstructed)
- `nlp_intro.py` — (to be documented/reconstructed)
- `sentiment_analysis.py` — (to be documented/reconstructed)

### Tests
- `tests/test_pytorch_basics.py`
- `tests/test_digit_classifier.py` *(optional / add later)*
- `tests/test_nlp_intro.py` *(optional / add later)*
- `tests/test_sentiment_analysis.py` *(optional / add later)*

---

## Conventions Used in This Folder

### 1) One problem per Python file
Solutions can be found in `https://github.com/spolivin/neetcode_solutions_repo/tree/master/machine_learning/pytorch_lib`
Each `.py` file contains:
- a short problem docstring (reconstructed)
- a `Solution` class 
- methods expected by the problem

### 2) One test file per problem
Tests live inside the `tests/` folder and follow this naming pattern:

- `test_<problem_name>.py`

Example:
- `pytorch_basics.py` → `tests/test_pytorch_basics.py`

### 3) Reconstructed prompt policy
Where the original problem statement is unavailable, the prompt is reconstructed from:
- function signatures
- code comments
- expected tensor shapes / operations
- common ML/PyTorch patterns

---

## Setup

### Requirements
- Python 3.9+ (recommended)
- PyTorch
- `torchtyping` (used in type hints)

### Install dependencies

```bash
pip install torch torchtyping
