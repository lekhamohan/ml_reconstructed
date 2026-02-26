# PyTorch Library Practice Problems (Reconstructed)

> **Note:** These are reconstructed practice prompts based on public solution structures and may not match the original wording exactly.

This folder contains PyTorch-focused ML practice problems (NeetCode-style) with **starter skeletons** and **runnable tests**.

## Purpose

The goal is to preserve and practice core ML/PyTorch exercises even when the original problem pages are unavailable.

## Folder Contents

### Problem files
- `pytorch_basics.py` — tensor operations (`reshape`, `mean`, `cat`, `mse_loss`)
- `digit_classifier.py` — simple FFN
- `nlp_intro.py` — dataset encoder
- `sentiment_analysis.py` — simple pytorch model 

### Tests
- `tests/test_pytorch_basics.py`
- `tests/test_digit_classifier.py` 
- `tests/test_nlp_intro.py` 
- `tests/test_sentiment_analysis.py` 

## Conventions

### 1) One problem per Python file
Each problem file contains:
- a **minimal prompt**
- an **optional hints/details section**
- a `Solution` class skeleton (`NotImplementedError` methods)

### 2) One test file per problem
Tests live in `tests/` and follow this naming convention:

- `test_<problem_name>.py`

Example:
- `pytorch_basics.py` → `tests/test_pytorch_basics.py`

### 3) Reconstructed prompt policy
When the original prompt is unavailable, the question is reconstructed from:
- function signatures
- code comments
- expected tensor shapes / operations
- standard ML/PyTorch patterns

## Reference (Solutions Source)

Public solutions that helped infer structure:
- `spolivin/neetcode_solutions_repo` (PyTorch lib folder)

(These prompts are reconstructed independently and may differ from the original wording.)

## Setup

### Requirements
- Python 3.9+ (recommended)
- `torch`
- `torchtyping`

### Install dependencies

```bash
pip install torch torchtyping
