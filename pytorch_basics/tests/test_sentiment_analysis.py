"""
Tests for sentiment_analysis.py

Run from the pytorch_lib/ directory:
    python tests/test_sentiment_analysis.py
"""

import torch
from sentiment_analysis import Solution


# ============================================================
# construction tests
# ============================================================

def test_model_is_nn_module():
    model = Solution(vocabulary_size=50)
    assert isinstance(model, torch.nn.Module), "Solution must inherit from torch.nn.Module"


def test_model_has_required_layers():
    model = Solution(vocabulary_size=123)

    assert hasattr(model, "embedding_layer"), "Missing attribute: embedding_layer"
    assert hasattr(model, "linear_layer"), "Missing attribute: linear_layer"
    assert hasattr(model, "activation"), "Missing attribute: activation"

    assert isinstance(model.embedding_layer, torch.nn.Embedding), "embedding_layer must be nn.Embedding"
    assert isinstance(model.linear_layer, torch.nn.Linear), "linear_layer must be nn.Linear"
    assert isinstance(model.activation, torch.nn.Sigmoid), "activation must be nn.Sigmoid"

    assert model.embedding_layer.num_embeddings == 123, (
        f"Expected num_embeddings == 123, got {model.embedding_layer.num_embeddings}"
    )
    assert model.embedding_layer.embedding_dim == 16, (
        f"Expected embedding_dim == 16, got {model.embedding_layer.embedding_dim}"
    )

    assert model.linear_layer.in_features == 16, (
        f"Expected linear_layer.in_features == 16, got {model.linear_layer.in_features}"
    )
    assert model.linear_layer.out_features == 1, (
        f"Expected linear_layer.out_features == 1, got {model.linear_layer.out_features}"
    )


# ============================================================
# forward pass tests
# ============================================================

def test_forward_output_shape_basic():
    model = Solution(vocabulary_size=100)
    model.eval()

    x = torch.tensor([
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [5, 6, 7, 8],
    ], dtype=torch.long)

    out = model(x)

    assert isinstance(out, torch.Tensor), "forward() must return a torch.Tensor"
    assert out.shape == (3, 1), f"Expected output shape (3, 1), got {out.shape}"


def test_forward_accepts_variable_sequence_length():
    model = Solution(vocabulary_size=50)
    model.eval()

    x = torch.tensor([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
    ], dtype=torch.long)

    out = model(x)

    assert out.shape == (4, 1), f"Expected output shape (4, 1), got {out.shape}"


def test_forward_output_in_sigmoid_range():
    model = Solution(vocabulary_size=200)
    model.eval()

    x = torch.randint(low=0, high=200, size=(5, 6), dtype=torch.long)
    out = model(x)

    assert torch.all(out >= 0).item(), "Sigmoid output should be >= 0"
    assert torch.all(out <= 1).item(), "Sigmoid output should be <= 1"


def test_forward_outputs_finite_values():
    model = Solution(vocabulary_size=20)
    model.eval()

    x = torch.zeros((2, 5), dtype=torch.long)
    out = model(x)

    assert out.shape == (2, 1), f"Expected output shape (2, 1), got {out.shape}"
    assert torch.isfinite(out).all().item(), "Output contains non-finite values (NaN/Inf)"


# ============================================================
# mean-over-sequence behavior (behavioral sanity checks)
# ============================================================

def test_forward_same_tokens_same_output_in_eval_mode():
    model = Solution(vocabulary_size=30)
    model.eval()

    # Identical rows should produce identical outputs in eval mode
    x = torch.tensor([
        [1, 2, 3, 4],
        [1, 2, 3, 4],
    ], dtype=torch.long)

    out = model(x)

    assert torch.allclose(out[0], out[1], atol=1e-7), (
        f"Identical inputs should produce identical outputs in eval mode.\nGot:\n{out}"
    )


# ============================================================
# Test runner
# ============================================================

def run_all_tests():
    test_model_is_nn_module()
    test_model_has_required_layers()
    test_forward_output_shape_basic()
    test_forward_accepts_variable_sequence_length()
    test_forward_output_in_sigmoid_range()
    test_forward_outputs_finite_values()
    test_forward_same_tokens_same_output_in_eval_mode()
    print("All sentiment_analysis tests passed ✅")


if __name__ == "__main__":
    run_all_tests()
