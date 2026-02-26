"""
Tests for digit_classifier.py

Run from the pytorch_lib/ directory:
    python tests/test_digit_classifier.py
"""

import torch
from digit_classifier import Solution


# ============================================================
# construction tests
# ============================================================

def test_model_is_nn_module():
    model = Solution()
    assert isinstance(model, torch.nn.Module), "Solution must inherit from torch.nn.Module"


def test_model_has_required_layers():
    model = Solution()

    assert hasattr(model, "linear_layer_1"), "Missing attribute: linear_layer_1"
    assert hasattr(model, "linear_layer_2"), "Missing attribute: linear_layer_2"

    assert isinstance(model.linear_layer_1, torch.nn.Linear), "linear_layer_1 must be nn.Linear"
    assert isinstance(model.linear_layer_2, torch.nn.Linear), "linear_layer_2 must be nn.Linear"

    assert model.linear_layer_1.in_features == 784, f"Expected linear_layer_1.in_features == 784, got {model.linear_layer_1.in_features}"
    assert model.linear_layer_1.out_features == 512, f"Expected linear_layer_1.out_features == 512, got {model.linear_layer_1.out_features}"

    assert model.linear_layer_2.in_features == 512, f"Expected linear_layer_2.in_features == 512, got {model.linear_layer_2.in_features}"
    assert model.linear_layer_2.out_features == 10, f"Expected linear_layer_2.out_features == 10, got {model.linear_layer_2.out_features}"


# ============================================================
# forward pass tests
# ============================================================

def test_forward_output_shape_batch_1():
    model = Solution()
    model.eval()

    x = torch.randn(1, 784)
    out = model(x)

    assert isinstance(out, torch.Tensor), "forward() must return a torch.Tensor"
    assert out.shape == (1, 10), f"Expected output shape (1, 10), got {out.shape}"


def test_forward_output_shape_batch_8():
    model = Solution()
    model.eval()

    x = torch.randn(8, 784)
    out = model(x)

    assert out.shape == (8, 10), f"Expected output shape (8, 10), got {out.shape}"


def test_forward_output_values_in_sigmoid_range():
    model = Solution()
    model.eval()

    x = torch.randn(4, 784)
    out = model(x)

    assert torch.all(out >= 0).item(), "Sigmoid output should be >= 0"
    assert torch.all(out <= 1).item(), "Sigmoid output should be <= 1"


def test_forward_accepts_zero_input():
    model = Solution()
    model.eval()

    x = torch.zeros(3, 784)
    out = model(x)

    assert out.shape == (3, 10), f"Expected output shape (3, 10), got {out.shape}"
    assert torch.isfinite(out).all().item(), "Output contains non-finite values (NaN/Inf)"


# ============================================================
# Test runner
# ============================================================

def run_all_tests():
    test_model_is_nn_module()
    test_model_has_required_layers()
    test_forward_output_shape_batch_1()
    test_forward_output_shape_batch_8()
    test_forward_output_values_in_sigmoid_range()
    test_forward_accepts_zero_input()
    print("All digit_classifier tests passed ✅")


if __name__ == "__main__":
    run_all_tests()
