"""
Tests for pytorch_basics.py

Run from the pytorch_lib/ directory:
    python tests/test_pytorch_basics.py
"""

import torch
from pytorch_basics import Solution


# ============================================================
# reshape() tests
# ============================================================

def test_reshape_basic():
    s = Solution()
    x = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
    ])

    out = s.reshape(x)

    expected = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
    ])

    assert out.shape == (4, 2), f"Expected shape (4, 2), got {out.shape}"
    assert torch.equal(out, expected), f"reshape failed.\nGot:\n{out}\nExpected:\n{expected}"


def test_reshape_already_two_columns():
    s = Solution()
    x = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])

    out = s.reshape(x)
    expected = x

    assert out.shape == (3, 2), f"Expected shape (3, 2), got {out.shape}"
    assert torch.equal(out, expected), f"reshape (already 2 cols) failed.\nGot:\n{out}\nExpected:\n{expected}"


# ============================================================
# average() tests
# ============================================================

def test_average_basic():
    s = Solution()
    x = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    out = s.average(x)
    expected = torch.tensor([2.5, 3.5, 4.5])

    assert out.shape == (3,), f"Expected shape (3,), got {out.shape}"
    assert torch.allclose(out, expected), f"average failed.\nGot:\n{out}\nExpected:\n{expected}"


def test_average_with_negatives():
    s = Solution()
    x = torch.tensor([
        [1.0, -1.0],
        [3.0, -3.0],
        [5.0, -5.0],
    ])

    out = s.average(x)
    expected = torch.tensor([3.0, -3.0])

    assert torch.allclose(out, expected), f"average with negatives failed.\nGot:\n{out}\nExpected:\n{expected}"


# ============================================================
# concatenate() tests
# ============================================================

def test_concatenate_basic():
    s = Solution()
    a = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])
    b = torch.tensor([
        [5.0, 6.0],
        [7.0, 8.0],
    ])

    out = s.concatenate(a, b)

    expected = torch.tensor([
        [1.0, 2.0, 5.0, 6.0],
        [3.0, 4.0, 7.0, 8.0],
    ])

    assert out.shape == (2, 4), f"Expected shape (2, 4), got {out.shape}"
    assert torch.equal(out, expected), f"concatenate failed.\nGot:\n{out}\nExpected:\n{expected}"


def test_concatenate_single_column_inputs():
    s = Solution()
    a = torch.tensor([[1.0], [2.0], [3.0]])
    b = torch.tensor([[10.0], [20.0], [30.0]])

    out = s.concatenate(a, b)

    expected = torch.tensor([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
    ])

    assert out.shape == (3, 2), f"Expected shape (3, 2), got {out.shape}"
    assert torch.equal(out, expected), f"concatenate single-column failed.\nGot:\n{out}\nExpected:\n{expected}"


# ============================================================
# get_loss() tests
# ============================================================

def test_get_loss_basic():
    s = Solution()
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 5.0])

    out = s.get_loss(pred, target)
    expected = torch.tensor(4.0 / 3.0)  # squared errors = [0, 0, 4]

    assert torch.allclose(out, expected, atol=1e-6), f"get_loss failed.\nGot:\n{out}\nExpected:\n{expected}"


def test_get_loss_zero():
    s = Solution()
    pred = torch.tensor([[1.0], [2.0], [3.0]])
    target = torch.tensor([[1.0], [2.0], [3.0]])

    out = s.get_loss(pred, target)
    expected = torch.tensor(0.0)

    assert torch.allclose(out, expected, atol=1e-6), f"get_loss zero case failed.\nGot:\n{out}\nExpected:\n{expected}"


# ============================================================
# Test runner
# ============================================================

def run_all_tests():
    test_reshape_basic()
    test_reshape_already_two_columns()
    test_average_basic()
    test_average_with_negatives()
    test_concatenate_basic()
    test_concatenate_single_column_inputs()
    test_get_loss_basic()
    test_get_loss_zero()
    print("All pytorch_basics tests passed ✅")


if __name__ == "__main__":
    run_all_tests()
