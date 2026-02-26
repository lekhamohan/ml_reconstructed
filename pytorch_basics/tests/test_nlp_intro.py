"""
Tests for nlp_intro.py

Run from the pytorch_lib/ directory:
    python tests/test_nlp_intro.py
"""

import torch
from nlp_intro import Solution


# ============================================================
# shape / basic behavior tests
# ============================================================

def test_returns_tensor():
    s = Solution()
    positive = ["good movie", "very good"]
    negative = ["bad movie"]

    out = s.get_dataset(positive, negative)

    assert isinstance(out, torch.Tensor), "get_dataset() must return a torch.Tensor"


def test_output_shape_batch_first():
    s = Solution()
    positive = ["good movie", "very very good"]
    negative = ["bad"]

    out = s.get_dataset(positive, negative)

    # total sentences = 3
    # max token length = 3 ("very very good")
    assert out.shape == (3, 3), f"Expected shape (3, 3), got {out.shape}"


def test_positive_then_negative_order():
    s = Solution()
    positive = ["alpha beta", "gamma"]
    negative = ["delta epsilon"]

    out = s.get_dataset(positive, negative)

    # Row count should follow positive sentences first, then negative
    assert out.shape[0] == 3, f"Expected 3 rows, got {out.shape[0]}"

    # Check row lengths by counting non-zero entries (padding assumed zero)
    nonzero_counts = [(row != 0).sum().item() for row in out]
    expected_counts = [2, 1, 2]
    assert nonzero_counts == expected_counts, f"Expected row token counts {expected_counts}, got {nonzero_counts}"


# ============================================================
# vocabulary / deterministic encoding tests
# ============================================================

def test_sorted_vocabulary_mapping_expected_ids():
    s = Solution()

    # Vocabulary across all sentences: {"apple", "banana", "carrot"}
    # Sorted -> ["apple", "banana", "carrot"]
    # IDs expected -> apple:1, banana:2, carrot:3
    positive = ["banana apple"]
    negative = ["carrot"]

    out = s.get_dataset(positive, negative)

    expected = torch.tensor([
        [2, 1],  # "banana apple"
        [3, 0],  # "carrot" + padding
    ])

    assert out.shape == (2, 2), f"Expected shape (2, 2), got {out.shape}"
    assert torch.equal(out, expected), f"Sorted vocab encoding failed.\nGot:\n{out}\nExpected:\n{expected}"


def test_repeated_words_keep_same_id():
    s = Solution()
    positive = ["wow wow wow"]
    negative = ["wow"]

    out = s.get_dataset(positive, negative)

    # Only one unique word => ID should be 1 everywhere non-padding
    expected = torch.tensor([
        [1, 1, 1],
        [1, 0, 0],
    ])

    assert torch.equal(out, expected), f"Repeated word encoding failed.\nGot:\n{out}\nExpected:\n{expected}"


# ============================================================
# padding behavior tests
# ============================================================

def test_padding_is_zero_and_batch_first():
    s = Solution()
    positive = ["one two three", "one"]
    negative = ["two three"]

    out = s.get_dataset(positive, negative)

    # 3 sentences total, max length 3
    assert out.shape == (3, 3), f"Expected shape (3, 3), got {out.shape}"

    # verify there is at least some padding zeros
    assert (out == 0).any().item(), "Expected padded zeros in output"

    # row-wise non-zero counts should be [3,1,2]
    nonzero_counts = [(row != 0).sum().item() for row in out]
    assert nonzero_counts == [3, 1, 2], f"Expected token counts [3, 1, 2], got {nonzero_counts}"


def test_no_padding_needed_when_lengths_match():
    s = Solution()
    positive = ["a b", "c d"]
    negative = ["e f"]

    out = s.get_dataset(positive, negative)

    # all sentences length 2 -> no padding zeros expected
    assert out.shape == (3, 2), f"Expected shape (3, 2), got {out.shape}"
    assert not (out == 0).any().item(), f"Did not expect padding zeros.\nGot:\n{out}"


# ============================================================
# Test runner
# ============================================================

def run_all_tests():
    test_returns_tensor()
    test_output_shape_batch_first()
    test_positive_then_negative_order()
    test_sorted_vocabulary_mapping_expected_ids()
    test_repeated_words_keep_same_id()
    test_padding_is_zero_and_batch_first()
    test_no_padding_needed_when_lengths_match()
    print("All nlp_intro tests passed ✅")


if __name__ == "__main__":
    run_all_tests()
