import torch
from gpt_dataset import Solution


def test_batch_loader_shapes_and_lengths():
    s = Solution()
    raw = "a b c d e f g h i j k l m n o p"
    context_length = 5
    batch_size = 4

    X, Y = s.batch_loader(raw, context_length, batch_size)

    assert isinstance(X, list) and isinstance(Y, list)
    assert len(X) == batch_size and len(Y) == batch_size

    for xseq, yseq in zip(X, Y):
        assert isinstance(xseq, list) and isinstance(yseq, list)
        assert len(xseq) == context_length
        assert len(yseq) == context_length


def test_batch_loader_shift_property():
    s = Solution()
    raw = "a b c d e f g h i j k l m n o p"
    context_length = 6
    batch_size = 3

    X, Y = s.batch_loader(raw, context_length, batch_size)

    for xseq, yseq in zip(X, Y):
        assert xseq[1:] == yseq[:-1], f"Shift property failed.\nX: {xseq}\nY: {yseq}"


def test_batch_loader_deterministic_with_seed():
    s = Solution()
    raw = "a b c d e f g h i j k l m n o p"
    context_length = 4
    batch_size = 2

    torch.manual_seed(0)
    X1, Y1 = s.batch_loader(raw, context_length, batch_size)

    torch.manual_seed(0)
    X2, Y2 = s.batch_loader(raw, context_length, batch_size)

    assert X1 == X2 and Y1 == Y2, "Expected deterministic output given the same torch.manual_seed()."
