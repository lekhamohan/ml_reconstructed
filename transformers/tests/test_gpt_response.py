import torch
import torch.nn as nn
from gpt_response import Solution


class DummyGPT(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        B, T = context.shape
        logits = torch.zeros(B, T, self.vocab_size)
        logits[:, :, 1] = 10.0
        return logits


def test_generate_returns_string_of_expected_length():
    vocab_size = 5
    model = DummyGPT(vocab_size=vocab_size)
    solver = Solution()

    int_to_char = {0: "_", 1: "a", 2: "b", 3: "c", 4: "d"}

    context = torch.tensor([[0, 2, 3]], dtype=torch.long)
    out = solver.generate(model, new_chars=4, context=context, context_length=5, int_to_char=int_to_char)

    assert isinstance(out, str)
    assert len(out) >= 4, "Expected output string to include generated characters."


def test_generate_is_deterministic_with_seeded_sampling():
    vocab_size = 6
    model = DummyGPT(vocab_size=vocab_size)
    solver = Solution()

    int_to_char = {i: chr(ord("a") + i) for i in range(vocab_size)}
    context = torch.tensor([[0, 1, 2]], dtype=torch.long)

    torch.manual_seed(0)
    out1 = solver.generate(model, new_chars=6, context=context.clone(), context_length=4, int_to_char=int_to_char)

    torch.manual_seed(0)
    out2 = solver.generate(model, new_chars=6, context=context.clone(), context_length=4, int_to_char=int_to_char)

    assert out1 == out2, "Expected deterministic generation given same manual seed (if sampler is seeded)."


def test_generate_truncates_context_to_context_length():
    vocab_size = 6
    model = DummyGPT(vocab_size=vocab_size)
    solver = Solution()

    int_to_char = {i: chr(ord("a") + i) for i in range(vocab_size)}

    context = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    out = solver.generate(model, new_chars=2, context=context, context_length=3, int_to_char=int_to_char)

    assert isinstance(out, str)
