import torch
from gpt_model import GPT


def test_gpt_output_shape():
    torch.manual_seed(0)
    vocab_size = 50
    context_length = 8
    model_dim = 16
    num_blocks = 2
    num_heads = 4

    model = GPT(vocab_size, context_length, model_dim, num_blocks, num_heads)
    model.eval()

    B, T = 3, 6
    context = torch.randint(0, vocab_size, (B, T), dtype=torch.long)
    out = model(context)

    assert out.shape == (B, T, vocab_size), f"Expected {(B, T, vocab_size)}, got {out.shape}"


def test_gpt_respects_context_length_by_accepting_smaller_T():
    torch.manual_seed(0)
    vocab_size = 40
    context_length = 10
    model_dim = 20
    num_blocks = 1
    num_heads = 4

    model = GPT(vocab_size, context_length, model_dim, num_blocks, num_heads)
    model.eval()

    context = torch.randint(0, vocab_size, (2, 3), dtype=torch.long)
    out = model(context)

    assert out.shape == (2, 3, vocab_size)


def test_gpt_positional_embeddings_affect_output():
    torch.manual_seed(0)
    vocab_size = 30
    context_length = 6
    model_dim = 16
    num_blocks = 1
    num_heads = 4

    model = GPT(vocab_size, context_length, model_dim, num_blocks, num_heads)
    model.eval()

    tok = 5
    ctx1 = torch.tensor([[tok, tok, tok, tok]], dtype=torch.long)
    out1 = model(ctx1)

    assert not torch.allclose(out1[:, 0, :], out1[:, 1, :], atol=1e-7), (
        "Logits at different positions are identical; positional embeddings may be missing."
    )
