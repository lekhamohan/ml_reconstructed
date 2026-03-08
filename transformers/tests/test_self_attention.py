import torch
from self_attention import SingleHeadAttention


def test_self_attention_output_shape():
    torch.manual_seed(0)
    B, T, C = 2, 5, 12
    D = 8
    attn = SingleHeadAttention(embedding_dim=C, attention_dim=D)

    x = torch.randn(B, T, C)
    out = attn(x)

    assert out.shape == (B, T, D), f"Expected output shape {(B, T, D)}, got {out.shape}"


def test_self_attention_is_causal_future_tokens_do_not_affect_past():
    torch.manual_seed(0)
    B, T, C = 1, 6, 10
    D = 6
    attn = SingleHeadAttention(embedding_dim=C, attention_dim=D)
    attn.eval()

    x1 = torch.randn(B, T, C)
    x2 = x1.clone()
    x2[:, -1, :] += 100.0

    out1 = attn(x1)
    out2 = attn(x2)

    assert torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-5), (
        "Causality failed: earlier outputs changed when only future token changed."
    )
