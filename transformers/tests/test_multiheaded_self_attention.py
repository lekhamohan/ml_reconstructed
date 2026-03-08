import torch
from multiheaded_self_attention import MultiHeadedSelfAttention


def test_multihead_output_shape():
    torch.manual_seed(0)
    B, T, C = 2, 5, 12
    attention_dim = 12
    num_heads = 3

    mha = MultiHeadedSelfAttention(embedding_dim=C, attention_dim=attention_dim, num_heads=num_heads)
    x = torch.randn(B, T, C)
    out = mha(x)

    assert out.shape == (B, T, attention_dim), f"Expected {(B, T, attention_dim)}, got {out.shape}"


def test_multihead_is_causal_future_tokens_do_not_affect_past():
    torch.manual_seed(0)
    B, T, C = 1, 6, 12
    attention_dim = 12
    num_heads = 4

    mha = MultiHeadedSelfAttention(embedding_dim=C, attention_dim=attention_dim, num_heads=num_heads)
    mha.eval()

    x1 = torch.randn(B, T, C)
    x2 = x1.clone()
    x2[:, -1, :] -= 50.0

    out1 = mha(x1)
    out2 = mha(x2)

    assert torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-5), (
        "Causality failed in multi-head: past outputs changed when future token changed."
    )
