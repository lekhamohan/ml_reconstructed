import torch
from transformer_block import TransformerBlock


def test_transformer_block_output_shape():
    torch.manual_seed(0)
    B, T, C = 2, 6, 16
    num_heads = 4

    block = TransformerBlock(model_dim=C, num_heads=num_heads)
    x = torch.randn(B, T, C)
    out = block(x)

    assert out.shape == (B, T, C), f"Expected output shape {(B, T, C)}, got {out.shape}"


def test_transformer_block_is_deterministic_in_eval():
    torch.manual_seed(0)
    B, T, C = 1, 5, 16
    num_heads = 4
    block = TransformerBlock(model_dim=C, num_heads=num_heads)
    block.eval()

    x = torch.randn(B, T, C)
    out1 = block(x)
    out2 = block(x)

    assert torch.allclose(out1, out2, atol=1e-6), "Expected deterministic output in eval mode."
