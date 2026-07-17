"""Lab 03: a Pre-LN residual block and its gradient path."""

import torch
import torch.nn as nn

from models.layers import MultiHeadAttention, PositionwiseFeedForward
from utils.mask_utils import create_causal_mask


class TinyPreLNBlock(nn.Module):
    def __init__(self, d_model=32, num_heads=4, d_ff=64):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout=0.0)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=0.0)

    def forward(self, x, mask):
        normalized = self.attn_norm(x)
        attention_output, weights = self.attn(
            normalized, normalized, normalized, mask
        )
        x = x + attention_output
        x = x + self.ffn(self.ffn_norm(x))
        return x, weights


def identity_path_gradient(d_model=8, seq_len=4):
    """Zero every residual branch and verify output/gradient stay identity."""
    block = TinyPreLNBlock(d_model=d_model, num_heads=2, d_ff=16)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.zero_()
    x = torch.randn(1, seq_len, d_model, requires_grad=True)
    output, _ = block(x, create_causal_mask(seq_len))
    output.sum().backward()
    return (output - x).abs().max().item(), x.grad


def run_demo():
    torch.manual_seed(0)
    x = torch.randn(2, 6, 32, requires_grad=True)
    block = TinyPreLNBlock()
    output, weights = block(x, create_causal_mask(6))
    output.square().mean().backward()

    assert output.shape == x.shape
    assert x.grad is not None and torch.isfinite(x.grad).all()
    identity_difference, identity_gradient = identity_path_gradient()
    assert identity_difference == 0.0
    assert torch.equal(identity_gradient, torch.ones_like(identity_gradient))

    normalized = block.attn_norm(x.detach())
    final_normalized = nn.LayerNorm(x.size(-1))(output.detach())
    input_rms = x.detach().pow(2).mean().sqrt()
    normalized_rms = normalized.pow(2).mean().sqrt()
    final_rms = final_normalized.pow(2).mean().sqrt()

    print("input/output:", x.shape, output.shape)
    print("attention weights:", weights.shape)
    print("input gradient norm:", x.grad.norm().item())
    print("input/normed RMS:", input_rms.item(), normalized_rms.item())
    print("final norm RMS after residual stack:", final_rms.item())
    print("zero-branch identity output max diff:", identity_difference)
    print("zero-branch input gradient unique values:", identity_gradient.unique().tolist())
    print("Pre-LN keeps an explicit identity path through both residual adds.")


if __name__ == "__main__":
    run_demo()
