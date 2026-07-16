"""Lab 03: a Pre-LN residual block and its gradient path."""

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def run_demo():
    torch.manual_seed(0)
    x = torch.randn(2, 6, 32, requires_grad=True)
    block = TinyPreLNBlock()
    output, weights = block(x, create_causal_mask(6))
    output.square().mean().backward()

    assert output.shape == x.shape
    assert x.grad is not None and torch.isfinite(x.grad).all()

    print("input/output:", x.shape, output.shape)
    print("attention weights:", weights.shape)
    print("input gradient norm:", x.grad.norm().item())
    print("Pre-LN keeps an explicit identity path through both residual adds.")


if __name__ == "__main__":
    run_demo()
