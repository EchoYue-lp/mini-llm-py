"""Lab 02: split a hidden state into heads and merge it back."""

import torch
import torch.nn as nn

from labs.lab01_attention_basics import scaled_dot_product_attention


def split_heads(x, num_heads):
    batch_size, seq_len, d_model = x.shape
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    head_dim = d_model // num_heads
    return x.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)


def merge_heads(x):
    batch_size, num_heads, seq_len, head_dim = x.shape
    return (
        x.transpose(1, 2)
        .contiguous()
        .view(batch_size, seq_len, num_heads * head_dim)
    )


class TinyMultiHeadAttention(nn.Module):
    def __init__(self, d_model=16, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        query = split_heads(self.q_proj(x), self.num_heads)
        key = split_heads(self.k_proj(x), self.num_heads)
        value = split_heads(self.v_proj(x), self.num_heads)
        attended, weights = scaled_dot_product_attention(
            query, key, value, mask
        )
        return self.out_proj(merge_heads(attended)), weights


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def run_demo():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    mask = torch.tril(torch.ones(5, 5, dtype=torch.bool)).view(1, 1, 5, 5)
    model = TinyMultiHeadAttention(d_model=16, num_heads=4)
    output, weights = model(x, mask)
    heads = split_heads(x, 4)

    assert output.shape == x.shape
    assert weights.shape == (2, 4, 5, 5)
    assert torch.allclose(merge_heads(heads), x)
    assert not heads.is_contiguous()
    assert parameter_count(model) == 4 * (16 * 16 + 16)

    print("input:", x.shape)
    print("per-head Q/K/V:", (2, 4, 5, 4))
    print("attention weights:", weights.shape)
    print("merged output:", output.shape)
    print("split stride:", heads.stride(), "contiguous:", heads.is_contiguous())
    print("Q/K/V/O parameters including bias:", parameter_count(model))


if __name__ == "__main__":
    run_demo()
