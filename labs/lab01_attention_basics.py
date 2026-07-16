"""Lab 01: scaled dot-product attention and causal masking."""

import math

import torch


def scaled_dot_product_attention(query, key, value, mask=None):
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    output = weights @ value
    return output, weights


def run_demo():
    torch.manual_seed(0)
    query = torch.randn(1, 4, 8)
    key = torch.randn(1, 4, 8)
    value = torch.randn(1, 4, 8)
    causal_mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))

    output, weights = scaled_dot_product_attention(
        query, key, value, causal_mask
    )

    assert output.shape == (1, 4, 8)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 4))
    assert torch.count_nonzero(weights[0].triu(diagonal=1)) == 0

    print("Q/K/V:", query.shape, key.shape, value.shape)
    print("attention scores:", (1, 4, 4))
    print("output:", output.shape)
    print("causal attention weights:\n", weights[0])


if __name__ == "__main__":
    run_demo()
