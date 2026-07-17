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


def score_scale_statistics(samples=4096, head_dim=64):
    """Measure raw and scaled dot-product standard deviations."""
    query = torch.randn(samples, head_dim)
    key = torch.randn(samples, head_dim)
    raw = (query * key).sum(dim=-1)
    scaled = raw / math.sqrt(head_dim)
    return raw.std(unbiased=False), scaled.std(unbiased=False)


def attention_entropy(weights):
    """Return mean row entropy, treating zero probabilities as zero terms."""
    safe = weights.clamp_min(torch.finfo(weights.dtype).tiny)
    return -(weights * safe.log()).sum(dim=-1).mean()


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

    raw_std, scaled_std = score_scale_statistics()
    assert raw_std > scaled_std * 6
    assert 0.8 < scaled_std < 1.2

    wide_query = torch.randn(1, 16, 64)
    wide_key = torch.randn(1, 16, 64)
    raw_weights = torch.softmax(wide_query @ wide_key.transpose(-2, -1), dim=-1)
    scaled_weights = torch.softmax(
        wide_query @ wide_key.transpose(-2, -1) / math.sqrt(64),
        dim=-1,
    )
    raw_entropy = attention_entropy(raw_weights)
    scaled_entropy = attention_entropy(scaled_weights)
    assert scaled_entropy > raw_entropy

    print("Q/K/V:", query.shape, key.shape, value.shape)
    print("attention scores:", (1, 4, 4))
    print("output:", output.shape)
    print("causal attention weights:\n", weights[0])
    print("raw/scaled score std:", raw_std.item(), scaled_std.item())
    print("raw/scaled attention entropy:", raw_entropy.item(), scaled_entropy.item())


if __name__ == "__main__":
    run_demo()
