"""Lab 06: prove cached autoregressive attention matches full attention."""

import math

import torch
import torch.nn as nn


class CachedSelfAttention(nn.Module):
    def __init__(self, d_model=16):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def full(self, x):
        if x.ndim != 3 or x.size(-1) != self.d_model:
            raise ValueError("x must have shape [B,T,d_model]")
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.d_model)
        mask = torch.tril(
            torch.ones(x.size(1), x.size(1), dtype=torch.bool, device=x.device)
        )
        weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
        return self.out_proj(weights @ value)

    def step(self, x, cache=None):
        if x.ndim != 3 or x.shape[1:] != (1, self.d_model):
            raise ValueError("step expects exactly one new token with shape [B,1,D]")
        query = self.q_proj(x)
        new_key = self.k_proj(x)
        new_value = self.v_proj(x)
        if cache is None:
            key, value = new_key, new_value
        else:
            if len(cache) != 2 or cache[0].shape != cache[1].shape:
                raise ValueError("cache must be a (key, value) pair with equal shapes")
            if cache[0].ndim != 3 or cache[0].size(0) != x.size(0):
                raise ValueError("cache batch shape must match the new token")
            if cache[0].size(-1) != self.d_model:
                raise ValueError("cache feature width must match d_model")
            key = torch.cat([cache[0], new_key], dim=1)
            value = torch.cat([cache[1], new_value], dim=1)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.d_model)
        weights = torch.softmax(scores, dim=-1)
        return self.out_proj(weights @ value), (key, value)


def projection_token_work(sequence_length):
    """Compare token projections with full-prefix recompute and KV caching."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    without_cache = sequence_length * (sequence_length + 1) // 2
    with_cache = sequence_length
    return without_cache, with_cache


def compare_full_and_cached():
    torch.manual_seed(0)
    model = CachedSelfAttention(d_model=16).eval()
    x = torch.randn(2, 7, 16)

    with torch.no_grad():
        full_output = model.full(x)
        cache = None
        incremental = []
        cache_lengths = []
        for position in range(x.size(1)):
            output, cache = model.step(x[:, position : position + 1], cache)
            incremental.append(output)
            cache_lengths.append(cache[0].size(1))
            assert cache[0].size(1) == position + 1
            assert cache[0].shape == cache[1].shape
        cached_output = torch.cat(incremental, dim=1)

    max_difference = (full_output - cached_output).abs().max().item()
    assert max_difference < 1e-6
    without_cache, with_cache = projection_token_work(x.size(1))
    print("full output:", full_output.shape)
    print("cached output:", cached_output.shape)
    print("final K/V cache:", cache[0].shape, cache[1].shape)
    print("cache length after each step:", cache_lengths)
    print("projected token positions without/with cache:", without_cache, with_cache)
    print("KV cache removes repeated projections; current Q still scans all history K/V.")
    print("max difference:", max_difference)
    return max_difference


if __name__ == "__main__":
    compare_full_and_cached()
