"""Lab 10: one implementation spanning MHA, MQA, and GQA."""

import math

import torch
import torch.nn as nn


class GroupedQuerySelfAttention(nn.Module):
    """Causal self-attention parameterized by query and KV head counts.

    num_kv_heads == num_query_heads -> MHA
    num_kv_heads == 1               -> MQA
    1 < num_kv_heads < query heads  -> GQA
    """

    def __init__(self, d_model=32, num_query_heads=8, num_kv_heads=2):
        super().__init__()
        if d_model % num_query_heads != 0:
            raise ValueError("d_model must be divisible by num_query_heads")
        if num_query_heads % num_kv_heads != 0:
            raise ValueError("num_query_heads must be divisible by num_kv_heads")

        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_query_heads
        self.q_proj = nn.Linear(d_model, num_query_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split(self, x, num_heads):
        batch_size, seq_len, _ = x.shape
        return (
            x.view(batch_size, seq_len, num_heads, self.head_dim)
            .transpose(1, 2)
        )

    def _expand_kv(self, x):
        repeats = self.num_query_heads // self.num_kv_heads
        return x.repeat_interleave(repeats, dim=1)

    def forward(self, x, mask=None):
        query = self._split(self.q_proj(x), self.num_query_heads)
        key_compact = self._split(self.k_proj(x), self.num_kv_heads)
        value_compact = self._split(self.v_proj(x), self.num_kv_heads)
        key = self._expand_kv(key_compact)
        value = self._expand_kv(value_compact)

        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is None:
            seq_len = x.size(1)
            mask = torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
            ).view(1, 1, seq_len, seq_len)
        weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
        attended = weights @ value
        merged = attended.transpose(1, 2).contiguous().view(
            x.size(0), x.size(1), self.d_model
        )
        return self.out_proj(merged), (key_compact, value_compact), weights

    def kv_cache_elements_per_token(self):
        return 2 * self.num_kv_heads * self.head_dim


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def run_demo():
    torch.manual_seed(0)
    x = torch.randn(2, 6, 32)
    variants = {
        "MHA": GroupedQuerySelfAttention(32, 8, 8),
        "GQA": GroupedQuerySelfAttention(32, 8, 2),
        "MQA": GroupedQuerySelfAttention(32, 8, 1),
    }

    parameter_counts = {}
    cache_sizes = {}
    for name, attention in variants.items():
        output, cache, weights = attention(x)
        assert output.shape == x.shape
        assert cache[0].shape == (2, attention.num_kv_heads, 6, 4)
        expanded_key = attention._expand_kv(cache[0])
        parameter_counts[name] = parameter_count(attention)
        cache_sizes[name] = attention.kv_cache_elements_per_token()
        print(
            f"{name}: Q heads={attention.num_query_heads}, "
            f"KV heads={attention.num_kv_heads}, "
            f"cache elements/token={attention.kv_cache_elements_per_token()}, "
            f"parameters={parameter_counts[name]}, "
            f"compact/expanded K elements={cache[0].numel()}/{expanded_key.numel()}, "
            f"output={tuple(output.shape)}, weights={tuple(weights.shape)}"
        )

    assert parameter_counts["MHA"] > parameter_counts["GQA"] > parameter_counts["MQA"]
    assert cache_sizes["MHA"] > cache_sizes["GQA"] > cache_sizes["MQA"]
    print("Production kernels keep compact KV and share it without materializing repeats.")


if __name__ == "__main__":
    run_demo()
