"""Lab 07: RMSNorm, RoPE, and SwiGLU used by many modern LLMs."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def apply_rope(x, base=10000.0):
    """Apply rotary position embeddings to [batch, heads, seq, head_dim]."""

    head_dim = x.size(-1)
    if head_dim % 2:
        raise ValueError("RoPE requires an even head dimension")
    positions = torch.arange(x.size(-2), device=x.device, dtype=x.dtype)
    inverse_frequency = 1.0 / (
        base
        ** (torch.arange(0, head_dim, 2, device=x.device, dtype=x.dtype) / head_dim)
    )
    angles = positions[:, None] * inverse_frequency[None, :]
    cosine = angles.cos().view(1, 1, x.size(-2), head_dim // 2)
    sine = angles.sin().view(1, 1, x.size(-2), head_dim // 2)

    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated = torch.stack(
        [even * cosine - odd * sine, even * sine + odd * cosine], dim=-1
    )
    return rotated.flatten(-2)


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.up = nn.Linear(d_model, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def run_demo():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 6, 8)
    rotated = apply_rope(x)
    assert torch.allclose(x.norm(dim=-1), rotated.norm(dim=-1), atol=1e-5)

    hidden = torch.randn(2, 6, 32)
    normalized = RMSNorm(32)(hidden)
    ffn_output = SwiGLU(32, 64)(normalized)
    assert normalized.shape == hidden.shape == ffn_output.shape

    print("RoPE input/output:", x.shape, rotated.shape)
    print("RMSNorm output:", normalized.shape)
    print("SwiGLU output:", ffn_output.shape)


if __name__ == "__main__":
    run_demo()
