"""Lab 07: RMSNorm, RoPE, and SwiGLU used by many modern LLMs."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        if d_model <= 0 or eps <= 0:
            raise ValueError("d_model and eps must be positive")
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x_float = x.float()
        inverse_rms = x_float.pow(2).mean(dim=-1, keepdim=True).add(
            self.eps
        ).rsqrt()
        return (x_float * inverse_rms * self.weight.float()).to(input_dtype)


def apply_rope(x, base=10000.0, position_offset=0):
    """Apply rotary position embeddings to [batch, heads, seq, head_dim]."""
    if x.ndim != 4:
        raise ValueError("RoPE input must have shape [B,H,T,Dh]")
    if base <= 0:
        raise ValueError("RoPE base must be positive")
    if not isinstance(position_offset, int) or position_offset < 0:
        raise ValueError("position_offset must be a non-negative integer")
    head_dim = x.size(-1)
    if head_dim % 2:
        raise ValueError("RoPE requires an even head dimension")
    positions = torch.arange(
        position_offset,
        position_offset + x.size(-2),
        device=x.device,
        dtype=torch.float32,
    )
    inverse_frequency = 1.0 / (
        base
        ** (
            torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32)
            / head_dim
        )
    )
    angles = positions[:, None] * inverse_frequency[None, :]
    cosine = angles.cos().to(x.dtype).view(1, 1, x.size(-2), head_dim // 2)
    sine = angles.sin().to(x.dtype).view(1, 1, x.size(-2), head_dim // 2)

    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated = torch.stack(
        [even * cosine - odd * sine, even * sine + odd * cosine], dim=-1
    )
    return rotated.flatten(-2)


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        if d_model <= 0 or hidden_dim <= 0:
            raise ValueError("d_model and hidden_dim must be positive")
        self.gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.up = nn.Linear(d_model, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def equal_budget_swiglu_hidden(d_model, classic_hidden):
    """Match 3*D*F SwiGLU weights to 2*D*Dff classic FFN weights."""
    if d_model <= 0 or classic_hidden <= 0:
        raise ValueError("d_model and classic_hidden must be positive")
    return round(2 * classic_hidden / 3)


def run_demo():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 6, 8)
    rotated = apply_rope(x)
    offset_rotated = apply_rope(x, position_offset=11)
    assert torch.allclose(x.norm(dim=-1), rotated.norm(dim=-1), atol=1e-5)
    assert torch.allclose(x.norm(dim=-1), offset_rotated.norm(dim=-1), atol=1e-5)
    assert not torch.allclose(rotated, offset_rotated)

    hidden = torch.randn(2, 6, 32)
    rms_norm = RMSNorm(32)
    normalized = rms_norm(hidden)
    layer_normalized = nn.LayerNorm(32, elementwise_affine=False)(hidden)
    swiglu = SwiGLU(32, 64)
    ffn_output = swiglu(normalized)
    assert normalized.shape == hidden.shape == ffn_output.shape
    assert torch.allclose(
        layer_normalized.mean(dim=-1),
        torch.zeros_like(layer_normalized.mean(dim=-1)),
        atol=1e-6,
    )

    classic_hidden = 4 * 32
    equal_budget_hidden = equal_budget_swiglu_hidden(32, classic_hidden)
    classic_weight_count = 2 * 32 * classic_hidden
    equal_budget_swiglu_weights = 3 * 32 * equal_budget_hidden

    print("RoPE input/output:", x.shape, rotated.shape)
    print("RoPE position offsets 0 and 11 preserve norm but change phase")
    print("RMSNorm output:", normalized.shape)
    print(
        "LayerNorm/RMSNorm mean for token 0:",
        layer_normalized[0, 0].mean().item(),
        normalized[0, 0].mean().item(),
    )
    print("SwiGLU output/parameters:", ffn_output.shape, parameter_count(swiglu))
    print(
        "classic/equal-budget SwiGLU weight counts:",
        classic_weight_count,
        equal_budget_swiglu_weights,
        "hidden:",
        equal_budget_hidden,
    )


if __name__ == "__main__":
    run_demo()
