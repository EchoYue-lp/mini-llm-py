"""Lab 09: a minimal LoRA linear layer, training, and weight fusion."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, linear, rank=4, alpha=8.0):
        super().__init__()
        self.linear = linear
        self.linear.requires_grad_(False)
        self.scale = alpha / rank
        input_dim = linear.in_features
        output_dim = linear.out_features
        self.lora_a = nn.Parameter(torch.empty(input_dim, rank))
        self.lora_b = nn.Parameter(torch.zeros(rank, output_dim))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        return self.linear(x) + self.scale * (x @ self.lora_a @ self.lora_b)

    def delta_weight(self):
        return self.scale * (self.lora_a @ self.lora_b).T

    def fuse(self):
        fused = nn.Linear(
            self.linear.in_features,
            self.linear.out_features,
            bias=self.linear.bias is not None,
            device=self.linear.weight.device,
            dtype=self.linear.weight.dtype,
        )
        with torch.no_grad():
            fused.weight.copy_(self.linear.weight + self.delta_weight())
            if self.linear.bias is not None:
                fused.bias.copy_(self.linear.bias)
        return fused


def initial_gradient_norms(lora, x, target):
    loss = F.mse_loss(lora(x), target)
    grad_a, grad_b = torch.autograd.grad(
        loss,
        (lora.lora_a, lora.lora_b),
    )
    return grad_a.norm().item(), grad_b.norm().item()


def run_demo():
    torch.manual_seed(0)
    base = nn.Linear(8, 6)
    lora = LoRALinear(base, rank=2, alpha=4.0)
    x = torch.randn(32, 8)
    target = torch.randn(32, 6)

    initial_difference = (lora(x) - base(x)).abs().max().item()
    assert initial_difference == 0.0
    initial_a_grad, initial_b_grad = initial_gradient_norms(lora, x, target)
    assert initial_a_grad == 0.0
    assert initial_b_grad > 0.0

    optimizer = torch.optim.AdamW(
        (parameter for parameter in lora.parameters() if parameter.requires_grad),
        lr=1e-2,
    )
    for _ in range(20):
        loss = F.mse_loss(lora(x), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    lora.eval()
    fused = lora.fuse().eval()
    with torch.no_grad():
        fused_difference = (lora(x) - fused(x)).abs().max().item()
    assert fused_difference < 1e-5
    assert torch.linalg.matrix_rank(lora.delta_weight().float()) <= lora.lora_a.size(1)

    print("initial LoRA/base difference:", initial_difference)
    print("initial A/B gradient norms:", initial_a_grad, initial_b_grad)
    print("trainable parameters:", sum(p.numel() for p in lora.parameters() if p.requires_grad))
    print("Delta-W rank bound:", torch.linalg.matrix_rank(lora.delta_weight().float()).item(), "<=", lora.lora_a.size(1))
    print("LoRA/fused difference:", fused_difference)
    return initial_difference, fused_difference


if __name__ == "__main__":
    run_demo()
