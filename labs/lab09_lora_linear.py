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
        )
        with torch.no_grad():
            fused.weight.copy_(self.linear.weight + self.delta_weight())
            if self.linear.bias is not None:
                fused.bias.copy_(self.linear.bias)
        return fused


def run_demo():
    torch.manual_seed(0)
    base = nn.Linear(8, 6)
    lora = LoRALinear(base, rank=2, alpha=4.0)
    x = torch.randn(32, 8)
    target = torch.randn(32, 6)

    initial_difference = (lora(x) - base(x)).abs().max().item()
    assert initial_difference == 0.0

    optimizer = torch.optim.AdamW(lora.parameters(), lr=1e-2)
    for _ in range(20):
        loss = F.mse_loss(lora(x), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    fused = lora.fuse()
    fused_difference = (lora(x) - fused(x)).abs().max().item()
    assert fused_difference < 1e-5

    print("initial LoRA/base difference:", initial_difference)
    print("trainable parameters:", sum(p.numel() for p in lora.parameters() if p.requires_grad))
    print("LoRA/fused difference:", fused_difference)
    return initial_difference, fused_difference


if __name__ == "__main__":
    run_demo()
