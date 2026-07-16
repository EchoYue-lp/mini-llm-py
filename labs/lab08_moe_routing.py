"""Lab 08: token-level Top-k Mixture-of-Experts routing."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.up = nn.Linear(d_model, hidden_dim)
        self.down = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class TopKMoE(nn.Module):
    def __init__(self, d_model=16, hidden_dim=32, num_experts=4, top_k=2):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Expert(d_model, hidden_dim) for _ in range(num_experts)]
        )

    def forward(self, x):
        original_shape = x.shape
        tokens = x.reshape(-1, x.size(-1))
        router_probs = torch.softmax(self.router(tokens), dim=-1)
        top_weights, top_indices = router_probs.topk(self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(tokens)
        expert_counts = []
        for expert_id, expert in enumerate(self.experts):
            token_indices, choice_indices = torch.where(top_indices == expert_id)
            expert_counts.append(int(token_indices.numel()))
            if token_indices.numel() == 0:
                continue
            expert_output = expert(tokens[token_indices])
            weighted = expert_output * top_weights[
                token_indices, choice_indices
            ].unsqueeze(-1)
            output.index_add_(0, token_indices, weighted)

        importance = router_probs.mean(dim=0)
        top1_load = F.one_hot(
            top_indices[:, 0], num_classes=self.num_experts
        ).float().mean(dim=0)
        load_balance_loss = self.num_experts * (importance * top1_load).sum()
        return (
            output.view(original_shape),
            load_balance_loss,
            top_indices.view(*original_shape[:-1], self.top_k),
            expert_counts,
        )


def run_demo():
    torch.manual_seed(0)
    x = torch.randn(2, 6, 16)
    moe = TopKMoE(num_experts=4, top_k=2)
    output, auxiliary_loss, routes, counts = moe(x)

    assert output.shape == x.shape
    assert routes.shape == (2, 6, 2)
    assert sum(counts) == 2 * 6 * 2
    assert torch.isfinite(auxiliary_loss)

    print("input/output:", x.shape, output.shape)
    print("top-2 routes for batch 0:\n", routes[0])
    print("expert assignments:", counts)
    print("load-balance auxiliary loss:", auxiliary_loss.item())


if __name__ == "__main__":
    run_demo()
