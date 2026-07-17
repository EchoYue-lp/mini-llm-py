"""Lab 08: token-level Top-k Mixture-of-Experts routing."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        if d_model <= 0 or hidden_dim <= 0:
            raise ValueError("d_model and hidden_dim must be positive")
        self.up = nn.Linear(d_model, hidden_dim)
        self.down = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class TopKMoE(nn.Module):
    def __init__(self, d_model=16, hidden_dim=32, num_experts=4, top_k=2):
        super().__init__()
        if d_model <= 0 or hidden_dim <= 0 or num_experts <= 0:
            raise ValueError("model dimensions and num_experts must be positive")
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Expert(d_model, hidden_dim) for _ in range(num_experts)]
        )

    def forward(self, x):
        if x.ndim < 2 or x.size(-1) != self.router.in_features:
            raise ValueError("MoE input must have shape [...,d_model]")
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


def router_diagnostics(moe, x):
    if x.ndim < 2 or x.size(-1) != moe.router.in_features:
        raise ValueError("MoE input must have shape [...,d_model]")
    tokens = x.reshape(-1, x.size(-1))
    probabilities = torch.softmax(moe.router(tokens).float(), dim=-1)
    top_weights, top_indices = probabilities.topk(moe.top_k, dim=-1)
    normalized_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
    entropy = -(
        probabilities * probabilities.clamp_min(1e-9).log()
    ).sum(dim=-1).mean()
    top1_counts = torch.bincount(
        top_indices[:, 0],
        minlength=moe.num_experts,
    )
    return probabilities, normalized_weights, top_indices, entropy, top1_counts


def run_demo():
    torch.manual_seed(0)
    x = torch.randn(2, 6, 16)
    moe = TopKMoE(num_experts=4, top_k=2)
    output, auxiliary_loss, routes, counts = moe(x)

    assert output.shape == x.shape
    assert routes.shape == (2, 6, 2)
    assert sum(counts) == 2 * 6 * 2
    assert torch.isfinite(auxiliary_loss)

    probabilities, top_weights, _, entropy, top1_counts = router_diagnostics(moe, x)
    assert torch.allclose(top_weights.sum(-1), torch.ones(top_weights.size(0)))
    total_loss = output.square().mean() + 0.01 * auxiliary_loss
    total_loss.backward()
    assert moe.router.weight.grad is not None
    assert torch.isfinite(moe.router.weight.grad).all()

    print("input/output:", x.shape, output.shape)
    print("top-2 routes for batch 0:\n", routes[0])
    print("expert assignments:", counts)
    print("load-balance auxiliary loss:", auxiliary_loss.item())
    print("uniform balance-loss reference: 1.0")
    print("router mean entropy:", entropy.item(), "max:", torch.log(torch.tensor(4.0)).item())
    print("top-1 expert counts:", top1_counts.tolist())
    print("router gradient norm:", moe.router.weight.grad.norm().item())
    print("router probability row sums:", probabilities.sum(-1)[:3].tolist())


if __name__ == "__main__":
    run_demo()
