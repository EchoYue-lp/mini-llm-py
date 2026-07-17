"""Lab 11: dense MoE, sparse MoE, and shared-expert sparse MoE."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, d_model=16, hidden_dim=32):
        super().__init__()
        if d_model <= 0 or hidden_dim <= 0:
            raise ValueError("d_model and hidden_dim must be positive")
        self.up = nn.Linear(d_model, hidden_dim)
        self.down = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class DenseMoE(nn.Module):
    """Evaluate every expert and combine all outputs with router probabilities."""

    def __init__(self, d_model=16, hidden_dim=32, num_experts=4):
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Expert(d_model, hidden_dim) for _ in range(num_experts)]
        )

    def forward(self, x):
        if x.ndim < 2 or x.size(-1) != self.router.in_features:
            raise ValueError("Dense MoE input must have shape [...,d_model]")
        weights = torch.softmax(self.router(x), dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        return (expert_outputs * weights.unsqueeze(-1)).sum(dim=-2), weights


class SparseMoE(nn.Module):
    """Evaluate only the routed Top-k experts for each token."""

    def __init__(
        self,
        d_model=16,
        hidden_dim=32,
        num_experts=8,
        top_k=2,
        renormalize_topk=True,
    ):
        super().__init__()
        if d_model <= 0 or hidden_dim <= 0 or num_experts <= 0:
            raise ValueError("model dimensions and num_experts must be positive")
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        self.num_experts = num_experts
        self.top_k = top_k
        self.renormalize_topk = renormalize_topk
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Expert(d_model, hidden_dim) for _ in range(num_experts)]
        )

    def route(self, tokens):
        if tokens.ndim != 2 or tokens.size(-1) != self.router.in_features:
            raise ValueError("routed tokens must have shape [N,d_model]")
        probabilities = torch.softmax(self.router(tokens), dim=-1)
        top_weights, top_indices = probabilities.topk(self.top_k, dim=-1)
        if self.renormalize_topk:
            top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        return probabilities, top_weights, top_indices

    def forward(self, x):
        if x.ndim < 2 or x.size(-1) != self.router.in_features:
            raise ValueError("Sparse MoE input must have shape [...,d_model]")
        shape = x.shape
        tokens = x.reshape(-1, x.size(-1))
        probabilities, top_weights, top_indices = self.route(tokens)

        output = torch.zeros_like(tokens)
        counts = []
        for expert_id, expert in enumerate(self.experts):
            token_indices, choice_indices = torch.where(top_indices == expert_id)
            counts.append(int(token_indices.numel()))
            if token_indices.numel() == 0:
                continue
            selected = expert(tokens[token_indices])
            selected = selected * top_weights[
                token_indices, choice_indices
            ].unsqueeze(-1)
            output.index_add_(0, token_indices, selected)

        importance = probabilities.mean(dim=0)
        load = F.one_hot(
            top_indices[:, 0], num_classes=self.num_experts
        ).float().mean(dim=0)
        balance_loss = self.num_experts * (importance * load).sum()
        return output.view(shape), top_indices.view(*shape[:-1], self.top_k), counts, balance_loss


class SharedExpertSparseMoE(nn.Module):
    """Always-on shared experts plus Top-k routed experts.

    This is a small teaching implementation of shared-expert isolation.  A
    Transformer block would add the residual connection outside this module.
    """

    def __init__(
        self,
        d_model=16,
        hidden_dim=32,
        num_shared_experts=1,
        num_routed_experts=8,
        top_k=2,
    ):
        super().__init__()
        if num_shared_experts <= 0 or num_routed_experts <= 0:
            raise ValueError("shared and routed expert counts must be positive")
        self.shared_experts = nn.ModuleList(
            [Expert(d_model, hidden_dim) for _ in range(num_shared_experts)]
        )
        self.routed = SparseMoE(
            d_model=d_model,
            hidden_dim=hidden_dim,
            num_experts=num_routed_experts,
            top_k=top_k,
            renormalize_topk=False,
        )

    def forward(self, x):
        shared_output = sum(expert(x) for expert in self.shared_experts)
        routed_output, routes, counts, balance_loss = self.routed(x)
        return shared_output + routed_output, routes, counts, balance_loss


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def run_demo():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)

    dense = DenseMoE(num_experts=4)
    sparse = SparseMoE(
        num_experts=8, top_k=2
    )
    shared = SharedExpertSparseMoE(
        num_shared_experts=1,
        num_routed_experts=8,
        top_k=2,
    )

    dense_output, dense_weights = dense(x)
    sparse_output, routes, counts, balance_loss = sparse(x)
    shared_output, shared_routes, shared_counts, shared_balance = shared(x)

    assert dense_output.shape == sparse_output.shape == shared_output.shape == x.shape
    assert dense_weights.shape == (2, 5, 4)
    assert routes.shape == shared_routes.shape == (2, 5, 2)
    assert sum(counts) == sum(shared_counts) == 2 * 5 * 2
    assert torch.allclose(dense_weights.sum(-1), torch.ones(2, 5))

    tokens = x.reshape(-1, x.size(-1))
    _, sparse_weights, _ = sparse.route(tokens)
    _, shared_routed_weights, _ = shared.routed.route(tokens)
    assert torch.allclose(sparse_weights.sum(-1), torch.ones(tokens.size(0)))
    assert torch.all(shared_routed_weights.sum(-1) <= 1 + 1e-6)

    (sparse_output.square().mean() + 0.01 * balance_loss).backward()
    experts_with_tokens = sum(count > 0 for count in counts)
    experts_with_grad = sum(
        expert.up.weight.grad is not None for expert in sparse.experts
    )
    assert experts_with_grad == experts_with_tokens

    dense_parameters = parameter_count(dense)
    sparse_parameters = parameter_count(sparse)
    shared_parameters = parameter_count(shared)

    print("Dense MoE: all 4 experts run for every token; parameters:", dense_parameters)
    print("Sparse MoE routed counts:", counts)
    print("Sparse balance loss:", balance_loss.item())
    print("Sparse parameters/active experts per token:", sparse_parameters, 2)
    print("Sparse experts with token/gradient:", experts_with_tokens, experts_with_grad)
    print("Shared-expert Sparse MoE routed counts:", shared_counts)
    print("Shared parameters/active experts per token:", shared_parameters, 3)
    print("Shared expert count: 1, always active for all tokens")
    print("Sparse top-k weight sum:", sparse_weights.sum(-1)[:3].tolist())
    print(
        "Shared routed weight mass before adding shared expert:",
        shared_routed_weights.sum(-1)[:3].tolist(),
    )
    print("Shared variant balance loss:", shared_balance.item())


if __name__ == "__main__":
    run_demo()
