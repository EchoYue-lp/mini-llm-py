"""Foundation F04: PyTorch tensor shapes, dtypes, devices, and masks.

Exercises after running this module:
1. Move the demo tensors to CUDA or MPS when available.
2. Create a fully masked row and inspect the output of ``masked_softmax``.
3. Deliberately omit ``contiguous`` before ``view`` and inspect the error.
"""

from __future__ import annotations

import numpy as np
import torch


def preferred_device() -> torch.device:
    """Choose an accelerator when one is available, otherwise use CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def linear_projection(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a PyTorch-style [out,in] weight to [...,in] inputs."""
    if inputs.size(-1) != weight.size(-1):
        raise ValueError("input feature size must match weight input size")
    output = inputs @ weight.transpose(-2, -1)
    return output if bias is None else output + bias


def split_heads(hidden: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Convert [B,T,D] to [B,H,T,Dh]."""
    if hidden.ndim != 3:
        raise ValueError("hidden must have shape [B,T,D]")
    batch_size, seq_len, d_model = hidden.shape
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    head_dim = d_model // num_heads
    return hidden.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)


def merge_heads(heads: torch.Tensor) -> torch.Tensor:
    """Convert [B,H,T,Dh] back to contiguous [B,T,D]."""
    if heads.ndim != 4:
        raise ValueError("heads must have shape [B,H,T,Dh]")
    batch_size, num_heads, seq_len, head_dim = heads.shape
    return (
        heads.transpose(1, 2)
        .contiguous()
        .view(batch_size, seq_len, num_heads * head_dim)
    )


def masked_softmax(
    scores: torch.Tensor,
    visible: torch.Tensor,
) -> torch.Tensor:
    """Normalize visible keys and define fully masked rows as all zeros."""
    if visible.dtype != torch.bool:
        raise TypeError("visible mask must have boolean dtype")
    masked_scores = scores.masked_fill(~visible, float("-inf"))
    probabilities = torch.softmax(masked_scores, dim=-1)
    return torch.nan_to_num(probabilities, nan=0.0)


def run_demo() -> None:
    hidden = torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4)
    feature_scale = torch.tensor([1.0, 0.1, 10.0, -1.0])
    broadcast_output = hidden * feature_scale
    assert broadcast_output.shape == hidden.shape

    weight = torch.arange(5 * 4, dtype=torch.float32).view(5, 4) / 10
    bias = torch.arange(5, dtype=torch.float32)
    projected = linear_projection(hidden, weight, bias)
    assert projected.shape == (2, 3, 5)

    heads = split_heads(hidden, num_heads=2)
    restored = merge_heads(heads)
    assert heads.shape == (2, 2, 3, 2)
    assert torch.equal(restored, hidden)

    scores = torch.randn(1, 2, 3, 3)
    causal = torch.tril(torch.ones(3, 3, dtype=torch.bool)).view(1, 1, 3, 3)
    probabilities = masked_softmax(scores, causal)
    assert torch.allclose(probabilities.sum(-1), torch.ones(1, 2, 3))
    assert torch.count_nonzero(probabilities.triu(1)) == 0

    array = np.arange(4, dtype=np.float32)
    shared_tensor = torch.from_numpy(array)
    shared_tensor[0] = 99
    assert array[0] == 99
    copied_tensor = torch.tensor(array)
    copied_tensor[1] = -1
    assert array[1] != -1

    print("hidden:", hidden.shape, hidden.dtype, hidden.device)
    print("preferred device:", preferred_device())
    print("broadcast output:", broadcast_output.shape)
    print("linear projection:", projected.shape)
    print("split heads:", heads.shape, "merge heads:", restored.shape)
    print("causal attention row sums:\n", probabilities.sum(-1))
    print("torch.from_numpy shares memory; torch.tensor copies data")


if __name__ == "__main__":
    run_demo()
