"""Foundation F01: NumPy arrays as a bridge to tensor programming.

Exercises after running this module:
1. Change ``num_heads`` from 2 to 4 and update the expected head dimension.
2. Use a boolean mask to select only positive values from an array.
3. Compare ``reshape`` with ``transpose`` and explain why they are different.
"""

from __future__ import annotations

import numpy as np


def stable_softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax along one axis while keeping dimensions for broadcast."""
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=axis, keepdims=True)


def linear_projection(
    inputs: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    """Match PyTorch Linear convention: weight is stored as [out, in]."""
    if inputs.shape[-1] != weight.shape[-1]:
        raise ValueError("input feature size must match weight input size")
    output = inputs @ weight.T
    return output if bias is None else output + bias


def split_heads(hidden: np.ndarray, num_heads: int) -> np.ndarray:
    """Convert [B,T,D] to [B,H,T,Dh]."""
    if hidden.ndim != 3:
        raise ValueError("hidden must have shape [B,T,D]")
    batch_size, seq_len, d_model = hidden.shape
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    head_dim = d_model // num_heads
    return hidden.reshape(batch_size, seq_len, num_heads, head_dim).transpose(
        0, 2, 1, 3
    )


def merge_heads(heads: np.ndarray) -> np.ndarray:
    """Convert [B,H,T,Dh] back to a contiguous [B,T,D] array."""
    if heads.ndim != 4:
        raise ValueError("heads must have shape [B,H,T,Dh]")
    batch_size, num_heads, seq_len, head_dim = heads.shape
    return heads.transpose(0, 2, 1, 3).copy().reshape(
        batch_size, seq_len, num_heads * head_dim
    )


def run_demo() -> None:
    hidden = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    feature_scale = np.array([1.0, 0.1, 10.0, -1.0], dtype=np.float32)
    broadcast_output = hidden * feature_scale
    assert broadcast_output.shape == hidden.shape

    weight = np.arange(5 * 4, dtype=np.float32).reshape(5, 4) / 10
    bias = np.arange(5, dtype=np.float32)
    projected = linear_projection(hidden, weight, bias)
    assert projected.shape == (2, 3, 5)

    heads = split_heads(hidden, num_heads=2)
    restored = merge_heads(heads)
    assert heads.shape == (2, 2, 3, 2)
    assert np.array_equal(restored, hidden)

    logits = np.array([[1000.0, 999.0, 998.0]], dtype=np.float64)
    probabilities = stable_softmax(logits)
    assert np.allclose(probabilities.sum(axis=-1), np.ones(1))

    padding_ids = np.array([[5, 7, 0], [4, 0, 0]])
    padding_mask = padding_ids != 0
    assert padding_mask.dtype == np.bool_

    print("hidden:", hidden.shape, hidden.dtype)
    print("broadcast scale:", feature_scale.shape, "->", broadcast_output.shape)
    print("linear projection: [2,3,4] @ [5,4].T ->", projected.shape)
    print("split heads:", heads.shape, "merge heads:", restored.shape)
    print("stable softmax:", probabilities.round(4).tolist())
    print("padding mask:\n", padding_mask)


if __name__ == "__main__":
    run_demo()
