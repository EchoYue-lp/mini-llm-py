"""Lab 00: sinusoidal, learned, and rotary position representations."""

import math

import torch
import torch.nn as nn

from labs.lab07_modern_blocks import apply_rope


def sinusoidal_position_encoding(seq_len, d_model, device="cpu"):
    """Return a fixed position table with shape ``[seq_len, d_model]``."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if d_model <= 0:
        raise ValueError("d_model must be positive")
    positions = torch.arange(seq_len, device=device).float().unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, d_model, 2, device=device).float()
        * (-math.log(10000.0) / d_model)
    )
    encoding = torch.zeros(seq_len, d_model, device=device)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(
        positions * frequencies[: encoding[:, 1::2].shape[1]]
    )
    return encoding


def shift_sinusoidal_pair(pair, delta_angle):
    """Move one [sin(angle), cos(angle)] pair by ``delta_angle``."""
    sine, cosine = pair
    delta_sine = torch.sin(torch.as_tensor(delta_angle, device=pair.device))
    delta_cosine = torch.cos(torch.as_tensor(delta_angle, device=pair.device))
    return torch.stack(
        (
            sine * delta_cosine + cosine * delta_sine,
            cosine * delta_cosine - sine * delta_sine,
        )
    )


class LearnedPositionEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        if max_len <= 0 or d_model <= 0:
            raise ValueError("max_len and d_model must be positive")
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, token_embeddings):
        if token_embeddings.ndim != 3:
            raise ValueError("token_embeddings must have shape [B,T,D]")
        if token_embeddings.size(-1) != self.embedding.embedding_dim:
            raise ValueError("input feature width must match d_model")
        if token_embeddings.size(1) > self.embedding.num_embeddings:
            raise ValueError("sequence length exceeds learned position table")
        positions = torch.arange(
            token_embeddings.size(1), device=token_embeddings.device
        )
        return token_embeddings + self.embedding(positions).unsqueeze(0)


def run_demo():
    torch.manual_seed(0)
    tokens = torch.randn(2, 6, 8)
    sinusoidal = sinusoidal_position_encoding(6, 8)
    with_sinusoidal = tokens + sinusoidal.unsqueeze(0)
    learned = LearnedPositionEmbedding(16, 8)
    with_learned = learned(tokens)

    q_or_k = torch.randn(2, 4, 6, 8)
    with_rope = apply_rope(q_or_k)

    assert sinusoidal.shape == (6, 8)
    assert with_sinusoidal.shape == with_learned.shape == tokens.shape
    assert with_rope.shape == q_or_k.shape
    assert torch.allclose(sinusoidal[0, 0::2], torch.zeros(4))
    assert torch.allclose(sinusoidal[0, 1::2], torch.ones(4))
    # The first frequency is 1 radian per position. Its pair at position 3 can
    # be derived from position 1 by a rotation of 2 radians.
    shifted_pair = shift_sinusoidal_pair(sinusoidal[1, :2], delta_angle=2.0)
    assert torch.allclose(shifted_pair, sinusoidal[3, :2], atol=1e-6)
    assert sum(parameter.numel() for parameter in learned.parameters()) == 16 * 8
    odd_encoding = sinusoidal_position_encoding(4, 7)
    assert odd_encoding.shape == (4, 7)

    print("sinusoidal PE:", sinusoidal.shape, "fixed, added to token embeddings")
    print(
        "learned PE output:",
        with_learned.shape,
        "trainable parameters:",
        16 * 8,
    )
    print("RoPE output:", with_rope.shape, "rotates Q/K instead of adding to x")
    print("sin/cos pair shift check: position 1 + delta 2 == position 3")
    print("odd d_model position table:", odd_encoding.shape)


if __name__ == "__main__":
    run_demo()
