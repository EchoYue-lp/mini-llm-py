"""Lab 00: sinusoidal, learned, and rotary position representations."""

import math

import torch
import torch.nn as nn

from labs.lab07_modern_blocks import apply_rope


def sinusoidal_position_encoding(seq_len, d_model, device="cpu"):
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


class LearnedPositionEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, token_embeddings):
        positions = torch.arange(
            token_embeddings.size(1), device=token_embeddings.device
        )
        return token_embeddings + self.embedding(positions).unsqueeze(0)


def run_demo():
    torch.manual_seed(0)
    tokens = torch.randn(2, 6, 8)
    sinusoidal = sinusoidal_position_encoding(6, 8)
    with_sinusoidal = tokens + sinusoidal.unsqueeze(0)
    with_learned = LearnedPositionEmbedding(16, 8)(tokens)

    q_or_k = torch.randn(2, 4, 6, 8)
    with_rope = apply_rope(q_or_k)

    assert sinusoidal.shape == (6, 8)
    assert with_sinusoidal.shape == with_learned.shape == tokens.shape
    assert with_rope.shape == q_or_k.shape
    assert torch.allclose(sinusoidal[0, 0::2], torch.zeros(4))
    assert torch.allclose(sinusoidal[0, 1::2], torch.ones(4))

    print("sinusoidal PE:", sinusoidal.shape, "fixed, added to token embeddings")
    print("learned PE output:", with_learned.shape, "trainable lookup table")
    print("RoPE output:", with_rope.shape, "rotates Q/K instead of adding to x")


if __name__ == "__main__":
    run_demo()
