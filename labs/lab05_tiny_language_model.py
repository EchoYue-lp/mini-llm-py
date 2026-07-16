"""Lab 05: train a decoder-only Transformer on a repeating token rule."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.transformer_models import DecoderOnlyModel
from utils.mask_utils import create_causal_mask


def make_pattern_batch(batch_size, seq_len, vocab_size, device="cpu"):
    starts = torch.randint(0, vocab_size, (batch_size, 1), device=device)
    offsets = torch.arange(seq_len + 1, device=device).unsqueeze(0)
    sequence = (starts + offsets) % vocab_size
    return sequence[:, :-1], sequence[:, 1:]


def generate(model, prompt, new_tokens):
    tokens = prompt.clone()
    for _ in range(new_tokens):
        mask = create_causal_mask(tokens.size(1), device=tokens.device)
        logits, _ = model(tokens, mask)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokens


def train_language_model(steps=100, device="cpu"):
    torch.manual_seed(11)
    vocab_size = 12
    model = DecoderOnlyModel(
        vocab_size,
        d_model=32,
        num_layers=1,
        num_heads=4,
        d_ff=64,
        max_len=24,
        dropout=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []

    model.train()
    for step in range(1, steps + 1):
        inputs, labels = make_pattern_batch(32, 8, vocab_size, device)
        mask = create_causal_mask(inputs.size(1), device=device)
        logits, _ = model(inputs, mask)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step == 1 or step % max(1, steps // 4) == 0:
            print(f"step={step:3d} loss={loss.item():.4f}")

    model.eval()
    prompt = torch.tensor([[3, 4, 5]], device=device)
    result = generate(model, prompt, new_tokens=6)
    print("generated:", result[0].tolist())
    return model, losses, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train_language_model(steps=args.steps, device=args.device)


if __name__ == "__main__":
    main()
