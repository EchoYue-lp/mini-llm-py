"""Lab 04: train a tiny encoder-decoder Transformer to copy token sequences."""

import argparse

import torch
import torch.nn.functional as F

from models.transformer_models import EncoderDecoderModel
from utils.mask_utils import create_causal_mask


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2


def make_copy_batch(batch_size, content_length, vocab_size, device="cpu"):
    content = torch.randint(
        3, vocab_size, (batch_size, content_length), device=device
    )
    bos = torch.full((batch_size, 1), BOS_ID, device=device)
    eos = torch.full((batch_size, 1), EOS_ID, device=device)
    target = torch.cat([bos, content, eos], dim=1)
    return content, target[:, :-1], target[:, 1:]


def greedy_copy(model, source, max_new_tokens):
    generated = torch.tensor([[BOS_ID]], device=source.device)
    for _ in range(max_new_tokens):
        mask = create_causal_mask(generated.size(1), device=source.device)
        logits, _ = model(source, generated, tgt_mask=mask)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if next_token.item() == EOS_ID:
            break
    return generated[0, 1:].tolist()


def train_copy_task(steps=400, device="cpu"):
    torch.manual_seed(7)
    vocab_size = 16
    model = EncoderDecoderModel(
        vocab_size,
        vocab_size,
        d_model=32,
        num_layers=1,
        num_heads=4,
        d_ff=64,
        max_len=16,
        dropout=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []

    model.train()
    for step in range(1, steps + 1):
        source, decoder_input, labels = make_copy_batch(
            batch_size=32,
            content_length=5,
            vocab_size=vocab_size,
            device=device,
        )
        target_mask = create_causal_mask(decoder_input.size(1), device=device)
        logits, _ = model(source, decoder_input, tgt_mask=target_mask)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step == 1 or step % max(1, steps // 4) == 0:
            print(f"step={step:3d} loss={loss.item():.4f}")

    model.eval()
    example = torch.tensor([[4, 7, 9, 5, 3]], device=device)
    prediction = greedy_copy(model, example, max_new_tokens=6)
    print("source:    ", example[0].tolist())
    print("prediction:", prediction)
    return model, losses, prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train_copy_task(steps=args.steps, device=args.device)


if __name__ == "__main__":
    main()
