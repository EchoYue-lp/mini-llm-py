"""Lab 04: train a tiny encoder-decoder Transformer to copy token sequences."""

import argparse
import math

import torch
import torch.nn.functional as F

from models.transformer_models import EncoderDecoderModel
from utils.mask_utils import create_causal_mask


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2


def make_copy_batch(batch_size, content_length, vocab_size, device="cpu"):
    """Return source, decoder input, and next-token labels for a copy task."""
    if batch_size <= 0 or content_length <= 0:
        raise ValueError("batch_size and content_length must be positive")
    if vocab_size <= EOS_ID + 1:
        raise ValueError("vocab_size must leave room for PAD/BOS/EOS and content")
    content = torch.randint(
        3, vocab_size, (batch_size, content_length), device=device
    )
    bos = torch.full((batch_size, 1), BOS_ID, device=device)
    eos = torch.full((batch_size, 1), EOS_ID, device=device)
    target = torch.cat([bos, content, eos], dim=1)
    return content, target[:, :-1], target[:, 1:]


def greedy_copy(model, source, max_new_tokens):
    """Greedily decode one source sequence, excluding the initial BOS in output."""
    if source.ndim != 2 or source.size(0) != 1:
        raise ValueError("greedy_copy currently expects source shape [1,S]")
    if source.dtype != torch.long:
        raise TypeError("source token ids must use torch.long")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    max_target_len = model.tgt_pos_enc.pe.size(1)
    if 1 + max_new_tokens > max_target_len:
        raise ValueError("requested generation exceeds target position table")

    generated = torch.full(
        (1, 1),
        BOS_ID,
        dtype=source.dtype,
        device=source.device,
    )
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                mask = create_causal_mask(generated.size(1), device=source.device)
                logits, _ = model(source, generated, tgt_mask=mask)
                next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.item() == EOS_ID:
                    break
    finally:
        model.train(was_training)
    return generated[0, 1:].tolist()


def train_copy_task(steps=400, device="cpu"):
    if steps <= 0:
        raise ValueError("steps must be positive")
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
    random_guess_loss = math.log(vocab_size)

    model.train()
    for step in range(1, steps + 1):
        source, decoder_input, labels = make_copy_batch(
            batch_size=32,
            content_length=5,
            vocab_size=vocab_size,
            device=device,
        )
        target_mask = create_causal_mask(decoder_input.size(1), device=device)
        assert decoder_input.shape == labels.shape
        assert torch.equal(decoder_input[:, 1:], labels[:, :-1])
        assert torch.count_nonzero(target_mask[0, 0].triu(1)) == 0
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
    print("random-uniform CE baseline log(V):", round(random_guess_loss, 4))
    print("first/final training loss:", round(losses[0], 4), round(losses[-1], 4))
    print("decoder input and labels are the same target shifted by one token")
    return model, losses, prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train_copy_task(steps=args.steps, device=args.device)


if __name__ == "__main__":
    main()
