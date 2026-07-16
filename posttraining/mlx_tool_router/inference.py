#!/usr/bin/env python3
"""Run a short Qwen3-0.6B chat completion with MLX."""

import argparse
from pathlib import Path

from mlx_lm import generate, load


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "Qwen3-0.6B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument(
        "--prompt",
        default="请用三句话说明什么是 LoRA 微调。",
    )
    parser.add_argument("--max-tokens", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer = load(args.model)
    messages = [
        {"role": "system", "content": "你是一个简洁、准确的中文助手。"},
        {"role": "user", "content": args.prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=args.max_tokens,
        verbose=False,
    )
    print(response.strip())


if __name__ == "__main__":
    main()
