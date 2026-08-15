#!/usr/bin/env python3
"""Run the fine-tuned tool router and print its JSON response."""

import argparse

from mlx_lm import generate, load

from utils.project_paths import (
    MLX_MODEL_DIR,
    TOOL_ROUTER_SHORT_ADAPTER_DIR,
    resolve_project_path,
)
from utils.tool_router_schema import SYSTEM_PROMPT

SYSTEM = SYSTEM_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", default="帮我查一下明天上海的天气")
    parser.add_argument(
        "--adapter",
        default=str(TOOL_ROUTER_SHORT_ADAPTER_DIR),
        help="Adapter path, absolute or relative to the project root",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer = load(
        str(MLX_MODEL_DIR),
        adapter_path=str(resolve_project_path(args.adapter)),
    )
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": args.text},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    print(
        generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            verbose=False,
        ).strip()
    )


if __name__ == "__main__":
    main()
