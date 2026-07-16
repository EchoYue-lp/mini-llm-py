#!/usr/bin/env python3
"""Run the fine-tuned tool router and print its JSON response."""

import argparse
from pathlib import Path

from mlx_lm import generate, load


PROJECT_ROOT = Path(__file__).resolve().parent
SYSTEM = (
    "你是工具路由模型。只输出一个JSON对象，不要解释。"
    "字段为action、intent、tool、arguments、missing_arguments。"
    "action只能是call_tool、ask_clarification或no_tool。"
    "工具签名：weather_query(city,date)、logistics_query(order_id)、"
    "order_cancel(order_id)、refund_query(refund_id)。"
    "缺少参数时tool必须为null，arguments必须为空对象，"
    "missing_arguments只能使用city、date、order_id、refund_id、"
    "single_intent、confirmed_intent或intent。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", default="帮我查一下明天上海的天气")
    parser.add_argument(
        "--adapter",
        default="adapters/tool-router-short",
        help="Adapter path relative to the project root",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer = load(
        str(PROJECT_ROOT / "models" / "Qwen3-0.6B"),
        adapter_path=str(PROJECT_ROOT / args.adapter),
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
