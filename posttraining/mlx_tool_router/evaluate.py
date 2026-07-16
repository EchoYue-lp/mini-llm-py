#!/usr/bin/env python3
"""Evaluate base or adapted models on the same deterministic JSONL test set."""

import argparse
import json
from pathlib import Path
from typing import Any

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "Qwen3-0.6B"
TEST_PATH = PROJECT_ROOT / "data" / "test.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", help="Adapter directory; omit for base model")
    parser.add_argument("--label", default="model")
    parser.add_argument("--output", help="Optional JSON report path")
    parser.add_argument("--max-tokens", type=int, default=128)
    return parser.parse_args()


def extract_json(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def load_rows() -> list[dict[str, Any]]:
    with TEST_PATH.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def normalized(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalized(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return sorted(normalized(item) for item in value)
    return value


def main() -> None:
    args = parse_args()
    adapter = str(PROJECT_ROOT / args.adapter) if args.adapter else None
    model, tokenizer = load(str(MODEL_PATH), adapter_path=adapter)
    sampler = make_sampler(temp=0.0)
    rows = load_rows()
    fields = ["action", "intent", "tool", "arguments", "missing_arguments"]
    totals = {"json_valid": 0, "exact_match": 0, **{field: 0 for field in fields}}
    predictions = []

    for row in rows:
        messages = row["messages"]
        expected = json.loads(messages[-1]["content"])
        prompt = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        raw = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            sampler=sampler,
            verbose=False,
        ).strip()
        actual = extract_json(raw)
        if actual is not None:
            totals["json_valid"] += 1
            for field in fields:
                if normalized(actual.get(field)) == normalized(expected.get(field)):
                    totals[field] += 1
            if normalized(actual) == normalized(expected):
                totals["exact_match"] += 1
        predictions.append(
            {
                "input": messages[-2]["content"],
                "expected": expected,
                "actual": actual,
                "raw": raw,
            }
        )

    count = len(rows)
    metrics = {key: round(value / count, 4) for key, value in totals.items()}
    report = {
        "label": args.label,
        "model": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "adapter": args.adapter,
        "samples": count,
        "metrics": metrics,
        "predictions": predictions,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        output = PROJECT_ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
