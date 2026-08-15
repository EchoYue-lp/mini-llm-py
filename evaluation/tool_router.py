#!/usr/bin/env python3
"""Evaluate base or adapted models on the same deterministic JSONL test set."""

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from typing import Any

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from evaluation.tool_router_metrics import (
    extract_json,
    is_schema_valid,
    normalized,
    parse_raw_json,
)

from utils.project_paths import (
    MLX_MODEL_DIR,
    MLX_MODEL_MANIFEST,
    PROJECT_ROOT,
    TOOL_ROUTER_DATA_DIR,
    resolve_project_path,
)

TEST_PATH = TOOL_ROUTER_DATA_DIR / "test.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", help="Adapter directory; omit for base model")
    parser.add_argument("--label", default="model")
    parser.add_argument("--output", help="Optional JSON report path")
    parser.add_argument("--max-tokens", type=int, default=128)
    return parser.parse_args()


def load_rows() -> list[dict[str, Any]]:
    with TEST_PATH.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    adapter = str(resolve_project_path(args.adapter)) if args.adapter else None
    model, tokenizer = load(str(MLX_MODEL_DIR), adapter_path=adapter)
    sampler = make_sampler(temp=0.0)
    rows = load_rows()
    fields = ["action", "intent", "tool", "arguments", "missing_arguments"]
    totals = {
        "raw_json_valid": 0,
        "extractable_json": 0,
        "schema_valid": 0,
        "exact_match": 0,
        **{field: 0 for field in fields},
    }
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
        raw_actual = parse_raw_json(raw)
        if raw_actual is not None:
            totals["raw_json_valid"] += 1
        actual = raw_actual if raw_actual is not None else extract_json(raw)
        if actual is not None:
            totals["extractable_json"] += 1
        if is_schema_valid(actual):
            totals["schema_valid"] += 1
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
        "model": str(MLX_MODEL_DIR.relative_to(PROJECT_ROOT)),
        "adapter": args.adapter,
        "samples": count,
        "provenance": {
            "model_manifest": (
                json.loads(MLX_MODEL_MANIFEST.read_text(encoding="utf-8"))
                if MLX_MODEL_MANIFEST.exists()
                else None
            ),
            "test_sha256": sha256_file(TEST_PATH),
            "python": platform.python_version(),
            "mlx_lm": version("mlx-lm"),
            "generation": {
                "temperature": 0.0,
                "max_tokens": args.max_tokens,
                "enable_thinking": False,
            },
        },
        "metrics": metrics,
        "predictions": predictions,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        output = resolve_project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
