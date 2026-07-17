#!/usr/bin/env python3
"""Validate chat records, tool schemas, labels, and cross-split duplicates."""

import json
from collections import Counter
from typing import Any

from utils.project_paths import TOOL_ROUTER_DATA_DIR

SPLITS = ("train", "valid", "test")
FIELDS = {"action", "intent", "tool", "arguments", "missing_arguments"}
ACTIONS = {"call_tool", "ask_clarification", "no_tool"}
TOOL_ARGUMENTS = {
    "weather_query": {"city", "date"},
    "logistics_query": {"order_id"},
    "order_cancel": {"order_id"},
    "refund_query": {"refund_id"},
}
MISSING_ARGUMENTS = {
    "city",
    "date",
    "order_id",
    "refund_id",
    "single_intent",
    "confirmed_intent",
    "intent",
}


def fail(location: str, message: str) -> None:
    raise ValueError(f"{location}: {message}")


def validate_answer(answer: dict[str, Any], location: str) -> None:
    if set(answer) != FIELDS:
        fail(location, f"fields must be exactly {sorted(FIELDS)}")
    if answer["action"] not in ACTIONS:
        fail(location, f"invalid action {answer['action']!r}")
    if not isinstance(answer["arguments"], dict):
        fail(location, "arguments must be an object")
    if not isinstance(answer["missing_arguments"], list):
        fail(location, "missing_arguments must be a list")
    unknown_missing = set(answer["missing_arguments"]) - MISSING_ARGUMENTS
    if unknown_missing:
        fail(location, f"unknown missing arguments: {sorted(unknown_missing)}")

    action = answer["action"]
    tool = answer["tool"]
    if action == "call_tool":
        if tool not in TOOL_ARGUMENTS:
            fail(location, f"unknown tool {tool!r}")
        required_arguments = TOOL_ARGUMENTS[tool]
        actual_arguments = set(answer["arguments"])
        unknown_arguments = actual_arguments - required_arguments
        if unknown_arguments:
            fail(location, f"unknown tool arguments: {sorted(unknown_arguments)}")
        missing_arguments = required_arguments - actual_arguments
        if missing_arguments:
            fail(location, f"missing tool arguments: {sorted(missing_arguments)}")
        if answer["missing_arguments"]:
            fail(location, "call_tool cannot contain missing arguments")
    else:
        if tool is not None:
            fail(location, f"{action} requires tool=null")
        if answer["arguments"]:
            fail(location, f"{action} requires empty arguments")
    if action == "ask_clarification" and not answer["missing_arguments"]:
        fail(location, "ask_clarification requires missing arguments")
    if action == "no_tool" and answer["missing_arguments"]:
        fail(location, "no_tool cannot contain missing arguments")


def main() -> None:
    seen: dict[str, str] = {}
    action_counts: Counter[str] = Counter()
    total = 0
    for split in SPLITS:
        path = TOOL_ROUTER_DATA_DIR / f"{split}.jsonl"
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                location = f"{path.name}:{line_number}"
                record = json.loads(line)
                messages = record.get("messages")
                if not isinstance(messages, list) or len(messages) != 3:
                    fail(location, "messages must contain system, user, assistant")
                if [message.get("role") for message in messages] != [
                    "system",
                    "user",
                    "assistant",
                ]:
                    fail(location, "unexpected chat roles")
                user_text = messages[1].get("content")
                if not isinstance(user_text, str) or not user_text.strip():
                    fail(location, "user content must be non-empty")
                if user_text in seen:
                    fail(location, f"duplicate user text also found in {seen[user_text]}")
                seen[user_text] = location
                answer = json.loads(messages[2]["content"])
                validate_answer(answer, location)
                action_counts[answer["action"]] += 1
                total += 1

    print(f"Validated {total} records across {', '.join(SPLITS)}")
    for action in sorted(ACTIONS):
        print(f"{action}: {action_counts[action]}")


if __name__ == "__main__":
    main()
