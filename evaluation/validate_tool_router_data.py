#!/usr/bin/env python3
"""Validate chat records, tool schemas, labels, and cross-split duplicates."""

import json
from collections import Counter
from typing import Any

from utils.project_paths import TOOL_ROUTER_DATA_DIR
from utils.tool_router_schema import (
    ACTIONS,
    AMBIGUOUS_INTENTS,
    INTENTS,
    INTENT_TO_TOOL,
    MISSING_ARGUMENTS,
    NO_TOOL_INTENTS,
    SYSTEM_PROMPT,
    TOOL_ARGUMENTS,
)

SPLITS = ("train", "valid", "test")
FIELDS = {"action", "intent", "tool", "arguments", "missing_arguments"}


def fail(location: str, message: str) -> None:
    raise ValueError(f"{location}: {message}")


def validate_answer(answer: dict[str, Any], location: str) -> None:
    if not isinstance(answer, dict):
        fail(location, "assistant answer must be a JSON object")
    if set(answer) != FIELDS:
        fail(location, f"fields must be exactly {sorted(FIELDS)}")
    if not isinstance(answer["action"], str):
        fail(location, "action must be a string")
    if answer["action"] not in ACTIONS:
        fail(location, f"invalid action {answer['action']!r}")
    if not isinstance(answer["intent"], str) or not answer["intent"].strip():
        fail(location, "intent must be a non-empty string")
    if answer["intent"] not in INTENTS:
        fail(location, f"unknown intent {answer['intent']!r}")
    if answer["tool"] is not None and not isinstance(answer["tool"], str):
        fail(location, "tool must be a string or null")
    if not isinstance(answer["arguments"], dict):
        fail(location, "arguments must be an object")
    invalid_argument_values = {
        key: value
        for key, value in answer["arguments"].items()
        if not isinstance(value, str) or not value.strip()
    }
    if invalid_argument_values:
        fail(location, "tool argument values must be non-empty strings")
    if not isinstance(answer["missing_arguments"], list):
        fail(location, "missing_arguments must be a list")
    if any(not isinstance(item, str) for item in answer["missing_arguments"]):
        fail(location, "missing_arguments entries must be strings")
    if len(answer["missing_arguments"]) != len(set(answer["missing_arguments"])):
        fail(location, "missing_arguments cannot contain duplicates")
    unknown_missing = set(answer["missing_arguments"]) - MISSING_ARGUMENTS
    if unknown_missing:
        fail(location, f"unknown missing arguments: {sorted(unknown_missing)}")

    action = answer["action"]
    intent = answer["intent"]
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
        expected_tool = INTENT_TO_TOOL.get(intent)
        if tool != expected_tool:
            fail(location, f"intent {intent!r} requires tool {expected_tool!r}")
    else:
        if tool is not None:
            fail(location, f"{action} requires tool=null")
        if answer["arguments"]:
            fail(location, f"{action} requires empty arguments")
    if action == "ask_clarification" and not answer["missing_arguments"]:
        fail(location, "ask_clarification requires missing arguments")
    if action == "no_tool" and answer["missing_arguments"]:
        fail(location, "no_tool cannot contain missing arguments")
    if action == "no_tool" and intent not in NO_TOOL_INTENTS:
        fail(location, f"intent {intent!r} cannot use no_tool")
    if action == "ask_clarification":
        if intent in INTENT_TO_TOOL:
            allowed_missing = TOOL_ARGUMENTS[INTENT_TO_TOOL[intent]]
        else:
            allowed_missing = AMBIGUOUS_INTENTS.get(intent, frozenset())
        if not set(answer["missing_arguments"]) <= allowed_missing:
            fail(location, f"invalid missing arguments for intent {intent!r}")


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
                if not isinstance(record, dict):
                    fail(location, "record must be a JSON object")
                messages = record.get("messages")
                if not isinstance(messages, list) or len(messages) != 3:
                    fail(location, "messages must contain system, user, assistant")
                if any(not isinstance(message, dict) for message in messages):
                    fail(location, "every message must be an object")
                if [message.get("role") for message in messages] != [
                    "system",
                    "user",
                    "assistant",
                ]:
                    fail(location, "unexpected chat roles")
                for message in messages:
                    content = message.get("content")
                    if not isinstance(content, str) or not content.strip():
                        fail(location, f"{message['role']} content must be non-empty")
                if messages[0]["content"] != SYSTEM_PROMPT:
                    fail(location, "system prompt does not match the task schema")
                user_text = messages[1].get("content")
                normalized_user_text = user_text.strip()
                if normalized_user_text in seen:
                    fail(
                        location,
                        "duplicate user text also found in "
                        f"{seen[normalized_user_text]}",
                    )
                seen[normalized_user_text] = location
                try:
                    answer = json.loads(messages[2]["content"])
                except json.JSONDecodeError as error:
                    fail(location, f"assistant content is not valid JSON: {error.msg}")
                validate_answer(answer, location)
                action_counts[answer["action"]] += 1
                total += 1

    print(f"Validated {total} records across {', '.join(SPLITS)}")
    for action in sorted(ACTIONS):
        print(f"{action}: {action_counts[action]}")


if __name__ == "__main__":
    main()
