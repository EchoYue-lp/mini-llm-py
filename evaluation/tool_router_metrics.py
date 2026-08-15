"""Dependency-free parsing and validation helpers for tool-router evaluation."""

import json
from typing import Any

from evaluation.validate_tool_router_data import validate_answer


def parse_raw_json(text: str) -> dict[str, Any] | None:
    """Parse the entire response as one JSON object."""
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort extraction used only as a separate diagnostic metric."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def is_schema_valid(value: dict[str, Any] | None) -> bool:
    if value is None:
        return False
    try:
        validate_answer(value, "model output")
    except ValueError:
        return False
    return True


def normalized(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalized(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return sorted(normalized(item) for item in value)
    return value
