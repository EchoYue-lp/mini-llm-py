import pytest

from evaluation.validate_tool_router_data import validate_answer


def answer(**overrides):
    value = {
        "action": "call_tool",
        "intent": "query_weather",
        "tool": "weather_query",
        "arguments": {"city": "上海", "date": "明天"},
        "missing_arguments": [],
    }
    value.update(overrides)
    return value


def test_call_tool_requires_every_declared_argument():
    with pytest.raises(ValueError, match="missing tool arguments"):
        validate_answer(
            answer(arguments={"city": "上海"}),
            "test:1",
        )


def test_call_tool_rejects_unknown_arguments():
    with pytest.raises(ValueError, match="unknown tool arguments"):
        validate_answer(
            answer(
                arguments={
                    "city": "上海",
                    "date": "明天",
                    "units": "celsius",
                }
            ),
            "test:2",
        )


def test_valid_call_tool_passes():
    validate_answer(answer(), "test:3")


def test_no_tool_cannot_carry_tool_state():
    with pytest.raises(ValueError, match="requires tool=null"):
        validate_answer(
            answer(
                action="no_tool",
                intent="chitchat",
                tool="weather_query",
                arguments={},
            ),
            "test:4",
        )


def test_clarification_requires_a_missing_reason():
    with pytest.raises(ValueError, match="requires missing arguments"):
        validate_answer(
            answer(
                action="ask_clarification",
                intent="query_weather",
                tool=None,
                arguments={},
                missing_arguments=[],
            ),
            "test:5",
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"intent": ""}, "intent must be a non-empty string"),
        ({"tool": 7}, "tool must be a string or null"),
        (
            {"arguments": {"city": "", "date": "明天"}},
            "argument values must be non-empty strings",
        ),
        (
            {"missing_arguments": ["city", "city"]},
            "cannot contain duplicates",
        ),
        (
            {"missing_arguments": ["city", 7]},
            "entries must be strings",
        ),
    ],
)
def test_field_types_and_values_are_validated(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_answer(answer(**overrides), "test:types")


def test_answer_must_be_an_object():
    with pytest.raises(ValueError, match="must be a JSON object"):
        validate_answer([], "test:object")
