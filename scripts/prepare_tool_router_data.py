#!/usr/bin/env python3
"""Create a small chat-format dataset for a tool-routing LoRA demo."""

import json
import random
from pathlib import Path

from utils.project_paths import TOOL_ROUTER_DATA_DIR

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


def result(
    action: str,
    intent: str,
    tool: str | None = None,
    arguments: dict[str, str] | None = None,
    missing: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "action": action,
            "intent": intent,
            "tool": tool,
            "arguments": arguments or {},
            "missing_arguments": missing or [],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


EXAMPLES = [
    ("帮我查一下明天上海的天气", result("call_tool", "query_weather", "weather_query", {"city": "上海", "date": "明天"})),
    ("北京今天会下雨吗", result("call_tool", "query_weather", "weather_query", {"city": "北京", "date": "今天"})),
    ("后天深圳多少度", result("call_tool", "query_weather", "weather_query", {"city": "深圳", "date": "后天"})),
    ("查一下杭州周五的天气", result("call_tool", "query_weather", "weather_query", {"city": "杭州", "date": "周五"})),
    ("广州明早天气怎么样", result("call_tool", "query_weather", "weather_query", {"city": "广州", "date": "明早"})),
    ("明天天气怎么样", result("ask_clarification", "query_weather", missing=["city"])),
    ("帮我看看天气", result("ask_clarification", "query_weather", missing=["city", "date"])),
    ("上海天气怎么样", result("ask_clarification", "query_weather", missing=["date"])),
    ("订单A1024到哪里了", result("call_tool", "query_logistics", "logistics_query", {"order_id": "A1024"})),
    ("查下订单20260715001的物流", result("call_tool", "query_logistics", "logistics_query", {"order_id": "20260715001"})),
    ("我的B7788订单什么时候到", result("call_tool", "query_logistics", "logistics_query", {"order_id": "B7788"})),
    ("帮我追踪一下快递，订单号C9001", result("call_tool", "query_logistics", "logistics_query", {"order_id": "C9001"})),
    ("看一下D521订单配送到哪了", result("call_tool", "query_logistics", "logistics_query", {"order_id": "D521"})),
    ("我的订单到哪了", result("ask_clarification", "query_logistics", missing=["order_id"])),
    ("帮我查物流", result("ask_clarification", "query_logistics", missing=["order_id"])),
    ("快递怎么还没到", result("ask_clarification", "query_logistics", missing=["order_id"])),
    ("取消订单A1024", result("call_tool", "cancel_order", "order_cancel", {"order_id": "A1024"})),
    ("我不想要B7788这个订单了", result("call_tool", "cancel_order", "order_cancel", {"order_id": "B7788"})),
    ("撤销订单20260715001", result("call_tool", "cancel_order", "order_cancel", {"order_id": "20260715001"})),
    ("麻烦把C9001订单取消掉", result("call_tool", "cancel_order", "order_cancel", {"order_id": "C9001"})),
    ("订单D521不要了", result("call_tool", "cancel_order", "order_cancel", {"order_id": "D521"})),
    ("帮我取消订单", result("ask_clarification", "cancel_order", missing=["order_id"])),
    ("这个订单我不想要了", result("ask_clarification", "cancel_order", missing=["order_id"])),
    ("可以撤销购买吗", result("ask_clarification", "cancel_order", missing=["order_id"])),
    ("退款单R301处理到哪了", result("call_tool", "query_refund", "refund_query", {"refund_id": "R301"})),
    ("查询退款编号R889的进度", result("call_tool", "query_refund", "refund_query", {"refund_id": "R889"})),
    ("R520退款成功了吗", result("call_tool", "query_refund", "refund_query", {"refund_id": "R520"})),
    ("看下退款单R102现在什么状态", result("call_tool", "query_refund", "refund_query", {"refund_id": "R102"})),
    ("退款编号R777到账没有", result("call_tool", "query_refund", "refund_query", {"refund_id": "R777"})),
    ("我的退款怎么样了", result("ask_clarification", "query_refund", missing=["refund_id"])),
    ("查一下退款进度", result("ask_clarification", "query_refund", missing=["refund_id"])),
    ("退款到账了吗", result("ask_clarification", "query_refund", missing=["refund_id"])),
    ("你好", result("no_tool", "chitchat")),
    ("谢谢你的帮助", result("no_tool", "chitchat")),
    ("你是谁", result("no_tool", "chitchat")),
    ("讲个笑话", result("no_tool", "chitchat")),
    ("今天天气真不错", result("no_tool", "chitchat")),
    ("物流是什么意思", result("no_tool", "knowledge_question")),
    ("怎么申请退款", result("no_tool", "knowledge_question")),
    ("订单号一般在哪里看", result("no_tool", "knowledge_question")),
    ("为什么需要提供城市", result("no_tool", "knowledge_question")),
    ("取消订单有什么规则", result("no_tool", "knowledge_question")),
    ("查明天成都天气，顺便看看订单A1024", result("ask_clarification", "multiple_intents", missing=["single_intent"])),
    ("取消订单B7788还是先查下物流吧", result("ask_clarification", "multiple_intents", missing=["confirmed_intent"])),
    ("帮我处理一下", result("ask_clarification", "unknown", missing=["intent"])),
    ("这个怎么弄", result("ask_clarification", "unknown", missing=["intent"])),
    ("给我查一下", result("ask_clarification", "unknown", missing=["intent"])),
    ("我有个事情需要处理", result("ask_clarification", "unknown", missing=["intent"])),
]


def to_record(item: tuple[str, str]) -> dict[str, list[dict[str, str]]]:
    user, assistant = item
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def write_jsonl(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(to_record(row), ensure_ascii=False) + "\n")


def main() -> None:
    rows = EXAMPLES.copy()
    random.Random(42).shuffle(rows)
    TOOL_ROUTER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(TOOL_ROUTER_DATA_DIR / "train.jsonl", rows[:38])
    write_jsonl(TOOL_ROUTER_DATA_DIR / "valid.jsonl", rows[38:43])
    write_jsonl(TOOL_ROUTER_DATA_DIR / "test.jsonl", rows[43:])
    print(f"Wrote {len(rows[:38])} train, {len(rows[38:43])} valid, {len(rows[43:])} test examples")


if __name__ == "__main__":
    main()
