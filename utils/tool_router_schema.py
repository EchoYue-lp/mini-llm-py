"""Single source of truth for the tool-routing teaching task."""

SYSTEM_PROMPT = (
    "你是工具路由模型。只输出一个JSON对象，不要解释。"
    "字段为action、intent、tool、arguments、missing_arguments。"
    "action只能是call_tool、ask_clarification或no_tool。"
    "工具签名：weather_query(city,date)、logistics_query(order_id)、"
    "order_cancel(order_id)、refund_query(refund_id)。"
    "缺少参数时tool必须为null，arguments必须为空对象，"
    "missing_arguments只能使用city、date、order_id、refund_id、"
    "single_intent、confirmed_intent或intent。"
)

ACTIONS = frozenset({"call_tool", "ask_clarification", "no_tool"})
TOOL_ARGUMENTS = {
    "weather_query": frozenset({"city", "date"}),
    "logistics_query": frozenset({"order_id"}),
    "order_cancel": frozenset({"order_id"}),
    "refund_query": frozenset({"refund_id"}),
}
INTENT_TO_TOOL = {
    "query_weather": "weather_query",
    "query_logistics": "logistics_query",
    "cancel_order": "order_cancel",
    "query_refund": "refund_query",
}
NO_TOOL_INTENTS = frozenset({"chitchat", "knowledge_question"})
AMBIGUOUS_INTENTS = {
    "multiple_intents": frozenset({"single_intent", "confirmed_intent"}),
    "unknown": frozenset({"intent"}),
}
INTENTS = frozenset(INTENT_TO_TOOL) | NO_TOOL_INTENTS | frozenset(AMBIGUOUS_INTENTS)
MISSING_ARGUMENTS = (
    frozenset().union(*TOOL_ARGUMENTS.values())
    | frozenset().union(*AMBIGUOUS_INTENTS.values())
)
