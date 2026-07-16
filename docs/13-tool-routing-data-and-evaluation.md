# 13 工具路由数据与评测

工具路由不是普通文本生成。模型需要输出可解析、符合业务 Schema 的结构化决策。

## 学习目标

读完后应能：

1. 区分 intent、action、tool 和 arguments。
2. 设计完整参数、缺参、no-tool 与边界样本。
3. 手算字段指标和 exact match。
4. 识别小测试集、宽松 JSON 解析和 test 泄漏的风险。

## 输出 Schema

```json
{
  "action": "call_tool",
  "intent": "query_logistics",
  "tool": "logistics_query",
  "arguments": {"order_id": "A1024"},
  "missing_arguments": []
}
```

固定字段：

- `action`
- `intent`
- `tool`
- `arguments`
- `missing_arguments`

## 三类动作

| 行为 | action | 示例 |
| --- | --- | --- |
| 调用工具 | `call_tool` | 有订单号时查询物流 |
| 缺参追问 | `ask_clarification` | 查天气但缺少城市 |
| 不调用工具 | `no_tool` | 闲聊、知识问题、OOS |

## 工具与参数

| 工具 | 参数 |
| --- | --- |
| `weather_query` | `city`, `date` |
| `logistics_query` | `order_id` |
| `order_cancel` | `order_id` |
| `refund_query` | `refund_id` |

若参数不足：

- `tool` 必须为 null。
- `arguments` 必须为空对象。
- `missing_arguments` 明确列出缺失项。

## 数据覆盖

训练集不能只包含成功调用。还应覆盖：

- 缺少一个或多个参数。
- 多意图请求。
- 模糊请求。
- 闲聊和知识问题。
- OOS 请求。
- 否定、口语、错别字和 ASR 错误。

## 数据拆分

```text
train: 38
valid: 5
test:  5
```

当前只有 48 条教学数据，用于验证流程，不代表生产能力。

## 校验

```bash
python -m evaluation.validate_tool_router_data
```

校验内容：

1. JSONL 可解析。
2. Chat 角色顺序正确。
3. 输出字段集合精确匹配。
4. Action 在允许集合内。
5. 工具参数合法。
6. 缺参动作与字段一致。
7. `no_tool` 不携带工具参数。
8. Train/valid/test 没有完全重复输入。

真实项目还应增加语义近似去重。

## 为什么不能只看 Loss

语言模型 loss 衡量目标 token 概率。即使 loss 很低，模型仍可能：

- 输出非法 JSON。
- 选错 action。
- 工具正确但参数错误。
- 漏掉缺失参数。
- 记住训练样本但不能泛化。

## 评测指标

| 指标 | 含义 |
| --- | --- |
| JSON 合法率 | 输出是否可解析 |
| 完全正确率 | 五个字段全部正确 |
| Action 准确率 | 调用、追问、不调用 |
| Intent 准确率 | 业务意图 |
| Tool 准确率 | 工具或不调用判断 |
| Arguments 准确率 | 参数名和值 |
| Missing 准确率 | 缺参列表 |

完全正确率是最严格的主指标，字段指标用于定位失败原因。

## 确定性评测

评测使用固定测试集和 `temperature=0`，减少采样噪声。所有模型必须使用同一：

- 基座模型。
- System prompt。
- Test split。
- 最大生成长度。
- JSON 解析规则。

## 逐样本报告

```bash
python -m evaluation.compare_tool_router_models
```

报告包含：

- 用户输入。
- 期望 JSON。
- 每个模型的原始输出。
- 解析结果。
- 是否完全正确。

聚合百分比不能解释失败模式，必须结合逐样本报告。

## 对照源码

- `scripts/prepare_tool_router_data.py`
- `evaluation/validate_tool_router_data.py`
- `evaluation/tool_router.py`
- `evaluation/compare_tool_router_models.py`

## 先定义决策边界

工具路由数据最难的部分不是 JSON 格式，而是标签边界。

例如：

```text
"上海明天天气怎么样"
```

信息完整，应 `call_tool`。

```text
"明天天气怎么样"
```

缺少 city，应 `ask_clarification`。

```text
"天气预报是怎么做出来的"
```

这是知识问题，应 `no_tool`。

三句话都包含“天气”，若数据只教关键词匹配，模型会错误地全部调用工具。

## Intent、Tool 与 Action 的区别

| 字段 | 回答的问题 |
| --- | --- |
| Intent | 用户想做什么 |
| Action | 当前应该调用、追问还是不调用 |
| Tool | 若调用，具体调用哪个工具 |

同一个 intent 可能对应不同 action：

```text
intent=query_weather
信息完整 -> call_tool
缺少城市 -> ask_clarification
```

因此不能把 intent 分类结果直接当工具调用结果。

## Missing Arguments 必须来自工具签名

若工具：

```text
weather_query(city, date)
```

输入只提供 city：

```json
{
  "action": "ask_clarification",
  "intent": "query_weather",
  "tool": null,
  "arguments": {},
  "missing_arguments": ["date"]
}
```

不要在 `arguments` 中放半完整参数同时又将 tool 设为 null，除非业务协议明确允许。
Schema 应先定义，再生成数据。

## 多意图请求

```text
"查明天成都天气，顺便看看订单 A1024"
```

可能策略：

- 要求用户选择一个意图。
- 支持一次返回多个 tool calls。
- 按优先级执行。

项目当前选择第一种，并标记 `missing_arguments=["single_intent"]`。真实系统必须明确
协议，不能让标注人员各自决定。

## 负样本为什么重要

若训练集大部分都是 `call_tool`，模型会形成过度调用倾向。需要足够：

- `no_tool`。
- `ask_clarification`。
- 相似关键词但不应调用的 hard negatives。
- 工具之间的边界样本。

负样本比例应接近真实流量，或在评测中按业务分布重新加权。

## 数据质量检查示例

以下记录应失败：

```json
{
  "action": "call_tool",
  "intent": "query_weather",
  "tool": "weather_query",
  "arguments": {},
  "missing_arguments": ["city"]
}
```

原因：`call_tool` 不应带 missing arguments，且必要参数不完整。

以下记录也应失败：

```json
{
  "action": "no_tool",
  "intent": "chitchat",
  "tool": "weather_query",
  "arguments": {},
  "missing_arguments": []
}
```

原因：`no_tool` 时 tool 必须为 null。

## JSON 提取

模型可能输出：

```text
好的，结果如下：
{"action":"no_tool", ...}
```

评测代码可以尝试提取首个 `{` 到最后一个 `}`，但生产环境更应使用：

- 结构化输出 API。
- JSON Schema。
- Grammar-constrained decoding。
- 严格重试策略。

宽松解析会掩盖模型不遵守输出协议的问题，因此仍应单独统计原始 JSON 合法率。

## 指标手算

期望：

```json
{
  "action": "call_tool",
  "intent": "query_logistics",
  "tool": "logistics_query",
  "arguments": {"order_id": "A1024"},
  "missing_arguments": []
}
```

模型只把 order id 预测成 `A1025`：

- JSON valid：正确。
- Action：正确。
- Intent：正确。
- Tool：正确。
- Arguments：错误。
- Missing：正确。
- Exact match：错误。

字段级指标告诉我们错误位置，Exact match 告诉我们该调用整体不可直接执行。

## 小测试集百分比的陷阱

测试集只有 5 条时：

```text
1 条样本 = 20%
```

从 60% 到 80% 只代表多答对一条，统计波动很大。报告百分比时必须同时报告样本数和逐条
结果。

## 混淆分析

建议统计 Action confusion matrix：

```text
expected call_tool -> predicted ask_clarification
expected no_tool   -> predicted call_tool
```

其中“把 no_tool 误判为 call_tool”可能产生真实外部操作，风险通常高于“应该调用但追问”。
业务评测应按错误成本设定优先级。

## Test 泄漏

以下行为都会污染 test：

- 根据 test 错误修改 prompt。
- 根据 test 选择 checkpoint。
- 将 test 样本改写后加入 train，再继续报告原 test。
- 多次试验后只汇报 test 最好的一次。

正确做法是使用 validation 迭代，test 仅在方案冻结后运行。

## 生产评测还需要什么

- 更大的人工审核集。
- 按意图、动作、语言风格分层统计。
- 危险工具单独评测。
- 参数格式与权限检查。
- 延迟、超时和重试指标。
- 线上 shadow traffic。
- 人工错误严重度分级。

## 动手练习

1. 为每个工具写 5 条完整参数和 5 条缺参样本。
2. 写 10 条包含工具关键词但应 `no_tool` 的 hard negatives。
3. 手算一个 5 条测试集的所有字段指标。
4. 增加 Action confusion matrix。
5. 故意创建一条跨 split 重复数据，确认 validator 拒绝。

## 自测

1. Intent 与 Action 为什么不能合并成一个字段？
2. JSON valid 为什么不等于业务正确？
3. 为什么 Exact match 是工具调用的重要主指标？
4. 5 条测试集的 80% 实际是多少条？
5. 哪类误调用具有最高业务风险？
