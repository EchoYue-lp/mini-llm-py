#!/usr/bin/env python3
"""Evaluate base, short-LoRA, and long-LoRA variants on one test split."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from utils.project_paths import (
    PROJECT_ROOT,
    TOOL_ROUTER_LONG_ADAPTER_DIR,
    TOOL_ROUTER_RESULTS_DIR,
    TOOL_ROUTER_SHORT_ADAPTER_DIR,
)

EXPERIMENTS = [
    ("base", None, TOOL_ROUTER_RESULTS_DIR / "base.json"),
    (
        "short-lora",
        TOOL_ROUTER_SHORT_ADAPTER_DIR,
        TOOL_ROUTER_RESULTS_DIR / "short-lora.json",
    ),
    (
        "long-lora",
        TOOL_ROUTER_LONG_ADAPTER_DIR,
        TOOL_ROUTER_RESULTS_DIR / "long-lora.json",
    ),
]


METRIC_NAMES = {
    "raw_json_valid": "原始JSON合法率",
    "extractable_json": "宽松提取率",
    "schema_valid": "Schema合法率",
    "exact_match": "整条完全正确率",
    "action": "动作准确率",
    "intent": "意图准确率",
    "tool": "工具准确率",
    "arguments": "参数准确率",
    "missing_arguments": "缺参判断准确率",
}


def compact(value: dict[str, Any] | None, raw: str) -> str:
    if value is None:
        return "JSON非法: " + raw.replace("|", "\\|")
    arguments = json.dumps(value.get("arguments", {}), ensure_ascii=False)
    missing = json.dumps(value.get("missing_arguments", []), ensure_ascii=False)
    return (
        f"action={value.get('action')}, intent={value.get('intent')}, "
        f"tool={value.get('tool')}, arguments={arguments}, missing={missing}"
    ).replace("|", "\\|")


def write_markdown(reports: list[dict[str, Any]]) -> Path:
    output = TOOL_ROUTER_RESULTS_DIR / "comparison.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics = list(METRIC_NAMES)
    lines = [
        "# 模型效果对比",
        "",
        "> 这是一个只有 8 条测试样本的教学 Demo，用来验证评测流程，不能代表生产效果。",
        "",
        "## 如何阅读指标",
        "",
        "- **原始JSON合法率**：完整输出能否直接解析为一个 JSON 对象。",
        "- **宽松提取率**：从首个 `{` 到末个 `}` 的片段能否解析。",
        "- **Schema合法率**：字段、类型和跨字段业务约束是否全部满足。",
        "- **整条完全正确率**：动作、意图、工具、参数和缺参列表全部正确。",
        "- **动作准确率**：是否正确选择调用工具、追问或不调用。",
        "- **意图准确率**：业务意图是否正确。",
        "- **工具准确率**：工具名称或不调用判断是否正确。",
        "- **参数准确率**：工具参数名称和值是否全部正确。",
        "- **缺参判断准确率**：缺少哪些参数是否判断正确。",
        "",
        "## 总体结果",
        "",
        "| 模型 | " + " | ".join(METRIC_NAMES[item] for item in metrics) + " |",
        "| --- | " + " | ".join("---:" for _ in metrics) + " |",
    ]
    for report in reports:
        values = [f"{report['metrics'][item]:.0%}" for item in metrics]
        lines.append(f"| {report['label']} | " + " | ".join(values) + " |")

    best_exact = max(report["metrics"]["exact_match"] for report in reports)
    best_labels = [
        report["label"]
        for report in reports
        if report["metrics"]["exact_match"] == best_exact
    ]
    extraction_gaps = [
        report["label"]
        for report in reports
        if report["metrics"]["extractable_json"]
        > report["metrics"]["raw_json_valid"]
    ]
    lines.extend(
        [
            "",
            "## 直观结论",
            "",
            f"- 整条完全正确率最高：{', '.join(best_labels)}（{best_exact:.0%}）。",
            "- 存在宽松可提取但并非原始 JSON 的模型："
            + (", ".join(extraction_gaps) if extraction_gaps else "无")
            + "。",
            "- 训练损失接近 0 不等于测试集全对，需要独立评测和更丰富的数据。",
            "",
            "## 逐条对比",
            "",
        ]
    )
    reference = reports[0]["predictions"]
    for index, expected_row in enumerate(reference, start=1):
        expected = expected_row["expected"]
        lines.extend(
            [
                f"### 样本 {index}: {expected_row['input']}",
                "",
                "期望输出：",
                "",
                "```json",
                json.dumps(expected, ensure_ascii=False, indent=2),
                "```",
                "",
                "| 模型 | 完全正确 | 实际输出 |",
                "| --- | --- | --- |",
            ]
        )
        for report in reports:
            prediction = report["predictions"][index - 1]
            actual = prediction["actual"]
            correct = actual == expected
            lines.append(
                f"| {report['label']} | {'是' if correct else '否'} | "
                f"{compact(actual, prediction['raw'])} |"
            )
        lines.append("")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main() -> None:
    for label, adapter, output in EXPERIMENTS:
        command = [
            sys.executable,
            "-m",
            "evaluation.tool_router",
            "--label",
            label,
            "--output",
            str(output),
        ]
        if adapter:
            command.extend(["--adapter", str(adapter)])
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )

    columns = [
        "raw_json_valid",
        "extractable_json",
        "schema_valid",
        "exact_match",
        "action",
        "intent",
        "tool",
        "arguments",
        "missing_arguments",
    ]
    reports = []
    print("模型\t" + "\t".join(METRIC_NAMES[column] for column in columns))
    for label, _, output in EXPERIMENTS:
        report = json.loads(output.read_text(encoding="utf-8"))
        reports.append(report)
        metrics = report["metrics"]
        values = [f"{metrics[column]:.2%}" for column in columns]
        print(label + "\t" + "\t".join(values))
    markdown = write_markdown(reports)
    print(f"\n详细逐条报告: {markdown}")


if __name__ == "__main__":
    main()
