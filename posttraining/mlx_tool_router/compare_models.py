#!/usr/bin/env python3
"""Evaluate base, short-LoRA, and long-LoRA variants on one test split."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENTS = [
    ("base", None, "results/base.json"),
    ("short-lora", "adapters/tool-router-short", "results/short-lora.json"),
    ("long-lora", "adapters/tool-router-long", "results/long-lora.json"),
]


METRIC_NAMES = {
    "json_valid": "JSON合法率",
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
    output = PROJECT_ROOT / "results" / "comparison.md"
    metrics = list(METRIC_NAMES)
    lines = [
        "# 模型效果对比",
        "",
        "> 这是一个只有 5 条测试样本的教学 Demo，用来验证评测流程，不能代表生产效果。",
        "",
        "## 如何阅读指标",
        "",
        "- **JSON合法率**：输出能否被程序解析。",
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

    lines.extend(
        [
            "",
            "## 直观结论",
            "",
            "- 基座模型能够输出 JSON，但没有稳定遵守本项目的意图和动作定义。",
            "- 短训练已经学到部分标签和工具格式，但可能产生非法 JSON。",
            "- 长训练的整条正确率更高，但仍会在缺参判断上犯错。",
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
            "evaluate.py",
            "--label",
            label,
            "--output",
            output,
        ]
        if adapter:
            command.extend(["--adapter", adapter])
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )

    columns = [
        "json_valid",
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
        report = json.loads((PROJECT_ROOT / output).read_text(encoding="utf-8"))
        reports.append(report)
        metrics = report["metrics"]
        values = [f"{metrics[column]:.2%}" for column in columns]
        print(label + "\t" + "\t".join(values))
    markdown = write_markdown(reports)
    print(f"\n详细逐条报告: {markdown}")


if __name__ == "__main__":
    main()
