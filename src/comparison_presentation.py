"""v0.5 比较整改闭环的安全导出。

导出层只消费已经通过严格校验的 ``RemediationPlan`` 或
``GovernanceRecord``，不读取原始数据，也不重新计算比较结论。
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Mapping

from .remediation import (
    GovernanceRecord,
    RemediationPlan,
    validate_governance_record,
    validate_remediation_plan,
)


_PRIORITY_LABELS = {
    "high": "高",
    "medium": "中",
    "low": "低",
}
_STATUS_LABELS = {
    "open": "待处理",
    "in_progress": "处理中",
    "done": "已完成",
    "accepted_risk": "已接受风险",
}
_CATEGORY_LABELS = {
    "risk": "风险",
    "metric": "指标",
    "assessability": "可评估性",
    "schema": "结构",
}
_CSV_FIELDNAMES = [
    "任务ID",
    "任务类别",
    "优先级",
    "状态",
    "任务标题",
    "任务说明",
    "验收标准",
    "建议责任角色",
    "负责人",
    "截止日期",
    "关联变化ID",
]


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8", errors="strict")


def _markdown_text(value: object) -> str:
    """转义计划内容中的 Markdown、HTML 和链接/图片语法。"""

    text = str(value)
    for character in (
        "\\",
        "`",
        "*",
        "_",
        "[",
        "]",
        "(",
        ")",
        "<",
        ">",
        "#",
        "!",
        "|",
    ):
        text = text.replace(character, f"\\{character}")
    return text


def _spreadsheet_safe(value: object) -> object:
    """阻止用户可控文本在常见表格软件中被解释为公式。"""

    if not isinstance(value, str):
        return value
    stripped = value.lstrip()
    if stripped.startswith(("=", "+", "-", "@")):
        return f"'{value}"
    return value


def serialize_action_plan_json(
    plan: RemediationPlan | Mapping[str, Any],
) -> bytes:
    """序列化经校验的结构化整改计划。"""

    return _json_bytes(validate_remediation_plan(plan))


def render_action_plan_markdown(
    plan: RemediationPlan | Mapping[str, Any],
) -> str:
    """生成供人工评审的 Markdown 整改计划。"""

    payload = validate_remediation_plan(plan)
    summary = payload["improvement_summary"]
    lines = [
        "# 数据质量整改行动计划",
        "",
        _markdown_text(summary["headline"]),
        "",
        "## 追溯信息",
        "",
        f"- 计划 ID：{_markdown_text(payload['plan_id'])}",
        f"- 计划哈希：{_markdown_text(payload['plan_sha256'])}",
        f"- 比较哈希：{_markdown_text(payload['comparison_sha256'])}",
        "- 生成方式：本地确定性模板（不调用外部模型）",
        "",
        "## 改进摘要",
        "",
        f"- 改善指标变化：{len(summary['improved_change_ids'])} 项",
        f"- 退化变化：{len(summary['regressed_change_ids'])} 项",
        f"- 已解除或降级风险：{len(summary['resolved_risk_change_ids'])} 项",
    ]
    limitations = summary["limitations"]
    if limitations:
        lines.extend(["", "### 结论限制", ""])
        lines.extend(f"- {_markdown_text(item)}" for item in limitations)

    lines.extend(["", "## 整改任务", ""])
    tasks = payload["tasks"]
    if not tasks:
        lines.append("当前比较没有生成确定性的整改任务。")
    for index, task in enumerate(tasks, start=1):
        lines.extend(
            [
                f"### {index}. {_markdown_text(task['title'])}",
                "",
                f"- 任务 ID：{_markdown_text(task['task_id'])}",
                f"- 类别：{_CATEGORY_LABELS[task['category']]}",
                f"- 优先级：{_PRIORITY_LABELS[task['priority']]}",
                f"- 状态：{_STATUS_LABELS[task['status']]}",
                f"- 建议责任角色：{_markdown_text(task['suggested_owner_role'])}",
                f"- 负责人：{_markdown_text(task['assignee'] or '未分派')}",
                f"- 截止日期：{_markdown_text(task['due_date'] or '未设置')}",
                f"- 关联变化：{_markdown_text('、'.join(task['change_ids']))}",
                "",
                _markdown_text(task["detail"]),
                "",
                "验收标准：",
                "",
            ]
        )
        lines.extend(
            f"- {_markdown_text(item)}"
            for item in task["acceptance_criteria"]
        )

    lines.extend(["", "## 下一轮建议", ""])
    lines.extend(
        f"- {_markdown_text(item)}"
        for item in payload["next_round_suggestions"]
    )
    lines.extend(
        [
            "",
            "---",
            "",
            "说明：任务由固定比较结果生成；负责人和截止日期是人工分派信息，"
            "不改变证据、优先级或比较结论。",
            "",
        ]
    )
    return "\n".join(lines)


def serialize_action_plan_markdown(
    plan: RemediationPlan | Mapping[str, Any],
) -> bytes:
    """生成 UTF-8 Markdown 整改计划。"""

    return render_action_plan_markdown(plan).encode(
        "utf-8",
        errors="strict",
    )


def serialize_action_plan_csv(
    plan: RemediationPlan | Mapping[str, Any],
) -> bytes:
    """生成不含原始值、兼容 Excel 且防公式注入的任务表。"""

    payload = validate_remediation_plan(plan)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=_CSV_FIELDNAMES)
    writer.writeheader()
    for task in payload["tasks"]:
        row = {
            "任务ID": task["task_id"],
            "任务类别": _CATEGORY_LABELS[task["category"]],
            "优先级": _PRIORITY_LABELS[task["priority"]],
            "状态": _STATUS_LABELS[task["status"]],
            "任务标题": task["title"],
            "任务说明": task["detail"],
            "验收标准": "；".join(task["acceptance_criteria"]),
            "建议责任角色": task["suggested_owner_role"],
            "负责人": task["assignee"] or "",
            "截止日期": task["due_date"] or "",
            "关联变化ID": "；".join(task["change_ids"]),
        }
        writer.writerow(
            {
                key: _spreadsheet_safe(value)
                for key, value in row.items()
            }
        )
    return output.getvalue().encode("utf-8-sig", errors="strict")


def serialize_governance_record(
    record: GovernanceRecord | Mapping[str, Any],
) -> bytes:
    """序列化经校验且绑定比较、计划与两份报告的治理记录。"""

    return _json_bytes(validate_governance_record(record))
