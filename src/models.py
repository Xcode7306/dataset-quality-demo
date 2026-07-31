"""项目中使用的轻量数据对象和稳定报告协议。"""

import hashlib
import json
from dataclasses import InitVar, asdict, dataclass, field as dc_field
from typing import Any, Literal
from urllib.parse import quote

from .config import ENGINE_VERSION, THRESHOLD_CONFIG_VERSION
from .metric_catalog import (
    DEFAULT_SELECTED_METRIC_IDS,
    METRIC_CATALOG_VERSION,
)


EvaluationStatus = Literal["success", "partial_success", "failed"]
MetricStatus = Literal["evaluated", "not_assessable"]


def build_metric_key(
    metric_id: str,
    scope: Literal["dataset", "field"],
    field: str | None = None,
) -> str:
    """生成可供 Agent 精确引用的稳定指标键。"""

    if scope == "dataset":
        return f"metric:{metric_id}:dataset"
    encoded_field = quote(
        field or "",
        safe="",
        encoding="utf-8",
        errors="replace",
    )
    return f"metric:{metric_id}:field:{encoded_field}"


def _default_evaluation_context() -> dict[str, Any]:
    """返回字段完备、可安全序列化的默认评估上下文。"""

    return {
        "engine_version": ENGINE_VERSION,
        "metric_catalog_version": METRIC_CATALOG_VERSION,
        "selected_metric_ids": list(DEFAULT_SELECTED_METRIC_IDS),
        "reference_date": None,
        "threshold_config_version": THRESHOLD_CONFIG_VERSION,
        "parser_path": None,
        "input_sha256": None,
        "input_size_bytes": None,
        "report_sha256": None,
    }


@dataclass
class DatasetInfo:
    """一次评估所对应的输入文件信息。"""

    name: str
    file_name: str
    file_type: str
    sheet_name: str | None = None


@dataclass
class MetricResult:
    """单个质量指标的统一输出结构。"""

    id: str
    name: str
    category: str
    status: MetricStatus
    value: float | int | None
    unit: str | None
    scope: Literal["dataset", "field"]
    field: str | None = None
    evidence: dict[str, Any] = dc_field(default_factory=dict)
    issue_locations: InitVar[list[dict[str, Any]] | None] = None
    reason: str | None = None
    metric_key: str = dc_field(init=False)

    def __post_init__(
        self,
        issue_locations: list[dict[str, Any]] | None,
    ) -> None:
        self.metric_key = build_metric_key(self.id, self.scope, self.field)
        # InitVar 不参与 dataclasses.asdict；位置明细因此不会被复制进
        # JSON 报告、报告哈希或 Agent 上下文。
        self.issue_locations = list(issue_locations or [])


@dataclass
class RiskItem:
    """风险提示：只描述值得关注的现象，不断言数据一定错误。"""

    id: str
    level: Literal["info", "attention", "warning"]
    title: str
    message: str
    related_metrics: list[str] = dc_field(default_factory=list)
    related_metric_keys: list[str] = dc_field(default_factory=list)
    evidence: dict[str, Any] = dc_field(default_factory=dict)


@dataclass
class NotAssessableItem:
    """因数据或外部规则不足而无法评估的事项。"""

    id: str
    name: str
    reason: str
    metric_key: str


@dataclass
class QualityReport:
    """评估过程使用的结构化报告对象；Markdown 由其生成。"""

    dataset: DatasetInfo
    status: EvaluationStatus = "success"
    schema_version: str = "0.3"
    profile: dict[str, Any] = dc_field(default_factory=dict)
    metrics: list[MetricResult] = dc_field(default_factory=list)
    risks: list[RiskItem] = dc_field(default_factory=list)
    not_assessable: list[NotAssessableItem] = dc_field(default_factory=list)
    evaluation_context: dict[str, Any] = dc_field(
        default_factory=_default_evaluation_context
    )
    execution: dict[str, list[str]] = dc_field(
        default_factory=lambda: {"warnings": [], "errors": []}
    )

    def to_dict(self) -> dict[str, Any]:
        """转换为供内部处理或扩展使用的普通字典。"""

        payload = asdict(self)
        evaluation_context = dict(payload["evaluation_context"])
        evaluation_context.pop("report_sha256", None)
        payload["evaluation_context"] = evaluation_context
        canonical_payload = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8", errors="replace")
        payload["evaluation_context"]["report_sha256"] = hashlib.sha256(
            canonical_payload
        ).hexdigest()
        return payload
