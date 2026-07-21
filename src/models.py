"""项目中使用的轻量数据对象。

第一版不引入额外的模型校验框架，先用 dataclass 固化输入输出协议中的关键字段。
"""

from dataclasses import asdict, dataclass, field as dc_field
from typing import Any, Literal


EvaluationStatus = Literal["success", "partial_success", "failed"]
MetricStatus = Literal["evaluated", "not_assessable"]


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
    reason: str | None = None


@dataclass
class RiskItem:
    """风险提示：只描述值得关注的现象，不断言数据一定错误。"""

    id: str
    level: Literal["info", "attention", "warning"]
    title: str
    message: str
    related_metrics: list[str] = dc_field(default_factory=list)
    evidence: dict[str, Any] = dc_field(default_factory=dict)


@dataclass
class NotAssessableItem:
    """因数据或外部规则不足而无法评估的事项。"""

    id: str
    name: str
    reason: str


@dataclass
class QualityReport:
    """对应输入输出协议中 report.json 的顶层结构。"""

    dataset: DatasetInfo
    status: EvaluationStatus = "success"
    schema_version: str = "0.1"
    profile: dict[str, Any] = dc_field(default_factory=dict)
    metrics: list[MetricResult] = dc_field(default_factory=list)
    risks: list[RiskItem] = dc_field(default_factory=list)
    not_assessable: list[NotAssessableItem] = dc_field(default_factory=list)
    execution: dict[str, list[str]] = dc_field(
        default_factory=lambda: {"warnings": [], "errors": []}
    )

    def to_dict(self) -> dict[str, Any]:
        """转换为可直接写入 report.json 的普通字典。"""

        return asdict(self)
