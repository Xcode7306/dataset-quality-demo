"""v0.5 确定性跨版本报告比较协议。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal


COMPARISON_SCHEMA_VERSION = "0.1"
COMPARATOR_VERSION = "0.5"
COMPARISON_POLICY_VERSION = "0.1"

MetricDirection = Literal["higher_is_better", "lower_is_better", "neutral"]
MetricClassification = Literal[
    "added",
    "removed",
    "unchanged",
    "improved",
    "worsened",
    "changed",
    "became_assessable",
    "became_not_assessable",
    "not_comparable",
]
RiskClassification = Literal[
    "added",
    "resolved",
    "persistent",
    "severity_increased",
    "severity_decreased",
    "not_comparable",
]
AssessabilityClassification = Literal[
    "became_assessable",
    "became_not_assessable",
    "persistent",
    "reason_changed",
    "added_with_metric",
    "removed_with_metric",
]
SchemaChangeKind = Literal[
    "field_added",
    "field_removed",
    "field_type_changed",
    "field_order_changed",
    "row_count_changed",
    "column_count_changed",
]


@dataclass(frozen=True)
class ReportReference:
    report_sha256: str
    input_sha256: str | None
    report_schema_version: str
    engine_version: str
    threshold_config_version: str
    reference_date: str | None
    parser_path: str | None
    dataset_name: str
    file_name: str
    status: str
    row_count: int
    column_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_sha256": self.report_sha256,
            "input_sha256": self.input_sha256,
            "report_schema_version": self.report_schema_version,
            "engine_version": self.engine_version,
            "threshold_config_version": self.threshold_config_version,
            "reference_date": self.reference_date,
            "parser_path": self.parser_path,
            "dataset_name": self.dataset_name,
            "file_name": self.file_name,
            "status": self.status,
            "row_count": self.row_count,
            "column_count": self.column_count,
        }


@dataclass(frozen=True)
class MetricChange:
    change_id: str
    metric_key: str
    metric_id: str
    name: str
    field: str | None
    direction: MetricDirection
    classification: MetricClassification
    baseline_status: str | None
    target_status: str | None
    baseline_value: int | float | None
    target_value: int | float | None
    delta: int | float | None
    unit: str | None
    reason_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_id": self.change_id,
            "metric_key": self.metric_key,
            "metric_id": self.metric_id,
            "name": self.name,
            "field": self.field,
            "direction": self.direction,
            "classification": self.classification,
            "baseline_status": self.baseline_status,
            "target_status": self.target_status,
            "baseline_value": self.baseline_value,
            "target_value": self.target_value,
            "delta": self.delta,
            "unit": self.unit,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class RiskChange:
    change_id: str
    risk_id: str
    title: str
    classification: RiskClassification
    baseline_level: str | None
    target_level: str | None
    related_metric_keys: tuple[str, ...]
    reason_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_id": self.change_id,
            "risk_id": self.risk_id,
            "title": self.title,
            "classification": self.classification,
            "baseline_level": self.baseline_level,
            "target_level": self.target_level,
            "related_metric_keys": list(self.related_metric_keys),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class AssessabilityChange:
    change_id: str
    metric_key: str
    name: str
    classification: AssessabilityClassification
    baseline_reason: str | None
    target_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_id": self.change_id,
            "metric_key": self.metric_key,
            "name": self.name,
            "classification": self.classification,
            "baseline_reason": self.baseline_reason,
            "target_reason": self.target_reason,
        }


@dataclass(frozen=True)
class SchemaChange:
    change_id: str
    kind: SchemaChangeKind
    field: str | None
    baseline_value: str | int | list[str] | None
    target_value: str | int | list[str] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_id": self.change_id,
            "kind": self.kind,
            "field": self.field,
            "baseline_value": deepcopy(self.baseline_value),
            "target_value": deepcopy(self.target_value),
        }


@dataclass(frozen=True)
class ReportComparison:
    """与两份 QualityReport 分离、可复现的比较事实。"""

    comparison_id: str
    comparison_sha256: str
    baseline: ReportReference
    target: ReportReference
    lineage: dict[str, Any]
    compatibility: dict[str, Any]
    summary: dict[str, int]
    metric_changes: tuple[MetricChange, ...]
    risk_changes: tuple[RiskChange, ...]
    assessability_changes: tuple[AssessabilityChange, ...]
    schema_changes: tuple[SchemaChange, ...]
    limitations: tuple[str, ...]
    schema_version: str = COMPARISON_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "comparison_id": self.comparison_id,
            "comparison_sha256": self.comparison_sha256,
            "comparator": {
                "name": "quality-report-comparator",
                "version": COMPARATOR_VERSION,
                "policy_version": COMPARISON_POLICY_VERSION,
            },
            "baseline": self.baseline.to_dict(),
            "target": self.target.to_dict(),
            "lineage": deepcopy(self.lineage),
            "compatibility": deepcopy(self.compatibility),
            "summary": dict(self.summary),
            "metric_changes": [
                change.to_dict() for change in self.metric_changes
            ],
            "risk_changes": [
                change.to_dict() for change in self.risk_changes
            ],
            "assessability_changes": [
                change.to_dict() for change in self.assessability_changes
            ],
            "schema_changes": [
                change.to_dict() for change in self.schema_changes
            ],
            "limitations": list(self.limitations),
        }
