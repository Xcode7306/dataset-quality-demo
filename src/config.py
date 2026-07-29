"""集中管理风险提示规则、阈值和报告引擎版本。"""

from dataclasses import dataclass


ENGINE_VERSION = "0.3"
RISK_RULE_VERSION = "0.3"
THRESHOLD_CONFIG_VERSION = "0.3"


@dataclass(frozen=True)
class RiskThresholds:
    """风险阈值配置；比例均使用 0 到 1 的小数。"""

    field_missing_attention: float = 0.10
    field_missing_warning: float = 0.50
    blank_record_attention: float = 0.01
    blank_record_warning: float = 0.10
    type_consistency_attention: float = 0.95
    type_consistency_warning: float = 0.80
    format_anomaly_attention: float = 0.00
    format_anomaly_warning: float = 0.20
    duplicate_attention: float = 0.00
    duplicate_warning: float = 0.10
    time_availability_attention: float = 0.90
    time_availability_warning: float = 0.50
    coverage_attention: float = 0.90
    coverage_warning: float = 0.50
    update_lag_attention_days: int = 365
    update_lag_warning_days: int = 730


DEFAULT_RISK_THRESHOLDS = RiskThresholds()
