"""v0.6 可选择指标目录与稳定选择规范化。

本模块只描述指标，不读取数据，也不参与指标计算。原 v0.4 的 13 项指标
与 DB31/T 1523-2024 正文表 2 至表 7 的 30 项指标使用不同 ID，因此即使
名称或含义接近，也不会被去重或互相覆盖。
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Iterable, Mapping


LEGACY_SOURCE = "v0.4_original"
DB31_SOURCE = "db31t_1523_2024"
METRIC_CATALOG_VERSION = "0.6"


class MetricSelectionError(ValueError):
    """指标选择为空、包含未知 ID 或类型不合法。"""


METRIC_DESCRIPTIONS: Mapping[str, str] = MappingProxyType(
    {
        "file_parse_rate": "上传文件能否被当前解析器成功读取。",
        "dataset_scale": "解析后可参与质量评估的数据记录数量。",
        "field_missing_rate": "每个字段中缺失值占该字段记录数的比例。",
        "blank_record_rate": "可识别内容字段全部为空的记录占比。",
        "field_type_consistency": "字段非空值中占比最高的基础类型所占比例。",
        "recognizable_format_anomaly_rate": "按字段名可识别的日期、数值、URL 或邮箱字段中，格式异常值的比例。",
        "exact_duplicate_rate": "排除技术标识列后，与首次出现记录内容完全相同的后续记录占比。",
        "normalized_duplicate_rate": "忽略自然文本的空白、大小写和常见标点后，后续重复记录占比。",
        "time_info_availability": "至少包含一个可解析日期或时间的记录占比。",
        "update_lag_days": "评估基准日期与最近可解析更新时间之间的天数。",
        "source_info_coverage": "包含可识别来源部门、链接或原始标识信息的记录占比。",
        "version_info_coverage": "包含版本、更新时间或处理记录信息的记录占比。",
        "statistical_outlier_rate": "按 IQR 规则识别的数值异常在被检查数值中的比例。",
        "db31_010100": "数据符合类型、格式和长度等数据标准的程度。",
        "db31_010101": "数据元素符合预期数据类型约束的程度。",
        "db31_010102": "数据元素符合预期格式约束的程度。",
        "db31_010103": "数据元素符合预期长度约束的程度。",
        "db31_010200": "数据符合目标实体、字段、关系和模式定义的程度。",
        "db31_010300": "数据内容符合独立元数据定义的程度。",
        "db31_010400": "数据符合公共服务业务规则的程度。",
        "db31_010500": "数据符合权威参考数据或权威参考源规则的程度。",
        "db31_010600": "数据符合安全、权限、脱敏和隐私规则的程度。",
        "db31_020100": "按业务规则要求应赋值的数据元素填写完整、无缺失的程度。",
        "db31_020200": "按业务规则要求应赋值的数据记录填写完整、无缺失的程度。",
        "db31_030100": "数据内容与预期值或真实值相符的程度。",
        "db31_030200": "数据类型、范围、长度和精度等格式满足预期要求的程度。",
        "db31_030300": "特定字段、记录、文件或数据集意外重复较少的程度。",
        "db31_030400": "特定字段、记录、文件或数据集保持唯一的程度。",
        "db31_030500": "数据中无效或不符合定义的脏数据较少的程度。",
        "db31_030600": "数据与明确的权威数据参照保持一致的程度。",
        "db31_040100": "同一数据在不同位置、应用、用户或版本中保持一致的程度。",
        "db31_040200": "根据关联数据一致性约束检查表内和跨表一致性的程度。",
        "db31_040201": "同一表中跨列元素应相等的关系保持一致的程度。",
        "db31_040202": "同一表中跨列元素逻辑关系保持一致的程度。",
        "db31_040203": "关联表之间元素等值关系保持一致的程度。",
        "db31_040204": "关联表之间元素逻辑关系保持一致的程度。",
        "db31_040300": "内容数据记录的数据项与独立元数据保持一致的程度。",
        "db31_050100": "基于日期范围的记录数或频率分布符合业务需求的程度。",
        "db31_050200": "应及时公开或提供的数据处于有效期限内的程度。",
        "db31_050300": "同一实体数据元素间相对时序关系正确的程度。",
        "db31_050400": "在数据授权有效周期内提供数据使用的程度。",
        "db31_060100": "数据在需要的时间可被获取的程度。",
        "db31_060200": "数据适配目标应用场景并可被使用的程度。",
    }
)


def _legacy(
    metric_id: str,
    name: str,
    category: str,
    formula: str,
    *,
    direction: str,
) -> dict[str, Any]:
    return {
        "id": metric_id,
        "name": name,
        "category": category,
        "dimension": category,
        "source": LEGACY_SOURCE,
        "source_label": "原 v0.4 指标",
        "standard_code": None,
        "level": "原有",
        "parent_id": None,
        "formula": formula,
        "description": METRIC_DESCRIPTIONS[metric_id],
        "direction": direction,
        "auto_assessable": True,
        "reason_code": None,
        "required_inputs": (),
        "available_proxy_metric_ids": (),
    }


def _db31(
    code: str,
    name: str,
    dimension: str,
    formula: str,
    *,
    level: str = "二级",
    parent_code: str | None = None,
    auto_assessable: bool = False,
    reason_code: str | None = None,
    required_inputs: tuple[str, ...] = (),
    available_proxy_metric_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "id": f"db31_{code}",
        "name": name,
        "category": dimension,
        "dimension": dimension,
        "source": DB31_SOURCE,
        "source_label": "DB31/T 1523-2024",
        "standard_code": code,
        "level": level,
        "parent_id": f"db31_{parent_code}" if parent_code else None,
        "formula": formula,
        "description": METRIC_DESCRIPTIONS[f"db31_{code}"],
        "direction": "higher_is_better",
        "auto_assessable": auto_assessable,
        "reason_code": reason_code,
        "required_inputs": required_inputs,
        "available_proxy_metric_ids": available_proxy_metric_ids,
    }


ORIGINAL_METRIC_CATALOG: tuple[Mapping[str, Any], ...] = tuple(
    MappingProxyType(item)
    for item in (
        _legacy(
            "file_parse_rate",
            "文件可解析率",
            "可读取性",
            "成功解析文件数 / 尝试解析文件数",
            direction="higher_is_better",
        ),
        _legacy(
            "dataset_scale",
            "数据规模",
            "规模",
            "解析后的记录数",
            direction="neutral",
        ),
        _legacy(
            "field_missing_rate",
            "字段缺失率",
            "完整性",
            "字段缺失值数 / 字段记录数",
            direction="lower_is_better",
        ),
        _legacy(
            "blank_record_rate",
            "空白记录率",
            "完整性",
            "内容字段均为空的记录数 / 总记录数",
            direction="lower_is_better",
        ),
        _legacy(
            "field_type_consistency",
            "字段类型一致率",
            "类型一致性",
            "字段主要类型值数 / 字段非空值数",
            direction="higher_is_better",
        ),
        _legacy(
            "recognizable_format_anomaly_rate",
            "可识别格式异常率",
            "格式规范性",
            "可识别格式异常值数 / 被检查非空值数",
            direction="lower_is_better",
        ),
        _legacy(
            "exact_duplicate_rate",
            "完全重复率",
            "唯一性",
            "后续完全重复记录数 / 总记录数",
            direction="lower_is_better",
        ),
        _legacy(
            "normalized_duplicate_rate",
            "规范化重复率",
            "唯一性",
            "规范化后续重复记录数 / 总记录数",
            direction="lower_is_better",
        ),
        _legacy(
            "time_info_availability",
            "时间信息可用率",
            "及时性",
            "含可解析时间信息的记录数 / 总记录数",
            direction="higher_is_better",
        ),
        _legacy(
            "update_lag_days",
            "更新滞后天数",
            "及时性",
            "评估基准日期 - 最近更新时间",
            direction="lower_is_better",
        ),
        _legacy(
            "source_info_coverage",
            "来源信息覆盖率",
            "可溯性",
            "含来源信息的记录数 / 总记录数",
            direction="higher_is_better",
        ),
        _legacy(
            "version_info_coverage",
            "版本信息覆盖率",
            "可溯性",
            "含版本、更新或处理记录信息的记录数 / 总记录数",
            direction="higher_is_better",
        ),
        _legacy(
            "statistical_outlier_rate",
            "统计异常值比例",
            "数据异常",
            "IQR 规则识别的统计异常值数 / 被检查数值数",
            direction="neutral",
        ),
    )
)


DB31_METRIC_CATALOG: tuple[Mapping[str, Any], ...] = tuple(
    MappingProxyType(item)
    for item in (
        _db31(
            "010100",
            "数据标准",
            "公共数据规范性",
            "X = (A + B + C) / 3；A、B、C 分别为 010101、010102、010103 得分",
            reason_code="missing_data_standard",
            required_inputs=(
                "数据类型约束",
                "数据格式约束",
                "数据长度约束",
            ),
            available_proxy_metric_ids=(
                "field_type_consistency",
                "recognizable_format_anomaly_rate",
            ),
        ),
        _db31(
            "010101",
            "数据类型约束规范性",
            "公共数据规范性",
            "X = A / B；A 为满足数据类型约束的元素数，B 为被评价元素数",
            level="三级",
            parent_code="010100",
            reason_code="missing_type_constraints",
            required_inputs=("逐字段预期数据类型标准",),
            available_proxy_metric_ids=("field_type_consistency",),
        ),
        _db31(
            "010102",
            "数据格式约束规范性",
            "公共数据规范性",
            "X = A / B；A 为满足数据格式约束的元素数，B 为被评价元素数",
            level="三级",
            parent_code="010100",
            reason_code="missing_format_constraints",
            required_inputs=("逐字段数据格式标准",),
            available_proxy_metric_ids=("recognizable_format_anomaly_rate",),
        ),
        _db31(
            "010103",
            "数据长度约束规范性",
            "公共数据规范性",
            "X = A / B；A 为满足数据长度约束的元素数，B 为被评价元素数",
            level="三级",
            parent_code="010100",
            reason_code="missing_length_constraints",
            required_inputs=("逐字段长度约束及字符或字节计量规则",),
        ),
        _db31(
            "010200",
            "数据模型",
            "公共数据规范性",
            "X = A / B；A 为满足数据模型要求的元素数，B 为被评价元素数",
            reason_code="missing_data_model",
            required_inputs=("目标实体、字段、关系、基数及模式定义",),
        ),
        _db31(
            "010300",
            "元数据",
            "公共数据规范性",
            "X = A / B；A 为满足元数据定义的元素数，B 为被评价元素数",
            reason_code="missing_metadata_definition",
            required_inputs=("独立元数据定义或数据字典",),
        ),
        _db31(
            "010400",
            "业务规则",
            "公共数据规范性",
            "X = A / B；A 为满足业务规则的元素数，B 为被评价元素数",
            reason_code="missing_business_rules",
            required_inputs=("经确认的业务规则",),
        ),
        _db31(
            "010500",
            "权威参考数据（权威参考源）",
            "公共数据规范性",
            "X = A / B；A 为满足参考数据规则的元素数，B 为被评价元素数",
            reason_code="missing_authoritative_reference",
            required_inputs=("权威参考数据源", "匹配键", "参照规则"),
            available_proxy_metric_ids=("source_info_coverage",),
        ),
        _db31(
            "010600",
            "安全规范",
            "公共数据规范性",
            "X = A / B；A 为满足安全规范的元素数，B 为被评价元素数",
            reason_code="missing_security_rules",
            required_inputs=("分类分级、权限、脱敏及隐私规则",),
        ),
        _db31(
            "020100",
            "数据元素完整性",
            "公共数据完整性",
            "X = A / B；A 为被赋值元素数，B 为预期被赋值元素数",
            reason_code="missing_required_element_rules",
            required_inputs=("应赋值元素或必填字段规则",),
            available_proxy_metric_ids=("field_missing_rate",),
        ),
        _db31(
            "020200",
            "数据记录完整性",
            "公共数据完整性",
            "X = A / B；按标准原文，A 为被赋值元素数，B 为预期被赋值元素数",
            reason_code="missing_expected_record_population",
            required_inputs=("预期记录总体、业务键或应有记录清单",),
            available_proxy_metric_ids=("blank_record_rate",),
        ),
        _db31(
            "030100",
            "数据内容正确性",
            "公共数据准确性",
            "X = A / B；A 为满足数据正确性要求的元素数，B 为被评价元素数",
            reason_code="missing_expected_or_true_values",
            required_inputs=("真实值、预期值或内容正确性规则",),
        ),
        _db31(
            "030200",
            "数据格式合规性",
            "公共数据准确性",
            "X = A / B；按标准原文，A 为满足业务精度需求的元素数，B 为被评价元素数",
            reason_code="missing_format_compliance_rules",
            required_inputs=("预期类型、范围、长度及精度规则",),
            available_proxy_metric_ids=(
                "field_type_consistency",
                "recognizable_format_anomaly_rate",
                "statistical_outlier_rate",
            ),
        ),
        _db31(
            "030300",
            "数据重复率",
            "公共数据准确性",
            "X = 1 - A / B；A 为重复元素数，B 为被评价元素数",
            auto_assessable=True,
            available_proxy_metric_ids=("exact_duplicate_rate",),
        ),
        _db31(
            "030400",
            "数据唯一性",
            "公共数据准确性",
            "X = A / B；A 为满足唯一性要求的元素数，B 为被评价元素数",
            auto_assessable=True,
            available_proxy_metric_ids=("exact_duplicate_rate",),
        ),
        _db31(
            "030500",
            "脏数据出现率",
            "公共数据准确性",
            "X = 1 - A / B；A 为脏数据元素数，B 为被评价元素数",
            reason_code="missing_dirty_data_definition",
            required_inputs=("可执行的脏数据判定规则",),
            available_proxy_metric_ids=(
                "field_missing_rate",
                "recognizable_format_anomaly_rate",
                "statistical_outlier_rate",
            ),
        ),
        _db31(
            "030600",
            "数据标准参照准确性",
            "公共数据准确性",
            "X = A / B；A 为与明确权威参照一致的元素数，B 为被评价元素数",
            reason_code="missing_authoritative_reference",
            required_inputs=("权威参照数据", "匹配方式"),
            available_proxy_metric_ids=("source_info_coverage",),
        ),
        _db31(
            "040100",
            "相同数据一致性",
            "公共数据一致性",
            "X = A / B；A 为满足数据源一致性的数据数，B 为被评价元素数",
            reason_code="missing_comparison_copy",
            required_inputs=("其他位置、应用、用户或版本中的同一数据",),
        ),
        _db31(
            "040200",
            "关联数据一致性",
            "公共数据一致性",
            "X = (A + B + C + D) / 4；A、B、C、D 分别为 040201 至 040204 得分",
            reason_code="missing_association_constraints",
            required_inputs=(
                "表内等值约束",
                "表内逻辑约束",
                "跨表等值约束",
                "跨表逻辑约束",
            ),
        ),
        _db31(
            "040201",
            "表内等值一致性",
            "公共数据一致性",
            "X = A / B；A 为满足表内跨列等值一致性的数据数，B 为被评价元素数",
            level="三级",
            parent_code="040200",
            reason_code="missing_intra_table_equality_rules",
            required_inputs=("需保持等值的字段对、条件及空值语义",),
        ),
        _db31(
            "040202",
            "表内逻辑一致性",
            "公共数据一致性",
            "X = A / B；A 为满足表内跨列逻辑一致性的数据数，B 为被评价元素数",
            level="三级",
            parent_code="040200",
            reason_code="missing_intra_table_logic_rules",
            required_inputs=("跨字段逻辑规则",),
        ),
        _db31(
            "040203",
            "跨表等值一致性",
            "公共数据一致性",
            "X = A / B；A 为满足跨表等值一致性的数据数，B 为被评价元素数",
            level="三级",
            parent_code="040200",
            reason_code="missing_related_tables",
            required_inputs=("关联表、连接键及跨表等值约束",),
        ),
        _db31(
            "040204",
            "跨表逻辑一致性",
            "公共数据一致性",
            "X = A / B；A 为满足跨表逻辑一致性的数据数，B 为被评价元素数",
            level="三级",
            parent_code="040200",
            reason_code="missing_related_tables",
            required_inputs=("关联表、连接键及跨表逻辑规则",),
        ),
        _db31(
            "040300",
            "内容数据记录数据项与元数据一致性",
            "公共数据一致性",
            "X = A / B；A 为内容数据项与元数据一致的数据数，B 为被评价元素数",
            reason_code="missing_metadata_definition",
            required_inputs=("独立元数据定义或数据字典",),
        ),
        _db31(
            "050100",
            "基于时间段的正确性",
            "公共数据时效性",
            "X = A / B；A 为满足有效性要求的元素数，B 为被评价元素数",
            reason_code="missing_time_period_requirements",
            required_inputs=("有效日期范围、期望记录数或频率分布",),
            available_proxy_metric_ids=("time_info_availability",),
        ),
        _db31(
            "050200",
            "基于时间点的及时性",
            "公共数据时效性",
            "X = A / B；A 为满足及时性要求的元素数，B 为被评价元素数",
            reason_code="missing_timeliness_sla",
            required_inputs=("公开或提供时间、截止时间、更新频率或有效期阈值",),
            available_proxy_metric_ids=(
                "time_info_availability",
                "update_lag_days",
            ),
        ),
        _db31(
            "050300",
            "时序性",
            "公共数据时效性",
            "X = A / B；A 为满足时序性要求的元素数，B 为被评价元素数",
            reason_code="missing_temporal_order_rules",
            required_inputs=("实体键、参与比较的时间字段及先后规则",),
        ),
        _db31(
            "050400",
            "数据授权使用时效性",
            "公共数据时效性",
            "X = A / B；A 为满足授权使用时效性要求的元素数，B 为被评价元素数",
            reason_code="missing_authorization_context",
            required_inputs=("授权起止时间、授权对象及当前使用上下文",),
        ),
        _db31(
            "060100",
            "可访问",
            "公共数据可访问性",
            "X = A / B；A 为满足可访问性要求的元素数，B 为被评价元素数",
            reason_code="missing_access_sla",
            required_inputs=("访问服务、需要时间及访问 SLA",),
            available_proxy_metric_ids=("file_parse_rate",),
        ),
        _db31(
            "060200",
            "可用性",
            "公共数据可访问性",
            "X = A / B；A 为满足可用性要求的元素数，B 为被评价元素数",
            reason_code="missing_application_scenario",
            required_inputs=("目标应用场景及可用性验收标准",),
        ),
    )
)


METRIC_CATALOG: tuple[Mapping[str, Any], ...] = (
    *ORIGINAL_METRIC_CATALOG,
    *DB31_METRIC_CATALOG,
)
METRIC_BY_ID: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {str(item["id"]): item for item in METRIC_CATALOG}
)
ORIGINAL_METRIC_IDS: tuple[str, ...] = tuple(
    str(item["id"]) for item in ORIGINAL_METRIC_CATALOG
)
DB31_METRIC_IDS: tuple[str, ...] = tuple(
    str(item["id"]) for item in DB31_METRIC_CATALOG
)
ALL_METRIC_IDS: tuple[str, ...] = tuple(
    str(item["id"]) for item in METRIC_CATALOG
)
DEFAULT_SELECTED_METRIC_IDS: tuple[str, ...] = ORIGINAL_METRIC_IDS


# 目录指标只能接收与其语义相符的 Rule DSL 类型。这个映射是有意保持显式
# 的：规则引擎会为每条规则生成 ``business_*`` 审计指标，执行层不能根据
# 指标名称猜测映射关系，也不能把“必填”这类完整性规则投影到格式、时效
# 或一致性指标上。未列出的指标没有可直接投影的 DSL 类型，必须继续要求
# 用户提供标准依据或使用自定义业务规则。
_ALL_RULE_TYPES: frozenset[str] = frozenset(
    {
        "primary_key",
        "required",
        "update_freshness",
        "allowed_values",
        "numeric_range",
        "regex_format",
        "string_length",
        "conditional_required",
        "field_comparison",
    }
)
METRIC_RULE_TYPE_ALLOWLIST: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        # 数据标准的父级指标需要 010101/010102/010103 分项，不能由一条
        # 规则直接覆盖；数据类型和元数据本身也没有对应的本地 DSL。
        "db31_010100": frozenset(),
        "db31_010101": frozenset(),
        "db31_010102": frozenset(
            {"allowed_values", "numeric_range", "regex_format", "field_comparison"}
        ),
        "db31_010103": frozenset({"string_length"}),
        "db31_010200": frozenset({"primary_key", "field_comparison"}),
        "db31_010300": frozenset(),
        "db31_010400": _ALL_RULE_TYPES,
        "db31_010500": frozenset({"allowed_values", "field_comparison"}),
        "db31_010600": frozenset(),
        "db31_020100": frozenset({"required", "conditional_required"}),
        "db31_020200": frozenset(
            {"primary_key", "required", "conditional_required"}
        ),
        "db31_030100": frozenset(
            {
                "allowed_values",
                "numeric_range",
                "regex_format",
                "string_length",
                "field_comparison",
            }
        ),
        "db31_030200": frozenset(
            {"allowed_values", "numeric_range", "regex_format", "string_length"}
        ),
        "db31_030500": frozenset(
            {
                "allowed_values",
                "numeric_range",
                "regex_format",
                "string_length",
            }
        ),
        "db31_030600": frozenset({"allowed_values", "field_comparison"}),
        "db31_040100": frozenset({"field_comparison"}),
        "db31_040200": frozenset({"field_comparison"}),
        "db31_040201": frozenset({"field_comparison"}),
        "db31_040202": frozenset({"field_comparison"}),
        "db31_040203": frozenset({"field_comparison"}),
        "db31_040204": frozenset({"field_comparison"}),
        "db31_040300": frozenset(),
        "db31_050100": frozenset({"numeric_range", "field_comparison"}),
        "db31_050200": frozenset({"update_freshness"}),
        "db31_050300": frozenset({"field_comparison"}),
        "db31_050400": frozenset({"update_freshness"}),
        "db31_060100": frozenset(),
        "db31_060200": frozenset(),
    }
)


# 已有确定性计算逻辑的指标提供可编辑的默认评价依据；其余指标必须由
# 用户结合业务标准补充，避免把通用示例误当成实际评价规则。
DEFAULT_EVALUATION_BASES: Mapping[str, str] = MappingProxyType(
    {
        "file_parse_rate": "文件能够被当前解析器成功读取。",
        "dataset_scale": "以解析后的有效记录数作为数据规模。",
        "field_missing_rate": "按字段缺失值数除以该字段记录数计算字段缺失率。",
        "blank_record_rate": "内容字段全部为空的记录视为空白记录。",
        "field_type_consistency": "按字段非空值的基础类型推断，主要类型占比作为类型一致率。",
        "recognizable_format_anomaly_rate": (
            "按字段名识别日期、数值、URL 和邮箱字段，统计可识别的格式异常值。"
        ),
        "exact_duplicate_rate": (
            "排除技术标识列后，内容完全相同的后续记录视为重复记录。"
        ),
        "normalized_duplicate_rate": (
            "忽略文本空白、大小写和常见标点后，后续相同记录视为规范化重复记录。"
        ),
        "time_info_availability": "记录至少包含一个可解析的日期或时间字段。",
        "update_lag_days": "以评估基准日期减去最近可解析更新时间，计算更新滞后天数。",
        "source_info_coverage": (
            "包含可识别来源部门、链接或原始标识信息的记录计入来源信息覆盖率。"
        ),
        "version_info_coverage": (
            "包含版本、更新时间或处理记录信息的记录计入版本信息覆盖率。"
        ),
        "statistical_outlier_rate": "对被检查的数值字段使用 IQR 规则识别统计异常值。",
        "db31_030300": "按当前单表重复识别规则计算数据重复率。",
        "db31_030400": "按当前单表唯一性识别规则计算数据唯一性。",
    }
)


def get_metric_definition(metric_id: str) -> Mapping[str, Any] | None:
    """按稳定 ID 获取只读目录项。"""

    return METRIC_BY_ID.get(metric_id)


def allowed_rule_types_for_metric(metric_id: str) -> frozenset[str]:
    """返回目录指标允许直接投影的 Rule DSL 类型。"""

    return METRIC_RULE_TYPE_ALLOWLIST.get(metric_id, frozenset())


def metric_rule_type_error(metric_id: str, rule_type: str) -> str | None:
    """检查规则类型与目录指标语义是否兼容，返回可展示的错误。"""

    definition = get_metric_definition(metric_id)
    if definition is None:
        return f"指标目标引用了未知目录指标：{metric_id}。"
    allowed = allowed_rule_types_for_metric(metric_id)
    if rule_type in allowed:
        return None
    name = str(definition.get("name") or metric_id)
    if not allowed:
        return (
            f"规则类型“{rule_type}”不能直接绑定指标“{name}”；"
            "该指标当前没有可直接投影的本地 Rule DSL 类型。"
        )
    allowed_text = "、".join(sorted(allowed))
    return (
        f"规则类型“{rule_type}”不能绑定指标“{name}”；"
        f"该指标只允许：{allowed_text}。"
    )


def metric_description(metric_id: str) -> str:
    """返回指标在页面提示中使用的简明含义。"""

    return str(METRIC_BY_ID[metric_id]["description"])


def default_evaluation_basis(metric_id: str) -> str:
    """返回已有确定性指标的默认评价依据，没有则返回空字符串。"""

    basis = str(DEFAULT_EVALUATION_BASES.get(metric_id, ""))
    return f"默认：{basis}" if basis else ""


def normalize_selected_metric_ids(
    selected_metric_ids: Iterable[str] | None,
) -> tuple[str, ...]:
    """验证选择并按目录顺序规范化，保证报告和哈希可复现。

    ``None`` 保持 v0.4 兼容行为，默认启用原 13 项指标。显式空选择没有
    评价意义，因此被拒绝。重复 ID 会去重，输出顺序不受用户点击顺序影响。
    """

    if selected_metric_ids is None:
        return DEFAULT_SELECTED_METRIC_IDS
    if isinstance(selected_metric_ids, (str, bytes)):
        raise MetricSelectionError("指标选择必须是指标 ID 集合，不能是单个字符串。")

    requested: list[str] = []
    for metric_id in selected_metric_ids:
        if not isinstance(metric_id, str) or not metric_id:
            raise MetricSelectionError("每个指标 ID 都必须是非空字符串。")
        requested.append(metric_id)
    if not requested:
        raise MetricSelectionError("请至少选择一个评价指标。")

    unknown = sorted(set(requested) - set(ALL_METRIC_IDS))
    if unknown:
        raise MetricSelectionError(
            "包含未知指标 ID：" + "、".join(unknown)
        )
    requested_set = set(requested)
    return tuple(
        metric_id
        for metric_id in ALL_METRIC_IDS
        if metric_id in requested_set
    )


def metric_selection_label(metric_id: str) -> str:
    """生成页面选择控件使用的无歧义标签。"""

    item = METRIC_BY_ID[metric_id]
    if item["source"] == LEGACY_SOURCE:
        return (
            f"[原 v0.4] {item['category']} / "
            f"{item['name']} ({metric_id})"
        )
    capability = "可直接计算" if item["auto_assessable"] else "需补充评价依据"
    return (
        f"[DB31/T {item['standard_code']}] {item['dimension']} / "
        f"{item['name']} · {capability}"
    )


def build_metric_catalog_rows() -> list[dict[str, str]]:
    """生成供页面展示的指标目录，不暴露内部可变对象。"""

    rows: list[dict[str, str]] = []
    for item in METRIC_CATALOG:
        rows.append(
            {
                "来源": str(item["source_label"]),
                "标准代码": str(item["standard_code"] or "—"),
                "一级维度": str(item["dimension"]),
                "层级": str(item["level"]),
                "指标名称": str(item["name"]),
                "指标含义": str(item["description"]),
                "计算方式": str(item["formula"]),
                "当前能力": (
                    "可直接计算"
                    if item["auto_assessable"]
                    else "缺少评价依据时标记为无法评估"
                ),
                "指标 ID": str(item["id"]),
            }
        )
    return rows
