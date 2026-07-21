"""字段名语义识别规则。

英文关键词只按字段 token 匹配；下划线、连字符、空格和驼峰边界均视为
token 边界。中文关键词仍按字段名中的连续文本匹配。
"""

from collections.abc import Iterable
import re


DATE_FIELD_PATTERN = re.compile(
    r"(?:^|[_\s-])(?:date|time|datetime|timestamp)(?:$|[_\s-])|"
    r"(?:^|[_\s-])(?:created|updated|modified|published)"
    r"(?:[_\s-]?(?:at|date|time|datetime|timestamp))?$|"
    r"(?:日期|时间)",
    re.IGNORECASE,
)
UPDATE_FIELD_PATTERN = re.compile(
    r"(?:^|[_\s-])(?:update|updated|modified)"
    r"(?:[_\s-]?(?:date|time|datetime|timestamp|at))?$|"
    r"(?:^|[_\s-])last[_\s-]?updated(?:[_\s-]?at)?$|"
    r"(?:更新日期|更新时间|最后更新|修改日期|修改时间|修订日期)$",
    re.IGNORECASE,
)
URL_FIELD_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:url|uri|link)(?![A-Za-z0-9])|(?:网址|链接)",
    re.IGNORECASE,
)
EMAIL_FIELD_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:email|e-mail)(?![A-Za-z0-9])|(?:邮箱|邮件)",
    re.IGNORECASE,
)
NUMERIC_FIELD_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:amount|count|number|quantity|total|rate|ratio|score|"
    r"age|days|hours)(?![A-Za-z0-9])|"
    r"(?:金额|数量|次数|人数|天数|时长|比例|比率|分数|年龄)",
    re.IGNORECASE,
)
SOURCE_FIELD_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:source|origin|publisher|department|provider)"
    r"(?![A-Za-z0-9])|(?:来源|发布部门|发布单位|部门|供应单位)",
    re.IGNORECASE,
)
SOURCE_IDENTIFIER_FIELD_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:source|origin|original)[_\s-]?id(?![A-Za-z0-9])|"
    r"(?:原始(?:标识|编号|编码)|来源(?:标识|编号|编码))",
    re.IGNORECASE,
)
VERSION_FIELD_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:version|revision|release|processed|workflow)"
    r"(?![A-Za-z0-9])|(?:版本|修订|修订日期|处理记录|处理时间)",
    re.IGNORECASE,
)
STRUCTURED_TEXT_FIELD_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:url|uri|link|email|e-mail|version|revision|code|number)"
    r"(?![A-Za-z0-9])|(?:网址|链接|邮箱|邮件|版本|修订|编号|编码|代码)",
    re.IGNORECASE,
)


_CAMEL_CASE_BOUNDARY_PATTERN = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
)


def normalize_field_name(field: object) -> str:
    """为语义匹配展开驼峰边界，同时保留原始分隔符和中文文本。"""

    return _CAMEL_CASE_BOUNDARY_PATTERN.sub(" ", str(field).strip())


def field_matches(pattern: re.Pattern[str], field: object) -> bool:
    """判断字段名是否匹配给定语义规则。"""

    return bool(pattern.search(normalize_field_name(field)))


def identify_semantic_fields(columns: Iterable[object]) -> dict[str, list[str]]:
    """返回画像和指标共同使用的五类可识别字段。

    返回值固定包含 ``date``、``numeric``、``url``、``source`` 和
    ``version`` 五个键；每个列表保持输入字段顺序并去重。
    """

    fields = list(dict.fromkeys(str(column) for column in columns))

    def matching(*patterns: re.Pattern[str]) -> list[str]:
        return [
            field
            for field in fields
            if any(field_matches(pattern, field) for pattern in patterns)
        ]

    url_fields = matching(URL_FIELD_PATTERN)
    email_fields = set(matching(EMAIL_FIELD_PATTERN))
    date_fields = [
        field
        for field in matching(DATE_FIELD_PATTERN, UPDATE_FIELD_PATTERN)
        if field not in url_fields and field not in email_fields
    ]
    source_fields = matching(
        SOURCE_FIELD_PATTERN,
        URL_FIELD_PATTERN,
        SOURCE_IDENTIFIER_FIELD_PATTERN,
    )
    version_fields = matching(VERSION_FIELD_PATTERN, UPDATE_FIELD_PATTERN)

    return {
        "date": date_fields,
        "numeric": matching(NUMERIC_FIELD_PATTERN),
        "url": url_fields,
        "source": source_fields,
        "version": version_fields,
    }
