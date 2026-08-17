"""跨输入入口复用的 Unicode 展示文本规范化和安全检查工具。"""

from __future__ import annotations

import unicodedata


_BIDI_OR_ISOLATE_CONTROLS = frozenset(
    {
        "\u061c",  # Arabic letter mark
        "\u200e",  # left-to-right mark
        "\u200f",  # right-to-left mark
        "\u202a",  # embedding/override controls
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2066",  # isolate controls
        "\u2067",
        "\u2068",
        "\u2069",
    }
)


def contains_unsafe_unicode_controls(value: object) -> bool:
    """Whether text contains controls unsafe for untrusted UI/API input.

    Normal line breaks and tabs remain allowed for natural-language input.  We
    reject other C0/C1 controls, surrogate code points, and bidirectional
    display controls that can make a reviewer and an API see different text.
    """

    for character in str(value):
        if character in {"\n", "\r", "\t"}:
            continue
        if character in _BIDI_OR_ISOLATE_CONTROLS:
            return True
        if unicodedata.category(character) in {"Cc", "Cs"}:
            return True
    return False


def replace_unsafe_unicode_controls(value: object, *, replacement: str = "�") -> str:
    """Replace unsafe controls while retaining ordinary text and line breaks."""

    result: list[str] = []
    for character in str(value):
        unsafe = (
            character in _BIDI_OR_ISOLATE_CONTROLS
            or (
                character not in {"\n", "\r", "\t"}
                and unicodedata.category(character) in {"Cc", "Cs"}
            )
        )
        result.append(replacement if unsafe else character)
    return "".join(result)


def normalize_display_text(value: object) -> tuple[str, bool]:
    """规范合法代理对，并替换无法用 UTF-8 表示的孤立代理码位。

    返回规范化文本，以及是否发生过替换。调用方可据此记录 warning。
    """

    text = str(value)
    encoded = text.encode("utf-16", errors="surrogatepass")
    try:
        return encoded.decode("utf-16"), False
    except UnicodeDecodeError:
        return encoded.decode("utf-16", errors="replace"), True


__all__ = [
    "contains_unsafe_unicode_controls",
    "normalize_display_text",
    "replace_unsafe_unicode_controls",
]
