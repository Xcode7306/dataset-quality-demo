"""跨输入入口复用的 Unicode 展示文本规范化工具。"""


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
