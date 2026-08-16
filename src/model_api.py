"""兼容 OpenAI Chat Completions 形态的模型 API 小工具。"""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any, Callable, Mapping


def normalize_chat_completions_url(value: str) -> str:
    """将用户输入的 API 地址规范为 Chat Completions 请求地址。"""

    endpoint = str(value or "").strip().rstrip("/")
    if not endpoint:
        return ""
    if endpoint.endswith("/chat/completions"):
        return endpoint
    return f"{endpoint}/chat/completions"


def make_chat_completions_client(
    *,
    timeout_seconds: float,
    client_factory: Callable[..., Any] | None = None,
) -> Any:
    """Create a verified HTTP client that honors normal proxy configuration.

    Injected factories intentionally receive only ``timeout`` so existing local
    tests and provider adapters do not need to emulate httpx-specific options.
    The production client explicitly keeps certificate verification enabled and
    reads standard proxy/no-proxy environment variables.
    """

    if client_factory is not None:
        return client_factory(timeout=timeout_seconds)
    try:
        import httpx
    except ImportError as error:
        raise RuntimeError("未安装 httpx。") from error
    return httpx.Client(
        timeout=timeout_seconds,
        trust_env=True,
        verify=True,
    )


def secret_fingerprint(value: str) -> str:
    """返回不包含原始密钥的短指纹，用于区分进程内缓存命名空间。"""

    return sha256(str(value or "").encode("utf-8")).hexdigest()[:16]


def extract_message_content(value: Any) -> str:
    """兼容字符串、内容块数组和 completion 文本三种常见响应。"""

    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("text", "content", "output_text", "value"):
            nested = value.get(key)
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
        return ""
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = extract_message_content(item)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    return ""


def parse_json_object_text(value: Any) -> Mapping[str, Any] | None:
    """从 JSON、代码围栏或带少量说明文字的模型输出中提取对象。"""

    text = extract_message_content(value)
    if not text:
        return None
    candidates = [text]
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidates.insert(0, "\n".join(lines).strip())
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, Mapping):
            return parsed
        start = candidate.find("{")
        if start >= 0:
            try:
                parsed, _ = decoder.raw_decode(candidate[start:])
            except (TypeError, ValueError):
                continue
            if isinstance(parsed, Mapping):
                return parsed
    return None


def response_error_detail(response: Any, *, maximum: int = 500) -> str:
    """提取不包含请求凭据的简短 HTTP 错误信息。"""

    status_code = getattr(response, "status_code", None)
    detail = ""
    try:
        body = response.json()
    except Exception:
        body = None
    if isinstance(body, Mapping):
        error = body.get("error")
        if isinstance(error, Mapping):
            detail = str(error.get("message") or error.get("detail") or "")
        elif isinstance(error, str):
            detail = error
        if not detail:
            detail = str(body.get("message") or body.get("detail") or "")
    if not detail:
        raw_text = getattr(response, "text", "")
        if isinstance(raw_text, str):
            detail = raw_text.strip()
    detail = " ".join(detail.split())[:maximum]
    if status_code is None:
        return detail or "模型 API 返回了无法识别的错误。"
    return (
        f"HTTP {status_code}"
        + (f"：{detail}" if detail else "：未提供错误详情")
    )


__all__ = [
    "extract_message_content",
    "make_chat_completions_client",
    "normalize_chat_completions_url",
    "parse_json_object_text",
    "response_error_detail",
    "secret_fingerprint",
]
