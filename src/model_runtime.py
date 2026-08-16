"""进程内模型调用的有界并发、限流和用量审计。

这个 Demo 没有服务端队列或跨进程存储，因此本模块只协调同一 Python
进程内的外部模型请求。它不保存 API Key、提示词、原始数据或模型文本；
仅保留有限的 provider/model 汇总计数，供页面状态和自动化测试使用。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
import re
from threading import RLock
from time import monotonic
from typing import Deque


DEFAULT_MAX_CONCURRENT_MODEL_REQUESTS = 4
DEFAULT_MAX_MODEL_REQUESTS_PER_MINUTE = 120
MAX_TRACKED_MODEL_IDENTITIES = 64
_RATE_WINDOW_SECONDS = 60.0
_SENSITIVE_LABEL_PATTERN = re.compile(
    r"(?i)(api[_ -]?key|authorization|access[_ -]?token|bearer|\bsk-[a-z0-9_-]{6,})"
)


class ModelRuntimeLimitError(RuntimeError):
    """A bounded local model-runtime guard rejected a request."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ModelRuntimeUsage:
    """Safe, aggregate usage for one provider/model identity."""

    provider: str
    model: str
    attempts: int
    successes: int
    failures: int
    rejected: int
    input_tokens: int
    output_tokens: int
    total_latency_ms: int
    active_requests: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "provider": self.provider,
            "model": self.model,
            "attempts": self.attempts,
            "successes": self.successes,
            "failures": self.failures,
            "rejected": self.rejected,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_latency_ms": self.total_latency_ms,
            "active_requests": self.active_requests,
        }


@dataclass
class _MutableUsage:
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    rejected: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_latency_ms: int = 0
    active_requests: int = 0
    touched_at: float = 0.0


def _positive_environment_limit(name: str, default: int, *, maximum: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return parsed if 1 <= parsed <= maximum else default


def _safe_identity_part(value: object, *, fallback: str, maximum: int = 100) -> str:
    text = " ".join(str(value or "").split())[:maximum]
    if not text:
        return fallback
    if _SENSITIVE_LABEL_PATTERN.search(text):
        return "[redacted]"
    return text


def _bounded_non_negative(value: object, *, maximum: int = 10_000_000) -> int:
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return min(max(0, parsed), maximum)


class ModelRequestLease:
    """One accepted request slot. ``finish`` is idempotent and thread-safe."""

    def __init__(self, runtime: "_ModelRuntime", key: tuple[str, str]) -> None:
        self._runtime = runtime
        self._key = key
        self._finished = False

    def finish(
        self,
        *,
        success: bool,
        input_tokens: object = 0,
        output_tokens: object = 0,
        latency_ms: object = 0,
    ) -> None:
        if self._finished:
            return
        self._finished = True
        self._runtime.finish(
            self._key,
            success=success,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )


class _ModelRuntime:
    def __init__(self) -> None:
        self._lock = RLock()
        self._max_concurrent = _positive_environment_limit(
            "QUALITY_DEMO_MAX_CONCURRENT_MODEL_REQUESTS",
            DEFAULT_MAX_CONCURRENT_MODEL_REQUESTS,
            maximum=64,
        )
        self._requests_per_minute = _positive_environment_limit(
            "QUALITY_DEMO_MAX_MODEL_REQUESTS_PER_MINUTE",
            DEFAULT_MAX_MODEL_REQUESTS_PER_MINUTE,
            maximum=10_000,
        )
        self._active_requests = 0
        self._accepted_starts: Deque[float] = deque()
        self._usage: dict[tuple[str, str], _MutableUsage] = {}

    def _identity(self, provider: object, model: object) -> tuple[str, str]:
        return (
            _safe_identity_part(provider, fallback="custom"),
            _safe_identity_part(model, fallback="unknown"),
        )

    def _usage_for(self, key: tuple[str, str], now: float) -> _MutableUsage:
        usage = self._usage.get(key)
        if usage is not None:
            usage.touched_at = now
            return usage
        if len(self._usage) >= MAX_TRACKED_MODEL_IDENTITIES:
            inactive = [
                item
                for item in self._usage.items()
                if item[1].active_requests == 0
            ]
            if inactive:
                oldest_key, _ = min(inactive, key=lambda item: item[1].touched_at)
                self._usage.pop(oldest_key, None)
            else:
                key = ("other", "other")
                usage = self._usage.get(key)
                if usage is not None:
                    usage.touched_at = now
                    return usage
        usage = _MutableUsage(touched_at=now)
        self._usage[key] = usage
        return usage

    def _discard_expired_starts(self, now: float) -> None:
        cutoff = now - _RATE_WINDOW_SECONDS
        while self._accepted_starts and self._accepted_starts[0] <= cutoff:
            self._accepted_starts.popleft()

    def acquire(self, provider: object, model: object) -> ModelRequestLease:
        now = monotonic()
        key = self._identity(provider, model)
        with self._lock:
            self._discard_expired_starts(now)
            usage = self._usage_for(key, now)
            if self._active_requests >= self._max_concurrent:
                usage.rejected += 1
                usage.failures += 1
                raise ModelRuntimeLimitError(
                    "当前模型请求较多，请稍后重试。",
                    code="concurrency_limited",
                )
            if len(self._accepted_starts) >= self._requests_per_minute:
                usage.rejected += 1
                usage.failures += 1
                raise ModelRuntimeLimitError(
                    "模型请求达到本进程每分钟上限，请稍后重试。",
                    code="rate_limited",
                )
            self._active_requests += 1
            usage.active_requests += 1
            usage.attempts += 1
            self._accepted_starts.append(now)
        return ModelRequestLease(self, key)

    def finish(
        self,
        key: tuple[str, str],
        *,
        success: bool,
        input_tokens: object,
        output_tokens: object,
        latency_ms: object,
    ) -> None:
        now = monotonic()
        with self._lock:
            usage = self._usage_for(key, now)
            self._active_requests = max(0, self._active_requests - 1)
            usage.active_requests = max(0, usage.active_requests - 1)
            if success:
                usage.successes += 1
            else:
                usage.failures += 1
            usage.input_tokens += _bounded_non_negative(input_tokens)
            usage.output_tokens += _bounded_non_negative(output_tokens)
            usage.total_latency_ms += _bounded_non_negative(latency_ms)

    def snapshot(self) -> tuple[ModelRuntimeUsage, ...]:
        with self._lock:
            return tuple(
                ModelRuntimeUsage(
                    provider=provider,
                    model=model,
                    attempts=usage.attempts,
                    successes=usage.successes,
                    failures=usage.failures,
                    rejected=usage.rejected,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    total_latency_ms=usage.total_latency_ms,
                    active_requests=usage.active_requests,
                )
                for (provider, model), usage in sorted(self._usage.items())
            )

    def reset_for_tests(
        self,
        *,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT_MODEL_REQUESTS,
        requests_per_minute: int = DEFAULT_MAX_MODEL_REQUESTS_PER_MINUTE,
    ) -> None:
        if not 1 <= max_concurrent <= 64:
            raise ValueError("max_concurrent 必须在 1 到 64 之间。")
        if not 1 <= requests_per_minute <= 10_000:
            raise ValueError("requests_per_minute 必须在 1 到 10000 之间。")
        with self._lock:
            self._max_concurrent = max_concurrent
            self._requests_per_minute = requests_per_minute
            self._active_requests = 0
            self._accepted_starts.clear()
            self._usage.clear()


_RUNTIME = _ModelRuntime()


def acquire_model_request(provider: object, model: object) -> ModelRequestLease:
    """Acquire one bounded external-model request slot for this process."""

    return _RUNTIME.acquire(provider, model)


def model_runtime_snapshot() -> tuple[ModelRuntimeUsage, ...]:
    """Return safe aggregate provider/model usage without prompts or secrets."""

    return _RUNTIME.snapshot()


def reset_model_runtime_for_tests(
    *,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT_MODEL_REQUESTS,
    requests_per_minute: int = DEFAULT_MAX_MODEL_REQUESTS_PER_MINUTE,
) -> None:
    """Reset process-local guard state; intentionally limited to test support."""

    _RUNTIME.reset_for_tests(
        max_concurrent=max_concurrent,
        requests_per_minute=requests_per_minute,
    )


__all__ = [
    "DEFAULT_MAX_CONCURRENT_MODEL_REQUESTS",
    "DEFAULT_MAX_MODEL_REQUESTS_PER_MINUTE",
    "MAX_TRACKED_MODEL_IDENTITIES",
    "ModelRequestLease",
    "ModelRuntimeLimitError",
    "ModelRuntimeUsage",
    "acquire_model_request",
    "model_runtime_snapshot",
    "reset_model_runtime_for_tests",
]
