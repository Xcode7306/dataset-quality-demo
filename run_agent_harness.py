"""Run the v0.9.1 rule-authoring and RAG golden suites."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from src.rule_authoring_harness import (
    compare_rule_authoring_providers,
    run_rag_retrieval_harness,
)
from src.rule_authoring_prompts import list_rule_authoring_prompts
from src.rule_authoring_providers import (
    DeepSeekRuleAuthoringProvider,
    OpenAICompatibleRuleAuthoringProvider,
    TemplateRuleAuthoringProvider,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def _provider(name: str):
    if name == "template":
        return TemplateRuleAuthoringProvider()
    if name == "deepseek":
        return DeepSeekRuleAuthoringProvider()
    if name == "custom":
        api_url = os.environ.get("HARNESS_MODEL_API_URL", "").strip()
        api_key = os.environ.get("HARNESS_MODEL_API_KEY", "").strip()
        model = os.environ.get("HARNESS_MODEL_NAME", "").strip()
        if not api_url or not api_key or not model:
            raise ValueError(
                "custom Provider 需要 HARNESS_MODEL_API_URL、"
                "HARNESS_MODEL_API_KEY 和 HARNESS_MODEL_NAME。"
            )
        return OpenAICompatibleRuleAuthoringProvider(
            api_url=api_url,
            api_key=api_key,
            model=model,
        )
    raise ValueError(f"未知 Harness Provider：{name}。")


def _atomic_json_write(path: Path, payload: object) -> None:
    if path.suffix.casefold() != ".json":
        raise ValueError("Harness 输出文件必须使用 .json 扩展名。")
    parent = path.resolve().parent
    if not parent.is_dir():
        raise ValueError("Harness 输出目录不存在。")
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8", errors="strict")
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(path.resolve())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行 v0.9.1 Agent Harness 规则编译、RAG 和重放回归。"
    )
    parser.add_argument(
        "--provider",
        action="append",
        choices=("template", "deepseek", "custom"),
        help="可重复指定以对比 Provider；默认仅运行 template。",
    )
    parser.add_argument(
        "--no-replay",
        action="store_true",
        help="仅运行一次每个规则案例，跳过重复执行一致性检查。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="可选写入包含逐案轨迹的严格 JSON。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    names = args.provider or ["template"]
    if len(names) != len(set(names)):
        parser.error("--provider 不能重复指定同一值。")
    try:
        providers = {name: _provider(name) for name in names}
        rule_reports = compare_rule_authoring_providers(
            PROJECT_ROOT,
            providers,
            replay=not args.no_replay,
        )
        rag_report = run_rag_retrieval_harness(PROJECT_ROOT)
        payload = {
            "schema_version": "0.1",
            "prompt_registry": list(list_rule_authoring_prompts()),
            "rule_authoring": [
                report.to_dict(include_traces=True) for report in rule_reports
            ],
            "rag_retrieval": rag_report.to_dict(include_traces=True),
            "passed": all(report.passed for report in rule_reports)
            and rag_report.passed,
        }
        if args.output is not None:
            _atomic_json_write(args.output, payload)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.exit(1, f"Agent Harness 失败：{error}\n")

    summary = {
        "schema_version": payload["schema_version"],
        "rule_authoring": [
            report.to_dict(include_traces=False) for report in rule_reports
        ],
        "rag_retrieval": rag_report.to_dict(include_traces=False),
        "passed": payload["passed"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
