"""Bounded, session-only rule-chat state helpers.

Streamlit session state is deliberately kept separate from domain workflow
objects.  These helpers retain a small recent window for UI continuity while
dropping malformed or oversized data before it can grow without bound.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .text_utils import contains_unsafe_unicode_controls


MAX_PRE_EVALUATION_CHAT_MESSAGES = 20
MAX_PRE_EVALUATION_CHAT_ATTACHMENTS = 5
MAX_PRE_EVALUATION_CHAT_ATTACHMENT_BYTES = 2 * 1024 * 1024
MAX_PRE_EVALUATION_CHAT_MESSAGE_LENGTH = 4000
MAX_PRE_EVALUATION_CHAT_RULE_TEXT_LENGTH = 4000


def bounded_rule_chat_state(
    messages: Iterable[object],
    attachments: Iterable[object],
) -> tuple[list[dict[str, str]], list[dict[str, bytes]], bool]:
    """Return safe recent chat state and whether anything was discarded.

    Attachment bytes are never decoded here.  They are bounded only by count
    and total bytes, then parsed by the existing allowlisted import service.
    """

    clean_messages: list[dict[str, str]] = []
    discarded = False
    for item in messages:
        if not isinstance(item, Mapping):
            discarded = True
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            discarded = True
            continue
        normalized = content.strip()
        if (
            not normalized
            or len(normalized) > MAX_PRE_EVALUATION_CHAT_MESSAGE_LENGTH
            or contains_unsafe_unicode_controls(normalized)
        ):
            discarded = True
            continue
        normalized_item = {"role": role, "content": normalized}
        if item.get("kind") == "attachment":
            # The batch parser must not treat the human-readable attachment
            # receipt as another natural-language rule description.
            normalized_item["kind"] = "attachment"
        clean_messages.append(normalized_item)
    if len(clean_messages) > MAX_PRE_EVALUATION_CHAT_MESSAGES:
        clean_messages = clean_messages[-MAX_PRE_EVALUATION_CHAT_MESSAGES:]
        discarded = True
    # All user turns are merged into one RuleBatchInput.  Keep that aggregate
    # within the downstream rule-input contract as well as limiting each turn.
    retained_reversed: list[dict[str, str]] = []
    user_text_length = 0
    for item in reversed(clean_messages):
        is_rule_turn = item["role"] == "user" and item.get("kind") != "attachment"
        additional_length = len(item["content"]) + (1 if user_text_length else 0)
        if (
            is_rule_turn
            and user_text_length + additional_length
            > MAX_PRE_EVALUATION_CHAT_RULE_TEXT_LENGTH
        ):
            discarded = True
            continue
        retained_reversed.append(item)
        if is_rule_turn:
            user_text_length += additional_length
    clean_messages = list(reversed(retained_reversed))

    clean_attachments_reversed: list[dict[str, bytes]] = []
    total_bytes = 0
    for item in reversed(list(attachments)):
        if not isinstance(item, Mapping):
            discarded = True
            continue
        name = item.get("name")
        content = item.get("content")
        if not isinstance(name, str) or not isinstance(content, (bytes, bytearray)):
            discarded = True
            continue
        normalized_name = name.strip()
        payload = bytes(content)
        if (
            not normalized_name
            or contains_unsafe_unicode_controls(normalized_name)
            or len(payload) > MAX_PRE_EVALUATION_CHAT_ATTACHMENT_BYTES
            or len(clean_attachments_reversed) >= MAX_PRE_EVALUATION_CHAT_ATTACHMENTS
            or total_bytes + len(payload) > MAX_PRE_EVALUATION_CHAT_ATTACHMENT_BYTES
        ):
            discarded = True
            continue
        clean_attachments_reversed.append(
            {"name": normalized_name[:255], "content": payload}
        )
        total_bytes += len(payload)
    clean_attachments = list(reversed(clean_attachments_reversed))
    return clean_messages, clean_attachments, discarded


__all__ = [
    "MAX_PRE_EVALUATION_CHAT_ATTACHMENTS",
    "MAX_PRE_EVALUATION_CHAT_ATTACHMENT_BYTES",
    "MAX_PRE_EVALUATION_CHAT_MESSAGES",
    "MAX_PRE_EVALUATION_CHAT_MESSAGE_LENGTH",
    "MAX_PRE_EVALUATION_CHAT_RULE_TEXT_LENGTH",
    "bounded_rule_chat_state",
]
