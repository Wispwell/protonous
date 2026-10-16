"""The only code that speaks an OpenAI/OpenRouter shape.

- normalize_in:       our ContextWindow -> the SDK `messages` list (pure flatten).
- normalize_tools:    our ToolSpecs -> the SDK `tools` list.
- normalize_chunk:    one streaming SDK delta -> our MessageChunk.

Reasoning round-trips: assistant messages (all of them, or only the latest — the
caller's choice) carry their `reasoning_details` back out,
in original order and unmodified, as OpenRouter requires to keep reasoning across
tool-calling turns. If a provider returned only a raw `reasoning` string, that goes
back instead. OpenRouter's `reasoning` / `reasoning_details` are provider extensions,
so they're read defensively.
"""
from collections.abc import Sequence
from typing import Any

from .context import (
    MessageChunk,
    ReasoningDetail,
    Role,
    ToolCall,
    ToolSpec,
    Usage,
)
from .layers import ContextWindow


def normalize_in(
    context: ContextWindow, *, replay_all_reasoning: bool = True
) -> list[dict[str, Any]]:
    """Flatten the window (layers in order, messages in order) into SDK messages.

    Reasoning goes back on every assistant message, or, with
    `replay_all_reasoning=False`, only on the most recent one.
    """
    view = context.view()
    last_assistant = max(
        (i for i, msg in enumerate(view) if msg.role is Role.ASSISTANT), default=None
    )
    messages: list[dict[str, Any]] = []
    for i, msg in enumerate(view):
        entry: dict[str, Any] = {"role": msg.role.value, "content": msg.content}
        if msg.role is Role.ASSISTANT and msg.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function_name, "arguments": tc.function_arguments},
                }
                for tc in msg.tool_calls
            ]
        if msg.role is Role.ASSISTANT and (replay_all_reasoning or i == last_assistant):
            if msg.reasoning_details:
                entry["reasoning_details"] = [
                    rd.model_dump(exclude_none=True) for rd in msg.reasoning_details
                ]
            elif msg.reasoning is not None:
                entry["reasoning"] = msg.reasoning
        if msg.role is Role.TOOL and msg.tool_call_id is not None:
            entry["tool_call_id"] = msg.tool_call_id
        messages.append(entry)
    return messages


def normalize_tools(specs: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    """Our ToolSpecs -> the SDK `tools` list (function tools only)."""
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }
        for spec in specs
    ]


def _to_tool_calls(raw: Any) -> tuple[ToolCall, ...] | None:
    if not raw:
        return None
    calls: list[ToolCall] = []
    for tc in raw:
        fn = getattr(tc, "function", None)
        calls.append(
            ToolCall(
                id=getattr(tc, "id", None) or "",
                type=getattr(tc, "type", None) or "function",
                index=getattr(tc, "index", None),
                function_name=(getattr(fn, "name", None) or "") if fn is not None else "",
                function_arguments=(getattr(fn, "arguments", None) or "") if fn is not None else "",
            )
        )
    return tuple(calls)


def _get(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _to_reasoning_details(raw: Any) -> tuple[ReasoningDetail, ...] | None:
    if not raw:
        return None
    return tuple(
        ReasoningDetail(
            type=_get(rd, "type") or "",
            id=_get(rd, "id"),
            format=_get(rd, "format"),
            index=_get(rd, "index"),
            summary=_get(rd, "summary"),
            data=_get(rd, "data"),
            text=_get(rd, "text"),
            signature=_get(rd, "signature"),
        )
        for rd in raw
    )


def _to_usage(raw: Any) -> Usage | None:
    if raw is None:
        return None
    ctd = getattr(raw, "completion_tokens_details", None)
    ptd = getattr(raw, "prompt_tokens_details", None)
    return Usage(
        prompt_tokens=getattr(raw, "prompt_tokens", None),
        completion_tokens=getattr(raw, "completion_tokens", None),
        total_tokens=getattr(raw, "total_tokens", None),
        reasoning_tokens=getattr(ctd, "reasoning_tokens", None) if ctd is not None else None,
        cached_tokens=getattr(ptd, "cached_tokens", None) if ptd is not None else None,
    )


def normalize_chunk(chunk: Any) -> MessageChunk:
    choices = getattr(chunk, "choices", None) or []
    choice = choices[0] if choices else None
    delta = getattr(choice, "delta", None) if choice is not None else None
    role = _get(delta, "role") if delta is not None else None
    return MessageChunk(
        content=_get(delta, "content") if delta is not None else None,
        role=Role(role) if role else None,
        reasoning=_get(delta, "reasoning") if delta is not None else None,
        reasoning_details=(
            _to_reasoning_details(_get(delta, "reasoning_details")) if delta is not None else None
        ),
        tool_calls=_to_tool_calls(_get(delta, "tool_calls")) if delta is not None else None,
        stop_reason=getattr(choice, "finish_reason", None) if choice is not None else None,
        usage=_to_usage(getattr(chunk, "usage", None)),
        provider_id=getattr(chunk, "id", None),
    )
