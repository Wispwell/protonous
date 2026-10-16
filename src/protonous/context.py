"""Core context types for protonous.

Provider-agnostic message/context model. The ONLY code that ever speaks an
OpenAI/OpenRouter shape is `normalizers.py`; everything in here is ours.

Design:
- Message / ToolCall / ReasoningDetail / Usage / MessageChunk are immutable (frozen).
- Context assembly (Layers, ContextWindow) lives in `layers.py`; this module holds
  the data types, plus `check_pairing`, the integrity rule for tool-call traffic.
- Token usage is only ever what the provider reports (`Usage` on assistant
  messages); protonous never counts tokens itself.
- System prompts and rules use the `system` role (open-weight chat templates key on it).
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .errors import ToolPairingError


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ReasoningDetail(BaseModel):
    """One structured reasoning block, per OpenRouter's reasoning_details[].

    Fields are type-dependent: `type` is one of reasoning.summary|encrypted|text;
    summary uses `summary`, encrypted uses `data`, text uses `text` (+ `signature`).
    """

    model_config = ConfigDict(frozen=True)

    type: str
    id: str | None = None
    format: str | None = None
    index: int | None = None
    summary: str | None = None
    data: str | None = None
    text: str | None = None
    signature: str | None = None


class ToolCall(BaseModel):
    """A single function tool call.

    `function_arguments` is the raw JSON string exactly as the provider sends it —
    possibly a partial fragment mid-stream. `id`/`function_name` default to "" so
    streaming fragments (which carry only `index` + an argument slice) are
    representable; a fully-merged call has them filled.
    """

    model_config = ConfigDict(frozen=True)

    id: str = ""
    type: str = "function"
    index: int | None = None
    function_name: str = ""
    function_arguments: str = ""

    @property
    def parsed_arguments(self) -> dict[str, Any]:
        """json.loads the arguments. Only valid once the call is complete."""
        parsed: dict[str, Any] = json.loads(self.function_arguments)
        return parsed


class Usage(BaseModel):
    """Token accounting for one response, exactly as the provider reported it.

    `prompt_tokens` is the whole input of that request (template and tool schemas
    included), of which `cached_tokens` were served from cache and
    `cache_write_tokens` written to it; `completion_tokens` includes
    `reasoning_tokens`. `cost` is in OpenRouter credits (USD). A field the provider
    didn't report stays None.

    `+` adds two usages field by field (None counts as nothing), so totals over
    many responses are `sum(usages, Usage())`.
    """

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None
    cost: float | None = None

    def __add__(self, other: Usage) -> Usage:
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            **{
                name: _add_optional(getattr(self, name), getattr(other, name))
                for name in Usage.model_fields
            }
        )


def _add_optional(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return a + b


class Message(BaseModel):
    """An immutable message. `tool_calls` is empty except on assistant messages;
    `tool_call_id` is set only on tool-result messages (role == TOOL)."""

    model_config = ConfigDict(frozen=True)

    role: Role
    content: str | None
    id: str = Field(default_factory=lambda: str(uuid4()))
    provider_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    usage: Usage | None = None  # set on assistant replies
    reasoning: str | None = None
    reasoning_details: tuple[ReasoningDetail, ...] | None = None
    stop_reason: str | None = None
    # Design choice: tool_calls is a tuple — a hard-immutable sequence, fitting a
    # frozen model. extras stays a dict because a mapping can't be a tuple; it's the
    # one intentionally mutable-typed field (frozen blocks field reassignment, not
    # in-place mutation of the dict's contents).
    tool_calls: tuple[ToolCall, ...] = Field(default_factory=tuple)
    tool_call_id: str | None = None
    extras: dict[str, Any] = Field(default_factory=dict)


def _concat(a: str | None, b: str | None) -> str | None:
    if a is None and b is None:
        return None
    return (a or "") + (b or "")


def _merge_tool_calls(
    a: tuple[ToolCall, ...] | None, b: tuple[ToolCall, ...] | None
) -> tuple[ToolCall, ...] | None:
    """Merge streaming tool-call fragments by `index`.

    The first fragment of an index carries id/name; later fragments append
    `function_arguments`. Meant for streaming accumulation, where fragments carry
    an `index`.
    """
    if a is None and b is None:
        return None
    acc: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    for tc in (a or ()) + (b or ()):
        idx = tc.index if tc.index is not None else 0
        if idx not in acc:
            acc[idx] = {
                "id": "",
                "type": "function",
                "index": tc.index,
                "function_name": "",
                "function_arguments": "",
            }
            order.append(idx)
        cur = acc[idx]
        if tc.id:
            cur["id"] = tc.id
        if tc.type:
            cur["type"] = tc.type
        if tc.function_name:
            cur["function_name"] = tc.function_name
        cur["function_arguments"] += tc.function_arguments
    return tuple(ToolCall(**acc[i]) for i in order)


def _merge_reasoning_details(
    a: tuple[ReasoningDetail, ...] | None, b: tuple[ReasoningDetail, ...] | None
) -> tuple[ReasoningDetail, ...] | None:
    """Merge streaming reasoning fragments by `index`, concatenating `text`
    (parallel to tool-call argument accumulation). Non-text fields are taken from
    whichever fragment supplies them.
    """
    if a is None and b is None:
        return None
    acc: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    for rd in (a or ()) + (b or ()):
        idx = rd.index if rd.index is not None else 0
        if idx not in acc:
            acc[idx] = {
                "type": rd.type,
                "id": None,
                "format": None,
                "index": rd.index,
                "summary": None,
                "data": None,
                "text": None,
                "signature": None,
            }
            order.append(idx)
        cur = acc[idx]
        for f in ("type", "id", "format", "summary", "data", "signature"):
            v = getattr(rd, f)
            if v is not None:
                cur[f] = v
        if rd.text is not None:
            cur["text"] = (cur["text"] or "") + rd.text
    return tuple(ReasoningDetail(**acc[i]) for i in order)


class MessageChunk(BaseModel):
    """A streaming delta. Fold a stream with `+`; finalize with `convert()`."""

    model_config = ConfigDict(frozen=True)

    content: str | None = None
    role: Role | None = None
    reasoning: str | None = None
    reasoning_details: tuple[ReasoningDetail, ...] | None = None
    tool_calls: tuple[ToolCall, ...] | None = None
    stop_reason: str | None = None
    usage: Usage | None = None
    provider_id: str | None = None

    def __add__(self, other: MessageChunk) -> MessageChunk:
        if not isinstance(other, MessageChunk):
            return NotImplemented
        return MessageChunk(
            content=_concat(self.content, other.content),
            role=other.role if other.role is not None else self.role,
            reasoning=_concat(self.reasoning, other.reasoning),
            reasoning_details=_merge_reasoning_details(
                self.reasoning_details, other.reasoning_details
            ),
            tool_calls=_merge_tool_calls(self.tool_calls, other.tool_calls),
            stop_reason=other.stop_reason if other.stop_reason is not None else self.stop_reason,
            usage=other.usage if other.usage is not None else self.usage,
            provider_id=other.provider_id if other.provider_id is not None else self.provider_id,
        )

    def convert(self) -> Message:
        """Finalize an accumulated stream into an immutable Message."""
        return Message(
            role=self.role if self.role is not None else Role.ASSISTANT,
            content=self.content,
            provider_id=self.provider_id,
            usage=self.usage,
            reasoning=self.reasoning,
            reasoning_details=self.reasoning_details,
            stop_reason=self.stop_reason,
            tool_calls=self.tool_calls or (),
        )


class ToolSpec(BaseModel):
    """A tool declaration as the model sees it: pure data, no callable.

    `parameters` is a JSON Schema object. Turning specs into the provider's shape is
    the normalizers' job; executing them is `run_tools`'s.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


class Step(BaseModel):
    """One turn: the model's reply and the results of its tool calls, in call order.
    `tool_results` is empty when the model answered without calling tools."""

    model_config = ConfigDict(frozen=True)

    assistant: Message
    tool_results: tuple[Message, ...] = ()

    @property
    def messages(self) -> tuple[Message, ...]:
        """The reply followed by its tool results: what to add to a history."""
        return (self.assistant, *self.tool_results)

    @property
    def usage(self) -> Usage:
        """What this step's model call consumed (empty if the provider reported none)."""
        return self.assistant.usage or Usage()


def check_pairing(messages: Sequence[Message]) -> None:
    """Raise ToolPairingError unless every tool call is answered before the next
    non-tool message, and every tool result answers a call of the turn it sits in."""
    pending: set[str] = set()
    for i, msg in enumerate(messages):
        if msg.role is Role.TOOL:
            if msg.tool_call_id not in pending:
                raise ToolPairingError(
                    f"message {i}: tool result {msg.tool_call_id!r} answers no open tool call"
                )
            pending.discard(msg.tool_call_id)
            continue
        if pending:
            raise ToolPairingError(
                f"message {i}: tool call(s) {sorted(pending)} have no result before it"
            )
        pending = {tc.id for tc in msg.tool_calls}
    if pending:
        raise ToolPairingError(f"tool call(s) {sorted(pending)} have no result")
