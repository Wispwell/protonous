"""Tools: a ToolSpec plus the code behind it, and how a tool call is executed.

Model mistakes (unknown tool, malformed or invalid arguments, a tool that raises)
become `Error: ...` tool results the model can read and react to. Programmer
mistakes (a tool returning a non-string, duplicate tool names) raise
ToolDefinitionError.
"""
from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from .context import Message, Role, ToolCall, ToolSpec
from .errors import ToolDefinitionError


class Tool(BaseModel):
    """Arguments are validated against `args` before `fn` is called; `fn` receives
    the validated model and returns a string."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    args: type[BaseModel]
    fn: Callable[[Any], Awaitable[str]]

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.args.model_json_schema(),
        )


def by_name(tools: Sequence[Tool]) -> dict[str, Tool]:
    """Index tools by name; names must be unique."""
    registry = {tool.name: tool for tool in tools}
    if len(registry) != len(tools):
        raise ToolDefinitionError("tool names must be unique")
    return registry


async def execute(registry: Mapping[str, Tool], call: ToolCall) -> Message:
    """Run one tool call and return its tool-result message."""
    tool = registry.get(call.function_name)
    if tool is None:
        available = ", ".join(sorted(registry)) or "none"
        return _error(call, f"unknown tool {call.function_name!r} (available: {available})")
    try:
        raw = json.loads(call.function_arguments or "{}")
    except json.JSONDecodeError as e:
        return _error(call, f"arguments are not valid JSON: {e}")
    try:
        args = tool.args.model_validate(raw)
    except ValidationError as e:
        return _error(call, f"invalid arguments: {e}")
    try:
        output = await tool.fn(args)
    except Exception as e:  # noqa: BLE001 — the tool's failure is the model's information
        return _error(call, f"{type(e).__name__}: {e}")
    if not isinstance(output, str):
        raise ToolDefinitionError(
            f"tool {tool.name!r} returned {type(output).__name__}, not str"
        )
    return Message(role=Role.TOOL, content=output, tool_call_id=call.id)


def _error(call: ToolCall, text: str) -> Message:
    return Message(
        role=Role.TOOL,
        content=f"Error: {text}",
        tool_call_id=call.id,
        extras={"error": True},
    )
