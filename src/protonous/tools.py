"""Tools: a ToolSpec plus the code behind it, and how a tool call is executed.

Collect an agent's tools in a Toolbox: decorating a function turns its name,
docstring and typed parameters into the tool's name, description and argument
schema, and registers the tool in that toolbox.

    tools = Toolbox()

    @tools.tool
    async def add(a: int, b: int) -> str:
        \"\"\"Add two numbers.\"\"\"
        return str(a + b)

    run(client, window, config, tools, ...)

`@tool` does the same without registering anywhere, and `Tool(...)` is the explicit
form underneath, for an existing Pydantic args model.

Model mistakes (unknown tool, malformed or invalid arguments, a tool that raises)
become `Error: ...` tool results the model can read and react to. Programmer
mistakes (a tool returning a non-string, duplicate tool names) raise
ToolDefinitionError.
"""
from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any, get_type_hints, overload

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

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


ToolFn = Callable[..., Awaitable[str]]


@overload
def tool(fn: ToolFn, /) -> Tool: ...
@overload
def tool(*, name: str | None = None, description: str | None = None) -> Callable[[ToolFn], Tool]: ...
def tool(
    fn: ToolFn | None = None, /, *, name: str | None = None, description: str | None = None
) -> Tool | Callable[[ToolFn], Tool]:
    """Turn an async function into a Tool: `@tool` or `@tool(name=..., description=...)`.

    Name defaults to the function's name, description to its docstring. Every
    parameter needs a type annotation; defaults make it optional, and
    `Annotated[T, Field(description=...)]` documents it for the model. The arguments
    are validated against a Pydantic model built from the signature (unknown
    arguments are rejected), then passed to the function as keywords.
    """

    def make(fn: ToolFn) -> Tool:
        if not inspect.iscoroutinefunction(fn):
            raise ToolDefinitionError(f"tool {fn.__name__!r} must be an async function")
        text = description if description is not None else inspect.getdoc(fn)
        if not text:
            raise ToolDefinitionError(
                f"tool {fn.__name__!r} needs a docstring or a description= (the model reads it)"
            )
        args = _args_model(fn)

        async def call(validated: BaseModel) -> str:
            return await fn(**{field: getattr(validated, field) for field in args.model_fields})

        return Tool(name=name or fn.__name__, description=text, args=args, fn=call)

    return make if fn is None else make(fn)


def _args_model(fn: ToolFn) -> type[BaseModel]:
    try:
        hints = get_type_hints(fn, include_extras=True)
    except NameError as e:
        raise ToolDefinitionError(f"tool {fn.__name__!r}: can't resolve annotation ({e})") from e
    fields: dict[str, Any] = {}
    for param in inspect.signature(fn).parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD, param.POSITIONAL_ONLY):
            raise ToolDefinitionError(
                f"tool {fn.__name__!r}: parameter {param.name!r} must be a plain named "
                "parameter (no *args, **kwargs or positional-only)"
            )
        if param.name not in hints:
            raise ToolDefinitionError(
                f"tool {fn.__name__!r}: parameter {param.name!r} needs a type annotation"
            )
        default = ... if param.default is param.empty else param.default
        fields[param.name] = (hints[param.name], default)
    model_name = "".join(part.title() for part in fn.__name__.split("_")) + "Args"
    return create_model(model_name, __config__=ConfigDict(extra="forbid"), **fields)


class Toolbox(Sequence[Tool]):
    """An agent's tool set, in registration order. Pass it wherever tools go.

    `@toolbox.tool` (or `@toolbox.tool(name=..., description=...)`) defines a tool
    like `@tool` and registers it here; `add` registers an existing Tool, so one tool
    can sit in several toolboxes. Names are unique within a toolbox.
    """

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: list[Tool] = []
        for t in tools:
            self.add(t)

    def add(self, t: Tool) -> Tool:
        if any(existing.name == t.name for existing in self._tools):
            raise ToolDefinitionError(f"toolbox already has a tool named {t.name!r}")
        self._tools.append(t)
        return t

    @overload
    def tool(self, fn: ToolFn, /) -> Tool: ...
    @overload
    def tool(
        self, *, name: str | None = None, description: str | None = None
    ) -> Callable[[ToolFn], Tool]: ...
    def tool(
        self,
        fn: ToolFn | None = None,
        /,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Tool | Callable[[ToolFn], Tool]:
        def register(fn: ToolFn) -> Tool:
            return self.add(tool(name=name, description=description)(fn))

        return register if fn is None else register(fn)

    @overload
    def __getitem__(self, index: int) -> Tool: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[Tool]: ...
    def __getitem__(self, index: int | slice) -> Tool | Sequence[Tool]:
        return self._tools[index]

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools)

    def __repr__(self) -> str:
        return f"Toolbox({[t.name for t in self._tools]})"


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
