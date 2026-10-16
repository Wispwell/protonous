"""The agent loop: native tool calling, one step at a time.

Each step: check tool-call pairing -> call the model -> run its tool calls in order ->
hand the step to `update`, which returns the window for the next step. The loop
never decides where messages go; `update` does (`append_to(layer)` covers the
common case). Stops when the model answers without tool calls, or at `max_steps`.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from .client import OpenrouterClient, OpenrouterConfig
from .context import Message, MessageChunk, Role, ToolSpec
from .errors import ConfigurationError, ToolPairingError
from .layers import ContextWindow, Layer
from .tools import Tool, by_name, execute

if TYPE_CHECKING:
    from .trace import Tracer


class Step(BaseModel):
    """One turn: the model's reply and the results of its tool calls, in call order.
    `tool_results` is empty on the final step."""

    model_config = ConfigDict(frozen=True)

    assistant: Message
    tool_results: tuple[Message, ...] = ()


Update = Callable[[ContextWindow, Step], ContextWindow]
OnChunk = Callable[[MessageChunk], None]


class RunResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    window: ContextWindow
    steps: tuple[Step, ...]
    stop_reason: Literal["done", "max_steps"]


def append_to(layer: Layer) -> Update:
    """An `update` that adds each step to `layer`, assistant message first."""

    def _update(window: ContextWindow, step: Step) -> ContextWindow:
        if not any(existing is layer for existing in window.layers):
            raise ConfigurationError("append_to: the layer is not part of the window")
        layer.add(step.assistant)
        for result in step.tool_results:
            layer.add(result)
        return window

    return _update


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


async def run(
    client: OpenrouterClient,
    window: ContextWindow,
    config: OpenrouterConfig,
    tools: Sequence[Tool] = (),
    *,
    update: Update,
    max_steps: int,
    replay_all_reasoning: bool = True,
    tracer: Tracer | None = None,
    on_chunk: OnChunk | None = None,
) -> RunResult:
    """Run until the model answers without tool calls, or `max_steps`.

    Replies are streamed; `on_chunk` (e.g. a chat UI) sees every chunk as it arrives.
    It is synchronous: to hand chunks to async code, put them on a queue.

    `replay_all_reasoning=False` sends reasoning back only on the latest assistant
    message instead of all of them (cheaper on long runs). Stored messages are
    never changed either way, so the steps keep all reasoning.

    With a `tracer`, each step is recorded before `update` sees it, and the end of
    the run is recorded when it returns. This is protonous's one exception handler:
    whatever escapes the run (including Ctrl-C and cancellation) is recorded by the
    tracer and re-raised unchanged.
    """
    try:
        return await _run(
            client, window, config, tools, update, max_steps, replay_all_reasoning, tracer,
            on_chunk,
        )
    except BaseException as e:
        if tracer is not None:
            tracer.error(e)
        raise


async def _run(
    client: OpenrouterClient,
    window: ContextWindow,
    config: OpenrouterConfig,
    tools: Sequence[Tool],
    update: Update,
    max_steps: int,
    replay_all_reasoning: bool,
    tracer: Tracer | None,
    on_chunk: OnChunk | None,
) -> RunResult:
    registry = by_name(tools)
    specs = [tool.spec for tool in tools] or None

    steps: list[Step] = []
    stop_reason: Literal["done", "max_steps"] = "max_steps"
    for _ in range(max_steps):
        check_pairing(window.view())
        reply = await _stream_reply(client, window, config, specs, replay_all_reasoning, on_chunk)
        results = tuple([await execute(registry, call) for call in reply.tool_calls])
        step = Step(assistant=reply, tool_results=results)
        steps.append(step)
        if tracer is not None:
            tracer.step(step)
        window = update(window, step)
        if not reply.tool_calls:
            stop_reason = "done"
            break
    result = RunResult(window=window, steps=tuple(steps), stop_reason=stop_reason)
    if tracer is not None:
        tracer.end(result)
    return result


async def _stream_reply(
    client: OpenrouterClient,
    window: ContextWindow,
    config: OpenrouterConfig,
    specs: list[ToolSpec] | None,
    replay_all_reasoning: bool,
    on_chunk: OnChunk | None,
) -> Message:
    acc = MessageChunk()
    async for chunk in client.stream(
        window, config, specs, replay_all_reasoning=replay_all_reasoning
    ):
        if on_chunk is not None:
            on_chunk(chunk)
        acc = acc + chunk
    return acc.convert()
