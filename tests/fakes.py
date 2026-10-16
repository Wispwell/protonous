"""A streaming stand-in for OpenrouterClient, and the reference agent loop.

Each scripted reply is streamed the way providers do it: in fragments, with tool
calls split into a header chunk and argument slices, so every test that runs the
loop also exercises chunk merging.
"""
from protonous.client import OpenrouterConfig
from protonous.context import Message, MessageChunk, Role, ToolCall
from protonous.model import Model


def _halves(text):
    middle = len(text) // 2
    return [part for part in (text[:middle], text[middle:]) if part]


def chunks_of(reply: Message) -> list[MessageChunk]:
    chunks = [MessageChunk(role=Role.ASSISTANT)]
    for part in _halves(reply.reasoning or ""):
        chunks.append(MessageChunk(reasoning=part))
    if reply.reasoning_details:
        chunks.append(MessageChunk(reasoning_details=reply.reasoning_details))
    for part in _halves(reply.content or ""):
        chunks.append(MessageChunk(content=part))
    for index, call in enumerate(reply.tool_calls):
        chunks.append(
            MessageChunk(
                tool_calls=(
                    ToolCall(index=index, id=call.id, function_name=call.function_name),
                )
            )
        )
        for part in _halves(call.function_arguments):
            chunks.append(
                MessageChunk(tool_calls=(ToolCall(index=index, function_arguments=part),))
            )
    chunks.append(MessageChunk(stop_reason="tool_calls" if reply.tool_calls else "stop"))
    if reply.usage is not None:
        chunks.append(MessageChunk(usage=reply.usage))  # providers send usage last
    return chunks


class ScriptedClient:
    """Streams the scripted replies in order and records what each call saw."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []  # (message contents in the window, tool names offered)
        self.closed = 0  # streams closed before their last chunk was read

    async def stream(self, window, config, tools=None):
        self.seen.append(([m.content for m in window.view()], [t.name for t in tools or []]))
        chunks = chunks_of(self.replies.pop(0))
        sent = 0
        try:
            for chunk in chunks:
                yield chunk
                sent += 1
        finally:
            if sent < len(chunks):
                self.closed += 1


class FailingClient:
    """Raises `exc` when the stream is read."""

    def __init__(self, exc):
        self.exc = exc

    async def stream(self, window, config, tools=None):
        raise self.exc
        yield  # makes this an async generator


def model_for(client):
    return Model(client, OpenrouterConfig(model="x"))


async def agent(model, window, history, tools=(), *, max_steps=10, tracer=None):
    """The loop the docs show: step until the model answers without tools."""
    steps = []

    async def _loop():
        for _ in range(max_steps):
            step = await model.step(window, tools)
            steps.append(step)
            if tracer is not None:
                tracer.step(step)
            history.extend(step.messages)
            if not step.tool_results:
                return "done"
        return "max_steps"

    if tracer is None:
        return steps, await _loop()
    with tracer.recording():
        stop = await _loop()
        tracer.end(stop)
    return steps, stop
