"""A streaming stand-in for OpenrouterClient.

Each scripted reply is streamed the way providers do it: in fragments, with tool
calls split into a header chunk and argument slices, so every test that runs the
loop also exercises chunk merging.
"""
from protonous.context import Message, MessageChunk, Role, ToolCall


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
    return chunks


class ScriptedClient:
    """Streams the scripted replies in order and records what each call saw."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []  # (message contents in the window, tool names offered)
        self.replay_flags = []

    async def stream(self, window, config, tools=None, *, replay_all_reasoning=True):
        self.replay_flags.append(replay_all_reasoning)
        self.seen.append(([m.content for m in window.view()], [t.name for t in tools or []]))
        for chunk in chunks_of(self.replies.pop(0)):
            yield chunk


class FailingClient:
    """Raises `exc` when the stream is read."""

    def __init__(self, exc):
        self.exc = exc

    async def stream(self, window, config, tools=None, *, replay_all_reasoning=True):
        raise self.exc
        yield  # makes this an async generator
