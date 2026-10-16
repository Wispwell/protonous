"""A model to call, and the calls an agent is built from.

    model = Model(OpenrouterClient(get_settings()), OpenrouterConfig(model="..."))
    reply = await model.call(window, tools)    # one streamed call -> Message
    step = await model.step(window, tools)     # the call plus its tool calls -> Step

    async with model.stream(window, tools) as stream:   # the same call, chunk by chunk
        async for chunk in stream:
            ...
        step = await stream.step()

None of them touches the window: what goes into it, when, and what gets evicted is
the caller's loop. The only check is on the request itself: tool calls in the
window must be paired with their results (ToolPairingError otherwise).
"""
from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType
from typing import Literal, Self

from .client import OpenrouterClient, OpenrouterConfig
from .context import Message, MessageChunk, Step, check_pairing
from .errors import StreamIncompleteError
from .layers import ContextWindow
from .tools import Tool, run_tools


class Model:
    """What to call (the client) and how (the per-request config)."""

    def __init__(self, client: OpenrouterClient, config: OpenrouterConfig) -> None:
        self.client = client
        self.config = config

    def stream(self, window: ContextWindow, tools: Sequence[Tool] = ()) -> ReplyStream:
        """One model call over `window`, offering `tools`, as a stream of chunks.
        The request is checked now and sent when the stream is first read."""
        return ReplyStream(self, window, tools)

    async def call(self, window: ContextWindow, tools: Sequence[Tool] = ()) -> Message:
        """One model call; returns the assembled reply."""
        return await self.stream(window, tools).collect()

    async def step(self, window: ContextWindow, tools: Sequence[Tool] = ()) -> Step:
        """One turn: `call`, then run the reply's tool calls in order."""
        return await self.stream(window, tools).step()


class ReplyStream:
    """One model call, read chunk by chunk.

    Iterate it for the chunks as they arrive (content, reasoning and tool-call
    fragments); then `reply` is the assembled Message and `await step()` runs its
    tool calls. `collect()` and `step()` read whatever hasn't been read yet, so they
    work without iterating at all.

    A stream that is closed early (`async with` exits, `aclose()`) or fails has no
    reply: `reply`, `collect()` and `step()` raise StreamIncompleteError. Use
    `async with` whenever the loop might stop reading early, so the connection is
    closed right away.
    """

    def __init__(self, model: Model, window: ContextWindow, tools: Sequence[Tool]) -> None:
        check_pairing(window.view())
        self._tools = tools
        specs = [tool.spec for tool in tools] or None
        self._chunks = model.client.stream(window, model.config, specs)
        self._acc = MessageChunk()
        self._state: Literal["open", "done", "failed", "closed"] = "open"
        self._reply: Message | None = None

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> MessageChunk:
        if self._state != "open":
            raise StopAsyncIteration
        try:
            chunk = await anext(self._chunks)
        except StopAsyncIteration:
            self._state = "done"
            self._reply = self._acc.convert()
            raise
        except BaseException:
            self._state = "failed"
            raise
        self._acc = self._acc + chunk
        return chunk

    @property
    def reply(self) -> Message:
        """The assembled reply; only once the stream has been read to the end."""
        if self._reply is None:
            raise StreamIncompleteError(f"the stream is {self._state}, so it has no reply")
        return self._reply

    async def collect(self) -> Message:
        """Read the rest of the stream (if any) and return the reply."""
        async for _ in self:
            pass
        return self.reply

    async def step(self) -> Step:
        """Read the rest of the stream, then run the reply's tool calls in order."""
        reply = await self.collect()
        return Step(assistant=reply, tool_results=await run_tools(self._tools, reply))

    async def aclose(self) -> None:
        """Stop reading and close the connection. Without a finished stream there is
        no reply afterwards."""
        if self._state == "open":
            self._state = "closed"
        await self._chunks.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
