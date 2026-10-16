"""Live model smokes — deselected by default.

Run deliberately: `pytest -m model` with PROTONOUS_TEST_MODEL set and
OPENROUTER_API_KEY in .env. Costs a fraction of a cent.
"""
import os

import pytest

from protonous.client import OpenrouterClient, OpenrouterConfig, collect
from protonous.context import Message, Role
from protonous.layers import ContextWindow, StaticLayer
from protonous.settings import get_settings

MODEL = os.environ.get("PROTONOUS_TEST_MODEL")

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(not MODEL, reason="set PROTONOUS_TEST_MODEL to run live smokes"),
]


def _client() -> OpenrouterClient:
    return OpenrouterClient(get_settings())


def _ctx() -> ContextWindow:
    return ContextWindow(
        layers=[StaticLayer([Message(role=Role.USER, content="Reply with exactly one word: hello")])]
    )


def _config() -> OpenrouterConfig:
    return OpenrouterConfig(
        # generous headroom: if the model reasons, its reasoning tokens count
        # against max_tokens — too small a budget gets consumed before any content.
        model=MODEL,
        max_tokens=1024,
        extra_body={"provider": {"data_collection": "deny"}},
    )


async def test_live_complete():
    msg = await _client().complete(_ctx(), _config())
    assert msg.role is Role.ASSISTANT
    assert msg.content and msg.content.strip()
    assert msg.usage is not None
    assert msg.usage.prompt_tokens and msg.usage.completion_tokens and msg.usage.total_tokens
    assert msg.usage.cost is not None  # OpenRouter reports cost in the final chunk
    assert msg.provider_id


async def test_live_stream_collect():
    msg = await collect(_client().stream(_ctx(), _config()))
    assert msg.role is Role.ASSISTANT
    assert msg.content and msg.content.strip()


async def test_live_loop_uses_a_tool():
    from protonous.layers import SlidingWindowLayer
    from protonous.model import Model
    from protonous.tools import tool

    @tool
    async def add(a: int, b: int) -> str:
        """Add two integers."""
        return str(a + b)

    history = SlidingWindowLayer()
    history.add(
        Message(role=Role.USER, content="Use the add tool to compute 17 + 25, then state the result.")
    )
    window = ContextWindow(layers=[history])
    model = Model(_client(), _config())
    steps = []
    for _ in range(4):
        step = await model.step(window, [add])
        steps.append(step)
        history.extend(step.messages)
        if not step.tool_results:
            break
    assert not steps[-1].tool_results  # finished by answering
    assert any(step.tool_results for step in steps)
    assert "42" in (steps[-1].assistant.content or "")


async def test_live_stream_chunks_and_early_close():
    from protonous.errors import StreamIncompleteError
    from protonous.model import Model

    model = Model(_client(), _config())
    stream = model.stream(_ctx())
    chunks = [chunk async for chunk in stream]
    assert len(chunks) > 1 and stream.reply.content and stream.reply.usage is not None

    long_ask = ContextWindow(
        layers=[StaticLayer([Message(role=Role.USER, content="Count from 1 to 300, one per line.")])]
    )
    async with model.stream(long_ask) as stream:
        async for _chunk in stream:
            break  # stop right away; the connection is closed on exit
    try:
        _ = stream.reply
        raise AssertionError("an early-closed stream must not have a reply")
    except StreamIncompleteError:
        pass
