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
    assert msg.token_count is not None and msg.token_count > 0
    assert msg.provider_id


async def test_live_stream_collect():
    msg = await collect(_client().stream(_ctx(), _config()))
    assert msg.role is Role.ASSISTANT
    assert msg.content and msg.content.strip()


async def test_live_loop_uses_a_tool():
    from protonous.layers import SlidingWindowLayer
    from protonous.loop import append_to, run
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
    result = await run(_client(), window, _config(), [add], update=append_to(history), max_steps=4)
    assert result.stop_reason == "done"
    assert any(step.tool_results for step in result.steps)
    assert "42" in (result.steps[-1].assistant.content or "")
