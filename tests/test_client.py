"""Integration tests: the client wired end to end with a mocked SDK stream, no model."""
from types import SimpleNamespace

import httpx2 as httpx
import openai
import pytest

from protonous.client import OpenrouterClient, OpenrouterConfig, collect
from protonous.context import Message, Role, ToolSpec
from protonous.errors import ProtonousError, ProviderError
from protonous.layers import ContextWindow, StaticLayer
from protonous.settings import Settings

CONFIG = OpenrouterConfig(model="x")


def _ctx():
    return ContextWindow(layers=[StaticLayer([Message(role=Role.USER, content="hi")])])


def _delta(**kw):
    base = {"role": None, "content": None, "tool_calls": None, "reasoning": None,
            "reasoning_details": None}
    base.update(kw)
    return SimpleNamespace(**base)


def _chunk(finish_reason=None, usage=None, **delta):
    return SimpleNamespace(
        id="g", usage=usage,
        choices=[SimpleNamespace(finish_reason=finish_reason, delta=_delta(**delta))],
    )


USAGE = SimpleNamespace(completion_tokens=2, prompt_tokens=1, total_tokens=3,
                        completion_tokens_details=None, prompt_tokens_details=None)
HELLO = [
    _chunk(role="assistant", content="He"),
    _chunk(content="llo"),
    _chunk(finish_reason="stop"),
    SimpleNamespace(id="g", choices=[], usage=USAGE),
]


def _client(stream_items=(), *, raise_on_create=None, seen=None):
    """A client whose SDK `create` returns a stream of `stream_items` (an item that is
    an exception is raised mid-stream) and records its kwargs into `seen`."""
    client = OpenrouterClient(Settings(openrouter_api_key="test"))

    async def _stream():
        for item in stream_items:
            if isinstance(item, BaseException):
                raise item
            yield item

    async def _create(**kwargs):
        if seen is not None:
            seen.update(kwargs)
        if raise_on_create is not None:
            raise raise_on_create
        return _stream()

    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )
    return client


async def test_stream_and_collect_folds():
    msg = await collect(_client(HELLO).stream(_ctx(), CONFIG))
    assert msg.content == "Hello" and msg.role is Role.ASSISTANT
    assert msg.stop_reason == "stop" and msg.token_count == 2


async def test_complete_is_a_collected_stream():
    seen = {}
    msg = await _client(HELLO, seen=seen).complete(_ctx(), CONFIG)
    assert msg.content == "Hello" and msg.token_count == 2
    assert seen["stream"] is True and seen["stream_options"] == {"include_usage": True}


async def test_tools_are_sent_only_when_given():
    seen = {}
    await _client(HELLO, seen=seen).complete(_ctx(), CONFIG)
    assert "tools" not in seen

    seen = {}
    spec = ToolSpec(name="f", description="d")
    await _client(HELLO, seen=seen).complete(
        _ctx(), OpenrouterConfig(model="x", tool_choice="auto"), tools=[spec]
    )
    assert seen["tools"][0]["function"]["name"] == "f"
    assert seen["tool_choice"] == "auto"


def _sdk_error():
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return openai.BadRequestError(
        "Error code: 400 - {... 'user_id': 'user_secret'}",
        response=response,
        body={"message": "bad model", "code": 400},
    ), request


async def test_provider_errors_are_wrapped():
    sdk_error, request = _sdk_error()
    with pytest.raises(ProviderError) as info:
        await _client(raise_on_create=sdk_error).complete(_ctx(), CONFIG)
    assert str(info.value) == "HTTP 400: bad model" and info.value.status_code == 400
    assert isinstance(info.value, ProtonousError)
    # the SDK exception (and the account id in its message) is not chained into tracebacks
    assert info.value.__cause__ is None and info.value.__suppress_context__

    conn = openai.APIConnectionError(request=request)
    with pytest.raises(ProviderError) as info:
        await _client(raise_on_create=conn).complete(_ctx(), CONFIG)
    assert info.value.status_code is None


async def test_sdk_error_mid_stream_is_wrapped():
    sdk_error, _ = _sdk_error()
    with pytest.raises(ProviderError, match="HTTP 400: bad model"):
        await _client([_chunk(role="assistant", content="He"), sdk_error]).complete(_ctx(), CONFIG)


async def test_transport_failure_mid_stream_is_wrapped():
    items = [_chunk(role="assistant", content="He"), ConnectionResetError("peer reset")]
    with pytest.raises(ProviderError, match="stream failed: ConnectionResetError: peer reset"):
        await _client(items).complete(_ctx(), CONFIG)


async def test_error_event_mid_stream_raises():
    error_chunk = SimpleNamespace(
        id="g", usage=None, error={"code": 502, "message": "upstream overloaded"},
        choices=[SimpleNamespace(finish_reason="error", delta=_delta())],
    )
    with pytest.raises(ProviderError, match="stream error: upstream overloaded"):
        await _client([_chunk(role="assistant", content="He"), error_chunk]).complete(_ctx(), CONFIG)


async def test_finish_reason_error_without_details_raises():
    with pytest.raises(ProviderError, match="stream error"):
        await _client([_chunk(role="assistant"), _chunk(finish_reason="error")]).complete(
            _ctx(), CONFIG
        )


async def test_truncated_stream_raises_instead_of_returning_partial_text():
    with pytest.raises(ProviderError, match="ended before the response finished"):
        await _client([_chunk(role="assistant", content="He"), _chunk(content="ll")]).complete(
            _ctx(), CONFIG
        )


async def test_chunks_before_a_failure_are_still_yielded():
    received = []
    with pytest.raises(ProviderError):
        async for chunk in _client([_chunk(role="assistant", content="He")]).stream(_ctx(), CONFIG):
            received.append(chunk.content)
    assert received == ["He"]  # a UI may have shown them; the caller still gets the error
