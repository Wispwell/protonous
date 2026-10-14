"""Integration tests — the client wired end-to-end with a mocked SDK, no model."""
from types import SimpleNamespace

from protonous.client import OpenrouterClient, OpenrouterConfig, collect
from protonous.context import Message, Role
from protonous.layers import ContextWindow, StaticLayer
from protonous.settings import Settings


def _client_returning(result):
    """Build a client whose SDK `create` returns `result` (a response or async iterator)."""
    client = OpenrouterClient(Settings(openrouter_api_key="test"))

    async def _create(**kwargs):
        return result

    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )
    return client


def _ctx():
    return ContextWindow(layers=[StaticLayer([Message(role=Role.USER, content="hi")])])


async def test_complete_returns_message():
    resp = SimpleNamespace(
        id="gen-1",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    role="assistant",
                    content="Hello",
                    tool_calls=None,
                    reasoning=None,
                    reasoning_details=None,
                ),
            )
        ],
        usage=SimpleNamespace(
            completion_tokens=5,
            prompt_tokens=1,
            total_tokens=6,
            completion_tokens_details=None,
            prompt_tokens_details=None,
        ),
    )
    msg = await _client_returning(resp).complete(_ctx(), OpenrouterConfig(model="x"))
    assert msg.content == "Hello" and msg.token_count == 5 and msg.role is Role.ASSISTANT


async def test_stream_and_collect_folds():
    def _delta(**kw):
        base = {
            "role": None,
            "content": None,
            "tool_calls": None,
            "reasoning": None,
            "reasoning_details": None,
        }
        base.update(kw)
        return SimpleNamespace(**base)

    chunks = [
        SimpleNamespace(id="g", usage=None,
                        choices=[SimpleNamespace(finish_reason=None, delta=_delta(role="assistant", content="He"))]),
        SimpleNamespace(id="g", usage=None,
                        choices=[SimpleNamespace(finish_reason=None, delta=_delta(content="llo"))]),
        SimpleNamespace(id="g", usage=None,
                        choices=[SimpleNamespace(finish_reason="stop", delta=_delta())]),
        SimpleNamespace(id="g", choices=[],
                        usage=SimpleNamespace(completion_tokens=2, prompt_tokens=1, total_tokens=3,
                                              completion_tokens_details=None, prompt_tokens_details=None)),
    ]

    async def _agen():
        for c in chunks:
            yield c

    client = _client_returning(_agen())
    msg = await collect(client.stream(_ctx(), OpenrouterConfig(model="x")))
    assert msg.content == "Hello"
    assert msg.role is Role.ASSISTANT
    assert msg.stop_reason == "stop"
    assert msg.token_count == 2


def _capturing_client():
    """A client whose SDK `create` records its kwargs and returns a minimal response."""
    client = OpenrouterClient(Settings(openrouter_api_key="test"))
    seen = {}

    async def _create(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            id="gen",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        role="assistant", content="ok", tool_calls=None,
                        reasoning=None, reasoning_details=None,
                    ),
                )
            ],
            usage=None,
        )

    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )
    return client, seen


async def test_complete_sends_tools_only_when_given():
    from protonous.context import ToolSpec

    client, seen = _capturing_client()
    await client.complete(_ctx(), OpenrouterConfig(model="x"))
    assert "tools" not in seen

    spec = ToolSpec(name="f", description="d")
    await client.complete(_ctx(), OpenrouterConfig(model="x", tool_choice="auto"), tools=[spec])
    assert seen["tools"][0]["function"]["name"] == "f"
    assert seen["tool_choice"] == "auto"


def _client_raising(exc):
    client = OpenrouterClient(Settings(openrouter_api_key="test"))

    async def _create(**kwargs):
        raise exc

    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )
    return client


async def test_provider_errors_are_wrapped():
    import httpx2 as httpx
    import openai
    import pytest

    from protonous.errors import ProtonousError, ProviderError

    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(400, request=request)
    sdk_error = openai.BadRequestError(
        "Error code: 400 - {... 'user_id': 'user_secret'}",
        response=response,
        body={"message": "bad model", "code": 400},
    )

    with pytest.raises(ProviderError) as info:
        await _client_raising(sdk_error).complete(_ctx(), OpenrouterConfig(model="x"))
    assert str(info.value) == "HTTP 400: bad model" and info.value.status_code == 400
    assert isinstance(info.value, ProtonousError)
    # the SDK exception (and the account id in its message) is not chained into tracebacks
    assert info.value.__cause__ is None and info.value.__suppress_context__

    with pytest.raises(ProviderError) as info:
        await collect(_client_raising(sdk_error).stream(_ctx(), OpenrouterConfig(model="x")))
    assert info.value.status_code == 400

    conn = openai.APIConnectionError(request=request)
    with pytest.raises(ProviderError) as info:
        await _client_raising(conn).complete(_ctx(), OpenrouterConfig(model="x"))
    assert info.value.status_code is None
