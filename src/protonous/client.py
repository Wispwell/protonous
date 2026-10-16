"""OpenRouter client (async, via the OpenAI SDK) + per-request config.

Streaming is the only transport: `stream()` yields MessageChunks, and `complete()`
is `collect(stream())`. A stream that fails partway, or ends before the response
finished, raises ProviderError; it never yields a silently truncated message.

The client is agent-agnostic: it takes a ContextWindow (plus optional ToolSpecs, which
it declares but never executes) and returns our Message /
MessageChunk. All provider-shaped mapping lives in normalizers. Folding a stream is
the caller's job (via MessageChunk `+`); `collect()` is the convenience for when
per-chunk access isn't needed.
"""
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from typing import Any, Literal, cast

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from .context import Message, MessageChunk, ToolSpec
from .errors import ProviderError
from .layers import ContextWindow
from .normalizers import normalize_chunk, normalize_in, normalize_tools
from .settings import Settings

_STD_PARAMS = (
    "temperature",
    "max_tokens",
    "top_p",
    "stop",
    "seed",
    "frequency_penalty",
    "presence_penalty",
    "tool_choice",
)


class OpenrouterConfig(BaseModel):
    """Per-request generation parameters. `model` is required and supplied by the
    caller; everything else is optional."""

    model: str
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    seed: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    # "auto" | "none" | "required", or {"type": "function", "function": {"name": ...}}
    # to force one tool. Only meaningful when tools are passed; providers vary in support.
    tool_choice: Literal["auto", "none", "required"] | dict[str, Any] | None = None
    extra_body: dict[str, Any] | None = None

    def to_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"model": self.model}
        for name in _STD_PARAMS:
            value = getattr(self, name)
            if value is not None:
                params[name] = value
        extra_body = dict(self.extra_body or {})
        if self.reasoning_effort is not None:
            extra_body.setdefault("reasoning", {})["effort"] = self.reasoning_effort
        if extra_body:
            params["extra_body"] = extra_body
        return params


def _messages(context: ContextWindow) -> list[ChatCompletionMessageParam]:
    # Our messages carry OpenRouter extensions (reasoning, reasoning_details) that the
    # SDK's TypedDicts don't model; the SDK sends them through unchanged.
    return cast(
        "list[ChatCompletionMessageParam]",
        normalize_in(context),
    )


def _request(config: OpenrouterConfig, tools: Sequence[ToolSpec] | None) -> dict[str, Any]:
    params = config.to_params()
    if tools:
        params["tools"] = normalize_tools(tools)
    return params


def _provider_error(e: openai.APIError) -> ProviderError:
    """The provider's own error message, not the SDK's rendering of the raw response:
    that one includes account identifiers (OpenRouter's user_id), which must not end
    up in traces. For the same reason callers raise this `from None`."""
    status = getattr(e, "status_code", None)
    body = e.body if isinstance(e.body, dict) else {}
    message = body.get("message") if isinstance(body.get("message"), str) else e.message
    prefix = f"HTTP {status}: " if status is not None else ""
    return ProviderError(f"{prefix}{message}", status_code=status)


def _raise_stream_error(raw: Any) -> None:
    """OpenRouter reports errors after streaming has begun as an `error` field and/or
    finish_reason "error" on a chunk, since the HTTP status (200) is already sent."""
    error = getattr(raw, "error", None)
    choices = getattr(raw, "choices", None) or []
    if error is None and not any(getattr(c, "finish_reason", None) == "error" for c in choices):
        return
    message = error.get("message") if isinstance(error, dict) else None
    raise ProviderError(f"stream error: {message or 'provider reported an error mid-stream'}")


def _default_headers(settings: Settings) -> dict[str, str]:
    headers: dict[str, str] = {}
    if settings.http_referer:
        headers["HTTP-Referer"] = settings.http_referer
    if settings.x_title:
        headers["X-Title"] = settings.x_title
    return headers


class OpenrouterClient:
    def __init__(self, settings: Settings) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            default_headers=_default_headers(settings),
        )

    async def complete(
        self,
        context: ContextWindow,
        config: OpenrouterConfig,
        tools: Sequence[ToolSpec] | None = None,
    ) -> Message:
        """The whole response as one Message (a collected stream)."""
        return await collect(self.stream(context, config, tools))

    async def stream(
        self,
        context: ContextWindow,
        config: OpenrouterConfig,
        tools: Sequence[ToolSpec] | None = None,
    ) -> AsyncGenerator[MessageChunk]:
        try:
            response = await self._client.chat.completions.create(
                messages=_messages(context),
                stream=True,
                stream_options={"include_usage": True},
                **_request(config, tools),
            )
        except openai.APIError as e:
            raise _provider_error(e) from None
        raw_chunks = response.__aiter__()
        finished = False
        while True:
            try:
                raw = await anext(raw_chunks)
            except StopAsyncIteration:
                break
            except openai.APIError as e:
                raise _provider_error(e) from None
            except Exception as e:
                raise ProviderError(f"stream failed: {type(e).__name__}: {e}") from e
            _raise_stream_error(raw)
            chunk = normalize_chunk(raw)
            finished = finished or chunk.stop_reason is not None
            yield chunk
        if not finished:
            raise ProviderError("stream ended before the response finished")


async def collect(stream: AsyncIterator[MessageChunk]) -> Message:
    """Fold a chunk stream into one Message (for when per-chunk access isn't needed)."""
    acc = MessageChunk()
    async for chunk in stream:
        acc = acc + chunk
    return acc.convert()
