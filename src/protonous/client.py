"""OpenRouter client (async, via the OpenAI SDK) + per-request config.

The client is agent-agnostic: it takes a ContextWindow (plus optional ToolSpecs, which
it declares but never executes) and returns our Message /
MessageChunk. All provider-shaped mapping lives in normalizers. Folding a stream is
the caller's job (via MessageChunk `+`); `collect()` is the convenience for when
per-chunk access isn't needed.
"""
from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel

from .context import Message, MessageChunk, ToolSpec
from .errors import ProviderError
from .layers import ContextWindow
from .normalizers import normalize_chunk, normalize_in, normalize_response, normalize_tools
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
        *,
        replay_all_reasoning: bool = True,
    ) -> Message:
        try:
            response = await self._client.chat.completions.create(
                messages=normalize_in(context, replay_all_reasoning=replay_all_reasoning),
                stream=False,
                **_request(config, tools),
            )
        except openai.APIError as e:
            raise _provider_error(e) from None
        return normalize_response(response)

    async def stream(
        self,
        context: ContextWindow,
        config: OpenrouterConfig,
        tools: Sequence[ToolSpec] | None = None,
        *,
        replay_all_reasoning: bool = True,
    ) -> AsyncIterator[MessageChunk]:
        try:
            stream = await self._client.chat.completions.create(
                messages=normalize_in(context, replay_all_reasoning=replay_all_reasoning),
                stream=True,
                stream_options={"include_usage": True},
                **_request(config, tools),
            )
            async for chunk in stream:
                yield normalize_chunk(chunk)
        except openai.APIError as e:
            raise _provider_error(e) from None


async def collect(stream: AsyncIterator[MessageChunk]) -> Message:
    """Fold a chunk stream into one Message (for when per-chunk access isn't needed)."""
    acc = MessageChunk()
    async for chunk in stream:
        acc = acc + chunk
    return acc.convert()
