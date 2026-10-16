"""Context layers: how a ContextWindow is assembled from Messages.

Each Layer is a retention policy (static, single latest message, sliding window). A
ContextWindow is an ordered list of Layers; the model sees their views concatenated
in order. Turning that view into provider messages is the normalizers' job.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from .context import Message, Role
from .errors import ConfigurationError, LayerError

__all__ = [
    "ContextWindow",
    "Layer",
    "LayerError",
    "SingleMessageLayer",
    "SlidingWindowLayer",
    "StaticLayer",
]


class Layer(ABC):
    """A retention policy over Messages: `add` stores, `view` is what the model sees."""

    @abstractmethod
    def add(self, msg: Message) -> None: ...

    @abstractmethod
    def view(self) -> list[Message]: ...

    def extend(self, messages: Iterable[Message]) -> None:
        """`add` each message in order (e.g. `history.extend(step.messages)`)."""
        for msg in messages:
            self.add(msg)


class StaticLayer(Layer):
    """Messages fixed at construction (e.g. a system prompt). `add` raises."""

    def __init__(self, messages: Iterable[Message] = ()) -> None:
        self._messages = list(messages)

    def add(self, msg: Message) -> None:
        raise LayerError("StaticLayer is fixed at construction")

    def view(self) -> list[Message]:
        return list(self._messages)


class SingleMessageLayer(Layer):
    """Keeps only the latest message (e.g. a state digest replaced every step).

    Rejects tool-call traffic: a lone tool result or a tool-calling assistant message
    would be unpaired once its partner is replaced.
    """

    def __init__(self) -> None:
        self._message: Message | None = None

    def add(self, msg: Message) -> None:
        if msg.role is Role.TOOL or msg.tool_calls:
            raise LayerError("SingleMessageLayer holds plain messages, not tool-call traffic")
        self._message = msg

    def view(self) -> list[Message]:
        return [self._message] if self._message is not None else []


class SlidingWindowLayer(Layer):
    """Keeps the most recent messages, at most `max_messages` (None = unbounded).

    Evicts whole turns from the front, never single messages: a turn is a non-tool
    message plus the tool results that follow it, so a tool call is never separated
    from its results. The newest turn is always kept, even if it alone exceeds the cap.
    """

    def __init__(self, max_messages: int | None = None) -> None:
        if max_messages is not None and max_messages < 1:
            raise ConfigurationError("max_messages must be >= 1 or None")
        self.max_messages = max_messages
        self._messages: list[Message] = []

    def add(self, msg: Message) -> None:
        self._messages.append(msg)
        if self.max_messages is None:
            return
        while len(self._messages) > self.max_messages:
            end = _first_turn_end(self._messages)
            if end == len(self._messages):
                break  # only the newest turn is left: keep it
            del self._messages[:end]

    def view(self) -> list[Message]:
        return list(self._messages)


def _first_turn_end(messages: list[Message]) -> int:
    """Index just past the first turn: its opening message plus any tool results."""
    end = 1
    while end < len(messages) and messages[end].role is Role.TOOL:
        end += 1
    return end


class ContextWindow(BaseModel):
    """An ordered list of Layers. The model sees their views concatenated in order."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    layers: list[Layer] = Field(default_factory=list)

    def view(self) -> list[Message]:
        return [msg for layer in self.layers for msg in layer.view()]
