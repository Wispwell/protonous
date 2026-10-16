"""Every exception protonous raises. Catch `ProtonousError` to catch all of them.

Raised wherever they happen; the one place that handles them is
`Tracer.recording()`, which records the failure in the run's trace and re-raises
it unchanged.
"""
from __future__ import annotations


class ProtonousError(Exception):
    """Base of every protonous exception."""


class ConfigurationError(ProtonousError, ValueError):
    """An argument protonous was given makes no sense (a bad size, a foreign layer,
    an unsafe run name)."""


class LayerError(ProtonousError):
    """A message was added to a layer that does not accept it."""


class ToolPairingError(ProtonousError):
    """The window holds a tool call without its result, or a result without its call."""


class ToolDefinitionError(ProtonousError):
    """A tool is defined or behaves wrongly: duplicate names, a non-string return."""


class ProviderError(ProtonousError):
    """The model provider failed: an HTTP error status, or no connection at all."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class SessionExistsError(ProtonousError, FileExistsError):
    """Session.create on a directory that already exists."""


class NotASessionError(ProtonousError, FileNotFoundError):
    """Session.open on a directory without a session.json."""


class RunExistsError(ProtonousError, FileExistsError):
    """A run file with that name already exists."""


class TraceFormatError(ProtonousError, ValueError):
    """A trace file is malformed (e.g. no run header line)."""


class StreamIncompleteError(ProtonousError):
    """A ReplyStream's reply was asked for, but the stream wasn't read to the end:
    it was closed early, or it failed (the failure itself was raised while reading)."""
