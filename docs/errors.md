# Errors

Errors are raised where they happen and handled in one place.

## The hierarchy

Everything protonous raises is a `ProtonousError` (`protonous.errors`):

| Error | Raised when | Also a |
|---|---|---|
| `ConfigurationError` | an argument makes no sense: bad window size, foreign layer, unsafe run name | `ValueError` |
| `LayerError` | a message is added to a layer that refuses it | |
| `ToolPairingError` | the window holds a tool call without its result, or the reverse | |
| `ToolDefinitionError` | duplicate tool names; a tool returning a non-`str` | |
| `ProviderError` | the provider returned an HTTP error, couldn't be reached, or a stream failed or ended early; has `status_code` | |
| `SessionExistsError` | `Session.create` on an existing directory | `FileExistsError` |
| `NotASessionError` | `Session.open` on a directory without `session.json` | `FileNotFoundError` |
| `RunExistsError` | a run file with that name already exists | `FileExistsError` |
| `TraceFormatError` | a run file has no header line | `ValueError` |
| `StreamIncompleteError` | a `ReplyStream`'s reply was asked for, but it was closed early or failed | |

`ProviderError` wraps the SDK's exceptions, so code using protonous never imports
`openai`. Its message is the provider's own error message; the SDK's raw rendering
is deliberately dropped (and not chained), because it contains account identifiers
(OpenRouter's `user_id`) that must not end up in traces.

## The one handler

`with tracer.recording():` catches anything that escapes the block, including
`KeyboardInterrupt` and `asyncio.CancelledError`. It records the failure (an `error`
line in the run file, the traceback in the session's `errors.jsonl`, see
[sessions](sessions.md#layout)), then re-raises the exception unchanged. Wrap your
agent loop in it. Nothing is swallowed, and nothing else in protonous catches
exceptions.

## The one deliberate exception

A tool that fails, or a model that calls a tool wrongly, is not an error of the
program. It is information the agent should see. Those become `Error: ...` tool
results (see [tools](tools.md#execution)) and the run continues.
