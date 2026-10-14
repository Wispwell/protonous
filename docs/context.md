# Context

What the model sees on each call. Two modules: `context.py` holds the data types,
`layers.py` holds how they are assembled into a window.

## Messages

All message types are frozen Pydantic models.

| Type | What it is |
|---|---|
| `Message` | one message: `role`, `content`, plus `reasoning`, `reasoning_details`, `tool_calls`, `tool_call_id`, `extras` |
| `ToolCall` | one function call requested by the model; `function_arguments` is the raw JSON string |
| `ReasoningDetail` | one structured reasoning block (`reasoning.text`, `.summary` or `.encrypted`) |
| `Usage` | token accounting for one response |
| `MessageChunk` | a streaming delta; fold a stream with `+`, finish with `convert()` |
| `ToolSpec` | a tool as the model sees it: name, description, JSON Schema. No code |

`Role` is `system`, `user`, `assistant` or `tool`. `extras` is free-form; protonous
itself only sets `extras["error"] = True` on error tool results.

## Layers

A `Layer` is a retention policy: `add(msg)` stores a message, `view()` returns what
the model should see.

| Layer | Keeps | `add` |
|---|---|---|
| `StaticLayer(messages)` | the messages given at construction | raises `LayerError` |
| `SingleMessageLayer()` | only the latest message | replaces; rejects tool calls and tool results |
| `SlidingWindowLayer(max_messages=None)` | the most recent messages; `None` = all | appends, then evicts from the front |

`SlidingWindowLayer` evicts **whole turns**, never single messages: a turn is a
non-tool message plus the tool results that follow it. A tool call is therefore
never separated from its results, which providers would reject. The newest turn is
always kept, even if it alone exceeds `max_messages`.

`SingleMessageLayer` refuses tool traffic for the same reason: replacing one half of
a call/result pair would leave the other half orphaned. Use it for things like a
state summary that is refreshed every step.

## ContextWindow

```python
window = ContextWindow(layers=[
    StaticLayer([Message(role=Role.SYSTEM, content="...")]),   # rules
    state,                                                     # SingleMessageLayer
    history,                                                   # SlidingWindowLayer
])
window.view()   # the layers' views, concatenated in order
```

The window is only an ordered list of layers. Which layer a new message goes into is
decided by the loop's `update` callback, not by the window (see [loop](loop.md)).

## Reasoning

Reasoning arrives on the assistant `Message` (`reasoning`, `reasoning_details`) and
travels with it. When the window is sent back, each assistant message carries its
`reasoning_details` back out, unmodified and in order, which is how OpenRouter keeps
reasoning across tool-calling turns; if a provider gave only a raw `reasoning`
string, that is sent instead. `run(..., replay_all_reasoning=False)` sends reasoning
only on the latest assistant message.

Stored messages are never changed, so traces always keep all reasoning. Models that
return encrypted reasoning round-trip fine but can't be read.

## Provider shape

`normalizers.py` is the only code that speaks the OpenAI/OpenRouter wire format:
`normalize_in` (window → messages), `normalize_tools` (specs → tools),
`normalize_response` and `normalize_chunk` (provider → our types). Everything else
works with protonous types only.
