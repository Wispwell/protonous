# The loop

```python
result = await run(
    client, window, config, tools,
    update=append_to(history),
    max_steps=20,
    replay_all_reasoning=True,   # default
    tracer=session.tracer(),     # optional, see sessions.md
    on_chunk=print_chunk,        # optional, see Streaming
)
```

Native tool calling, one step at a time:

1. check that every tool call in the window has its result (`check_pairing`)
2. call the model with the window and the tools' specs
3. run the reply's tool calls, in order, one after another
4. record the step (if there is a tracer)
5. `window = update(window, step)`

It stops when the model answers without tool calls (`stop_reason="done"`) or after
`max_steps` (`"max_steps"`).

## Step, RunResult

```python
class Step(BaseModel):
    assistant: Message                  # the reply: content, reasoning, tool_calls
    tool_results: tuple[Message, ...]   # in call order; empty on the final step

class RunResult(BaseModel):
    window: ContextWindow               # what the last update returned
    steps: tuple[Step, ...]
    stop_reason: Literal["done", "max_steps"]
```

Both are frozen.

## update

`update(window, step) -> window` is the only thing that decides where a step's
messages go. The loop knows nothing about layers.

`append_to(layer)` is the common case: add the assistant message, then its tool
results, to one layer. For more, write your own and compose:

```python
remember = append_to(history)

def update(window, step):
    window = remember(window, step)
    state.add(render_state())   # refresh a SingleMessageLayer every step
    return window
```

`update` receives the assistant message together with all of its tool results, so
keeping calls and results together is natural. If a custom `update` does split them,
`check_pairing` raises `ToolPairingError` before the next model call, instead of the
provider rejecting the request.

## Streaming

Every reply is streamed; there is no non-streaming path. `on_chunk` receives each
`MessageChunk` as it arrives (content, reasoning and tool-call fragments), which is
what a chat UI needs; leave it out and the loop just assembles the full reply. It
is synchronous, so to hand chunks to async code, put them on a queue:

```python
queue: asyncio.Queue[MessageChunk] = asyncio.Queue()
await run(..., on_chunk=queue.put_nowait)
```

A stream that fails partway raises `ProviderError`, whether the provider reports an
error after streaming began, the connection drops, or the stream simply stops
before the response finished. The half-received reply is never turned into a step,
so traces only contain complete replies; chunks already passed to `on_chunk` were
shown, but the run fails with the error.

## Reasoning replay

`replay_all_reasoning=True` sends each assistant message's reasoning back to the
provider; `False` sends it only on the latest one, which keeps input cost flat on
long runs. See [context](context.md#reasoning).

## Client

`OpenrouterClient(get_settings())` holds the connection: `client.stream(...)` yields
chunks, and `client.complete(...)` collects them into one `Message`. `OpenrouterConfig` holds
per-request generation parameters (`model`, `temperature`, `max_tokens`,
`reasoning_effort`, `tool_choice`, `extra_body`, ...). Tools are not part of the
config: the client receives their specs separately and only declares them; it never
runs anything.
