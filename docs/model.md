# Model and the agent loop

protonous gives you the pieces of an agent and leaves the loop to you. How many
steps, when to stop, what goes into the window and when to trim it are decisions
that differ between systems (a long autonomous task, a chat with many short turns,
an experiment), so none of them are made for you.

## Model

```python
model = Model(OpenrouterClient(get_settings()), OpenrouterConfig(model="..."))
```

A `Model` is what to call (the client) and how (the per-request config: `model`,
`temperature`, `max_tokens`, `reasoning_effort`, `tool_choice`, `extra_body`, ...).
It has three calls, all the same model call underneath:

| Call | Does | Returns |
|---|---|---|
| `await model.call(window, tools)` | one model call over the window, offering the tools | the reply, a `Message` |
| `await model.step(window, tools)` | `call`, then runs the reply's tool calls in order | a `Step` |
| `model.stream(window, tools)` | the same call, read chunk by chunk (see [Streaming](#streaming)) | a `ReplyStream` |

None of them touches the window. The one check is on the request itself: every tool
call in the window must have its result, or `ToolPairingError` is raised before
anything is sent.

```python
class Step(BaseModel):
    assistant: Message                  # the reply: content, reasoning, tool_calls, usage
    tool_results: tuple[Message, ...]   # in call order; empty if the model called no tools

    messages   # the reply followed by its results: what to add to a history
    usage      # what this step's model call consumed
```

## Writing the loop

The common shape: step until the model answers without calling tools.

```python
history = SlidingWindowLayer()
history.add(Message(role=Role.USER, content=task))
window = ContextWindow(layers=[StaticLayer([system_prompt]), history])
tracer = session.tracer()

stop = "max_steps"
with tracer.recording():                  # failures are recorded, then re-raised
    for _ in range(20):
        step = await model.step(window, tools)
        tracer.step(step)
        history.extend(step.messages)
        if not step.tool_results:
            stop = "done"
            break
    tracer.end(stop)
```

Everything a framework would usually hide is a line you can see and change: where
a step's messages go (`history.extend`), what else is refreshed each step (a state
layer, see below), when the loop stops, and what is traced. A chat system writes a
different loop, one `step` per user message, perhaps; an experiment might stop on a
condition of its environment.

### Keeping the provider's cache warm

Providers cache the start of the window: a request whose beginning matches the
previous one pays a fraction of the input price for that part (see
[context](context.md#usage)). The loop decides how much of that you get:

- **Only append during a run.** Each step's window is then the previous one plus
  new messages, so everything before them is a cache hit.
- **Put layers that change every step last.** A refreshed state message at the end
  only invalidates itself; placed before the history, it invalidates everything
  after it.
- **Trim between runs, not during them.** Evicting from the front of a history
  shifts the whole window, so the next request starts cold.

Some providers (Anthropic, Qwen) cache only when asked:
`OpenrouterConfig(..., extra_body={"cache_control": {"type": "ephemeral"}})`.

## Streaming

Every call is streamed; `call` and `step` just read the stream for you. To see the
chunks as they arrive (content, reasoning and tool-call fragments), as a chat UI
does, read it yourself:

```python
async with model.stream(window, tools) as stream:
    async for chunk in stream:
        await ui.send(chunk.content)
    step = await stream.step()        # or: reply = stream.reply
```

- `stream.reply` is the assembled `Message`, once the stream has been read to the
  end. `await stream.collect()` and `await stream.step()` read whatever is left
  first, so they work after a partial read too.
- Stopping early is fine: leaving the `async with` block (or `await
  stream.aclose()`) closes the connection right away. Use `async with` whenever the
  loop might stop reading, e.g. when a user presses stop.
- A stream that was closed early or failed has no reply: `reply`, `collect()` and
  `step()` raise `StreamIncompleteError`. A half-received reply is never returned.

A stream that fails partway raises `ProviderError` while it is being read, whether
the provider reports an error after streaming began, the connection drops, or the
stream stops before the response finished. The window is checked
(`ToolPairingError`) when `model.stream` is called, before any request is sent.

## Below the model

`OpenrouterClient` is the transport: `client.stream(window, config, specs)` yields
raw chunks and `client.complete(...)` collects them into one `Message`. It declares tool
specs but never runs anything; `run_tools(tools, reply)` (in `protonous.tools`) does
that, and `model.step` is just the two together.
