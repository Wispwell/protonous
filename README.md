# protonous

A lightweight agent framework over OpenRouter: typed context, one provider layer, traced runs, and a loop you write yourself.

```python
tools = Toolbox()

@tools.tool
async def add(a: float, b: float) -> str:
    """Add two numbers."""
    return str(a + b)

model = Model(OpenrouterClient(get_settings()), OpenrouterConfig(model="..."))
history = SlidingWindowLayer()
history.add(Message(role=Role.USER, content="Compute (17 + 25) * 3."))
window = ContextWindow(layers=[StaticLayer([system_prompt]), history])
tracer = Session.create("sessions/demo", {"model": model.config}).tracer()

with tracer.recording():
    for _ in range(20):
        step = await model.step(window, tools)   # one streamed call + its tool calls
        tracer.step(step)
        history.extend(step.messages)
        if not step.tool_results:
            break
    tracer.end("done")
```

```bash
protonous replay demo
```

```
run-0001 · done · 4 step(s)
  [0] add({"a":17,"b":25}) → 42.0
  [1] multiply({"a":42,"b":3}) → 126.0
  [3] says: (17 + 25) * 3 = 126
```

- **Your loop, built from the model's calls**: `model.call`, `model.step` and
  `model.stream`. When to stop,
  what goes into the window and when to trim it are yours to decide, so a long
  autonomous task and a chat with many short turns can manage inference differently.
- **Context in layers**: static rules, a single refreshed state message, a sliding
  history that never splits a tool call from its result.
- **Native tool calling**: decorate a typed async function to add it to an agent's
  `Toolbox`; model mistakes come back to the model as tool results instead of
  crashing the run.
- **Streaming throughout**: every call streams; `model.stream()` is an async iterator
  for live UIs, and a stream that fails or is stopped never returns half a reply.
- **Reasoning round-trip**: reasoning is sent back to the provider across tool turns
  and kept in full in every trace.
- **Usage per turn**: the provider-reported tokens and cost of every step, summed per
  run and session.
- **Sessions**: one directory per session, one JSONL file per run, failures recorded
  with their tracebacks.

## Docs

- [Context](docs/context.md): messages, layers, the window, reasoning
- [Tools](docs/tools.md): defining tools, what the model sees when they fail
- [Model and the agent loop](docs/model.md): `call`, `step`, `stream`, writing the loop, caching
- [Sessions and traces](docs/sessions.md): files, format, loading
- [Errors](docs/errors.md): the hierarchy and the one handler
- [CLI](docs/cli.md): `protonous replay`
- [Development](docs/development.md): setup, tests, layout, conventions

`examples/demo.py` runs everything end to end.

## License

Apache-2.0
