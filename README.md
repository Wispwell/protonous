# protonous

A lightweight agent framework over OpenRouter: typed context, one provider layer, one loop, traced runs.

```python
tools = Toolbox()

@tools.tool
async def add(a: float, b: float) -> str:
    """Add two numbers."""
    return str(a + b)

history = SlidingWindowLayer()
history.add(Message(role=Role.USER, content="Compute (17 + 25) * 3."))
window = ContextWindow(layers=[StaticLayer([system_prompt]), history])

session = Session.create("sessions/demo", {"config": config})
result = await run(
    OpenrouterClient(get_settings()), window, config, tools,
    update=append_to(history), max_steps=20, tracer=session.tracer(),
)
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

- **Context in layers**: static rules, a single refreshed state message, a sliding
  history that never splits a tool call from its result.
- **Native tool calling**: decorate a typed async function to add it to an agent's
  `Toolbox`; model mistakes come back to the model as tool results instead of
  crashing the run.
- **Reasoning round-trip**: reasoning is sent back to the provider across tool turns
  and kept in full in every trace.
- **Sessions**: one directory per session, one JSONL file per run, failures recorded
  with their tracebacks.

## Docs

- [Context](docs/context.md): messages, layers, the window, reasoning
- [Tools](docs/tools.md): defining tools, what the model sees when they fail
- [The loop](docs/loop.md): `run`, `update`, steps
- [Sessions and traces](docs/sessions.md): files, format, loading
- [Errors](docs/errors.md): the hierarchy and the one handler
- [CLI](docs/cli.md): `protonous replay`
- [Development](docs/development.md): setup, tests, layout, conventions

`examples/demo.py` runs everything end to end.

## License

Apache-2.0
