# Development

```bash
conda create -n protonous python=3.13
conda activate protonous
pip install -e ".[dev]"
cp .env.example .env        # OPENROUTER_API_KEY
```

## Tests

```bash
pytest                      # offline; fakes stand in for the provider
ruff check .
```

Live smokes call a real model and are deselected by default:

```bash
PROTONOUS_TEST_MODEL='<model-id>' pytest -m model
```

Quote the model id: OpenRouter's alias ids start with `~`, which an unquoted shell
expands as a home directory.

## Layout

```
src/protonous/
  context.py      Message, ToolCall, ReasoningDetail, Usage, MessageChunk, ToolSpec
  layers.py       Layer, StaticLayer, SingleMessageLayer, SlidingWindowLayer, ContextWindow
  normalizers.py  the only code that speaks the OpenAI/OpenRouter wire format
  client.py       OpenrouterClient, OpenrouterConfig
  tools.py        Tool, by_name, execute
  loop.py         run, Step, RunResult, append_to, check_pairing
  trace.py        Session, Tracer, load_session, load_trace
  errors.py       ProtonousError and subclasses
  cli.py          the `protonous` command
  settings.py     connection secrets from .env
examples/
  demo.py         everything end to end
docs/             this wiki
```

## Conventions

- Data types are frozen Pydantic models, not dataclasses.
- Only `normalizers.py` knows the provider's wire format.
- Errors are `ProtonousError`s, raised where they happen; `run()` is the only handler.
- Scope: one agent, one loop, tools, traces. Personas, memory, containment and
  environments belong to the projects that use protonous.
