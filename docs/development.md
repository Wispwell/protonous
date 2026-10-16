# Development

```bash
conda create -n protonous python=3.13
conda activate protonous
pip install -e ".[dev]"
cp .env.example .env        # OPENROUTER_API_KEY
```

Then, once per clone:

```bash
pre-commit install         # run the checks on every commit
```

## Checks

```bash
pre-commit run --all-files  # ruff, strict mypy on src/, file hygiene, private-key scan
pytest                      # offline; fakes stand in for the provider
```

CI (`.github/workflows/ci.yml`) runs exactly these two commands on every push and
pull request. The hook versions are pinned in `.pre-commit-config.yaml`, so local
and CI checks are identical.

Live smokes call a real model and are deselected by default:

```bash
PROTONOUS_TEST_MODEL='<model-id>' pytest -m model
```

Quote the model id: OpenRouter's alias ids start with `~`, which an unquoted shell
expands as a home directory.

## Layout

```
src/protonous/
  context.py      Message, ToolCall, ReasoningDetail, Usage, MessageChunk, ToolSpec, Step,
                  check_pairing
  layers.py       Layer, StaticLayer, SingleMessageLayer, SlidingWindowLayer, ContextWindow
  normalizers.py  the only code that speaks the OpenAI/OpenRouter wire format
  client.py       OpenrouterClient, OpenrouterConfig
  tools.py        Toolbox, tool (decorator), Tool, run_tools, by_name, execute
  model.py        Model (call, step, stream), ReplyStream
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
- `src/` passes `mypy --strict`.
- Only `normalizers.py` knows the provider's wire format.
- Errors are `ProtonousError`s, raised where they happen; `Tracer.recording()` is the
  only handler.
- The agent loop belongs to the caller: protonous provides the calls and never
  manages the window.
- Scope: one agent, tools, traces. Personas, memory, containment and
  environments belong to the projects that use protonous.
