# Sessions and traces

A session is a directory of runs that share initial data. A run is one `run()` call.

```python
session = Session.create("sessions/exp-1", {"config": config, "tools": [t.spec for t in tools]})
tracer = session.tracer(meta={"condition": "a", "seed": 3})
await run(..., tracer=tracer)

session = Session.open("sessions/exp-1")   # later: add more runs
```

- `Session.create(path, initial)` makes the directory and writes `session.json`.
  Fails with `SessionExistsError` if the directory exists.
- `Session.open(path)` opens an existing session. Fails with `NotASessionError` if
  there is no `session.json`.
- `session.tracer(name=None, meta=None)` creates a new run file. Without a name it
  claims the next free `run-NNNN`, atomically, so runs started in parallel never
  share a file. Names may use letters, digits, `.`, `_`, `-`; `errors` is reserved.
- `initial` and `meta` can hold anything JSON-serializable, Pydantic models included.

A `Tracer(path, meta)` can also be used on its own, without a session.

## Layout

```
sessions/exp-1/
  session.json      {"created": ..., "initial": {...}}
  run-0001.jsonl
  run-0002.jsonl
  errors.jsonl      only if something failed
```

A run file:

```
{"type": "run", "name": "run-0001", "meta": {...}}
{"type": "step", "index": 0, "step": {...}}            one line per Step
{"type": "end", "stop_reason": "done", "steps": 3}     run() returned
```

or, if `run()` raised, an error line instead of the end line:

```
{"type": "error", "error": "ProviderError", "message": "HTTP 400: ..."}
```

`errors.jsonl` holds one line per failure across the session's runs, with the full
traceback: `{"run", "time", "error", "message", "traceback"}`.

Every line is appended and the file closed immediately, so a crash loses nothing.
Each step is written before `update` sees it. A run file with neither an end nor an
error line belongs to a process that was killed outright.

## Loading

```python
s = load_session("sessions/exp-1")
s.initial                      # dict
s.runs["run-0001"].steps       # list[Step], exactly as recorded
s.runs["run-0001"].stop_reason # "done" | "max_steps" | None (didn't finish)
s.runs["run-0001"].error       # RunError(error, message) | None
s.runs["run-0001"].usage       # Usage summed over the run's steps
s.usage                        # Usage summed over the whole session
s.errors                       # list[ErrorRecord], with tracebacks
```

`load_trace(path)` loads a single run file. For reading a session at a glance, see
[cli](cli.md).
