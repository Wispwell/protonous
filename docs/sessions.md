# Sessions and traces

A session is a directory of runs that share initial data. A run is one pass of your
agent loop, recorded by a `Tracer`.

```python
session = Session.create("sessions/exp-1", {"config": config, "tools": [t.spec for t in tools]})
tracer = session.tracer(meta={"condition": "a", "seed": 3})
with tracer.recording():
    ...                                    # tracer.step(step) for every step
    tracer.end("done")                     # any stop reason your loop uses

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
{"type": "end", "stop_reason": "done", "steps": 3}     tracer.end("done")
```

or, if something raised inside `tracer.recording()`, an error line instead:

```
{"type": "error", "error": "ProviderError", "message": "HTTP 400: ..."}
```

`errors.jsonl` holds one line per failure across the session's runs, with the full
traceback: `{"run", "time", "error", "message", "traceback"}`.

Every line is appended and the file closed immediately, so a crash loses nothing.
Each step is on disk as soon as `tracer.step` returns. A run file with neither an end nor an
error line belongs to a process that was killed outright.

## Loading

```python
s = load_session("sessions/exp-1")
s.initial                      # dict
s.runs["run-0001"].steps       # list[Step], exactly as recorded
s.runs["run-0001"].stop_reason # what tracer.end got, or None (didn't finish)
s.runs["run-0001"].error       # RunError(error, message) | None
s.runs["run-0001"].usage       # Usage summed over the run's steps
s.usage                        # Usage summed over the whole session
s.errors                       # list[ErrorRecord], with tracebacks
```

`load_trace(path)` loads a single run file. For reading a session at a glance, see
[cli](cli.md).

## Watching runs live

`SessionFollower` reads a session while its runs are still writing, for a live
view of your own (a dashboard, a chat UI):

```python
follower = SessionFollower("sessions/exp-1")
while True:
    for event in follower.poll():      # only what was written since the last poll
        ...                            # RunStarted | StepRecorded | RunEnded | RunFailed
    await asyncio.sleep(1)
```

Each `poll()` covers every run file, runs started since the last poll included. A
line still being written is left for the next poll, so a half-written record is
never parsed. `protonous replay --follow` is built on it.
