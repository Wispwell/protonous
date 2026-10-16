# CLI

```bash
protonous replay <session_id>           # the session at ./sessions/<session_id>
protonous replay <session_id> --root DIR
protonous replay <session_id> --full    # don't cut long text
```

`replay` prints a session without reading the raw JSONL: one line per run, one line
per event.

```
session demo-20261016-163859 · created 2026-10-16 16:38:59 UTC · 1 run(s) · 3,562 tokens (3,325 in, 237 out) · $0.0005
initial: config, window, tools

run-0001 · done · 4 step(s) · 3,562 tokens (3,325 in, 237 out) · $0.0005
  meta: {"task": "Compute (17 + 25) * 3. Write the result to the notepad, then state it."}
  [0] thinks: We need compute. Need use tools. First add 17+25, then multiply by 3, …
  [0] add({"a":17,"b":25}) → 42.0
  [1] multiply({"a":42,"b":3}) → 126.0
  [2] write_note({"text":"(17 + 25) * 3 = 126"}) → noted (1 note(s) total)
  [3] says: (17 + 25) * 3 = 126
```

- Run status is `done`, `max_steps`, `ERROR <type>: <message>`, or `unfinished`
  (no end or error line).
- Tokens and cost are what the provider reported, summed per run and per session;
  they're left out when nothing was reported.
- `→` is a tool result, `✗` an error result.
- Text is flattened to one line and cut at 160 characters unless `--full`.
- Tracebacks are not printed; the summary points to `errors.jsonl`.

A missing session exits with status 1.
