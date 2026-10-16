from protonous.cli import main
from protonous.context import Message, Role, ToolCall, Usage
from protonous.errors import ProviderError
from protonous.loop import Step
from protonous.trace import Session


def _session(root):
    session = Session.create(root / "s1", {"config": {"model": "x"}, "tools": []})
    tracer = session.tracer(meta={"task": "add"})
    call = Message(
        role=Role.ASSISTANT,
        content=None,
        usage=Usage(prompt_tokens=1200, completion_tokens=34, total_tokens=1234, cost=0.0021),
        reasoning="I should\nadd these",
        tool_calls=(
            ToolCall(id="c1", function_name="add", function_arguments='{"a":1,"b":2}'),
            ToolCall(id="c2", function_name="nope", function_arguments="{}"),
        ),
    )
    results = (
        Message(role=Role.TOOL, content="3", tool_call_id="c1"),
        Message(role=Role.TOOL, content="Error: unknown tool", tool_call_id="c2",
                extras={"error": True}),
    )
    tracer.step(Step(assistant=call, tool_results=results))
    tracer.step(Step(assistant=Message(role=Role.ASSISTANT, content="x" * 500)))
    session.tracer().error(ProviderError("HTTP 400: bad model", status_code=400))
    return session


def test_replay_prints_runs_and_events(tmp_path, capsys):
    _session(tmp_path)
    assert main(["replay", "s1", "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "session s1" in out and "2 run(s)" in out
    assert "initial: config, tools" in out
    assert "run-0001 · unfinished · 2 step(s) · 1,234 tokens (1,200 in, 34 out) · $0.0021" in out
    assert "session s1 · " in out and "2 run(s) · 1,234 tokens" in out
    assert "run-0002 · ERROR ProviderError: HTTP 400: bad model · 0 step(s)\n" in out + "\n"
    assert '  meta: {"task": "add"}' in out
    assert "  [0] thinks: I should add these" in out  # flattened to one line, indent kept
    assert '  [0] add({"a":1,"b":2}) → 3' in out
    assert "  [0] nope({}) ✗ Error: unknown tool" in out
    assert "run-0002 · ERROR ProviderError: HTTP 400: bad model · 0 step(s)" in out
    assert "1 error(s); tracebacks in errors.jsonl" in out
    says = next(line for line in out.splitlines() if "[1] says:" in line)
    assert len(says) == 160 and says.endswith("…")


def test_replay_full_does_not_cut(tmp_path, capsys):
    _session(tmp_path)
    main(["replay", "s1", "--root", str(tmp_path), "--full"])
    out = capsys.readouterr().out
    assert "x" * 500 in out


def test_replay_missing_session(tmp_path, capsys):
    assert main(["replay", "nope", "--root", str(tmp_path)]) == 1
    assert "no session at" in capsys.readouterr().err
