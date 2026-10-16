import pytest

from protonous import Message, NotASessionError, ProviderError, Role, Session, Step, ToolCall, Usage
from protonous.cli import follow, main
from protonous.trace import (
    RunEnded,
    RunFailed,
    RunStarted,
    SessionFollower,
    StepRecorded,
    TraceFormatError,
)


def _step(text, usage=None):
    return Step(assistant=Message(role=Role.ASSISTANT, content=text, usage=usage))


def test_poll_returns_only_what_is_new(tmp_path):
    session = Session.create(tmp_path / "s", {})
    tracer = session.tracer(meta={"m": 1})
    follower = SessionFollower(session.path)
    assert follower.poll() == [RunStarted(run="run-0001", meta={"m": 1})]
    assert follower.poll() == []

    tracer.step(_step("hello"))
    [event] = follower.poll()
    assert isinstance(event, StepRecorded) and event.index == 0
    assert event.step.assistant.content == "hello"

    tracer.end("done")
    assert follower.poll() == [RunEnded(run="run-0001", stop_reason="done", steps=1)]


def test_poll_picks_up_runs_started_later_and_failures(tmp_path):
    session = Session.create(tmp_path / "s", {})
    follower = SessionFollower(session.path)
    assert follower.poll() == []
    tracer = session.tracer()
    with pytest.raises(ProviderError), tracer.recording():
        raise ProviderError("HTTP 400: bad model")
    events = follower.poll()
    assert [type(e) for e in events] == [RunStarted, RunFailed]
    assert events[1].error.message == "HTTP 400: bad model"
    assert not any(e for e in follower.poll())  # errors.jsonl is not read as a run


def test_a_half_written_line_waits_for_the_next_poll(tmp_path):
    session = Session.create(tmp_path / "s", {})
    session.tracer()
    follower = SessionFollower(session.path)
    follower.poll()
    run_file = session.path / "run-0001.jsonl"
    with run_file.open("a") as f:
        f.write('{"type": "end", "stop_reason": "do')  # the writer hasn't finished the line
    assert follower.poll() == []
    with run_file.open("a") as f:
        f.write('ne", "steps": 0}\n')
    assert follower.poll() == [RunEnded(run="run-0001", stop_reason="done", steps=0)]


def test_follower_needs_a_session(tmp_path):
    (tmp_path / "plain").mkdir()
    with pytest.raises(NotASessionError):
        SessionFollower(tmp_path / "plain")


def test_unknown_record_types_are_rejected(tmp_path):
    session = Session.create(tmp_path / "s", {})
    (session.path / "run-0001.jsonl").write_text('{"type": "mystery"}\n')
    with pytest.raises(TraceFormatError):
        SessionFollower(session.path).poll()


def test_follow_prints_events_as_runs_write_them(tmp_path):
    session = Session.create(tmp_path / "s", {})
    first = session.tracer(meta={"task": "add"})
    call = Message(
        role=Role.ASSISTANT, content=None, reasoning="add them",
        tool_calls=(ToolCall(id="c1", function_name="add", function_arguments='{"a":1,"b":2}'),),
        usage=Usage(prompt_tokens=100, completion_tokens=10, total_tokens=110, cost=0.001),
    )
    first.step(Step(assistant=call, tool_results=(
        Message(role=Role.TOOL, content="3", tool_call_id="c1"),)))

    # between polls, the "running" agents write more: first finishes, a second starts
    writes = [
        lambda: (first.step(_step("3", Usage(prompt_tokens=45, completion_tokens=5, total_tokens=50, cost=0.0005))), first.end("done")),
        lambda: session.tracer().step(_step("parallel run")),
    ]
    out = []
    follow(session.path, polls=3, out=out.append, sleep=lambda _: writes.pop(0)())
    assert out == [
        'run-0001 · started · meta {"task": "add"}',
        "run-0001 [0] thinks: add them",
        'run-0001 [0] add({"a":1,"b":2}) → 3',
        "run-0001 [1] says: 3",
        "run-0001 · done · 2 step(s) · 160 tokens (145 in, 15 out) · $0.0015",
        "run-0002 · started",
        "run-0002 [0] says: parallel run",
    ]


def test_follow_missing_session_exits_with_an_error(tmp_path, capsys):
    assert main(["replay", "nope", "--root", str(tmp_path), "--follow"]) == 1
    assert "not a session" in capsys.readouterr().err
