import asyncio
import json

import pytest
from pydantic import BaseModel

from protonous.client import OpenrouterConfig
from protonous.context import Message, ReasoningDetail, Role, ToolCall
from protonous.errors import (
    ConfigurationError,
    NotASessionError,
    ProviderError,
    RunExistsError,
    SessionExistsError,
)
from protonous.layers import ContextWindow, SlidingWindowLayer, StaticLayer
from protonous.loop import RunResult, Step, append_to, run
from protonous.tools import Tool
from protonous.trace import Session, Tracer, load_session, load_trace

CONFIG = OpenrouterConfig(model="x")


class ScriptedClient:
    def __init__(self, *replies):
        self.replies = list(replies)

    async def complete(self, window, config, tools=None, *, replay_all_reasoning=True):
        return self.replies.pop(0)


class AddArgs(BaseModel):
    a: int
    b: int


async def _add(args):
    return str(args.a + args.b)


ADD = Tool(name="add", description="Add two ints.", args=AddArgs, fn=_add)


def _setup():
    system = Message(role=Role.SYSTEM, content="sys")
    history = SlidingWindowLayer()
    history.add(Message(role=Role.USER, content="go"))
    window = ContextWindow(layers=[StaticLayer([system]), history])
    return window, history


def _call_reply():
    return Message(
        role=Role.ASSISTANT,
        content=None,
        reasoning="adding",
        reasoning_details=(ReasoningDetail(type="reasoning.text", index=0, text="adding"),),
        tool_calls=(
            ToolCall(id="c1", function_name="add", function_arguments='{"a": 2, "b": 3}'),
            ToolCall(id="c2", function_name="nope", function_arguments="{}"),
        ),
    )


def _done(text="5"):
    return Message(role=Role.ASSISTANT, content=text)


async def test_session_run_round_trip(tmp_path):
    window, history = _setup()
    session = Session.create(
        tmp_path / "s", {"config": CONFIG, "window": window.view(), "tools": [ADD.spec]}
    )
    tracer = session.tracer(meta={"condition": "a", "seed": 3})
    result = await run(
        ScriptedClient(_call_reply(), _done()), window, CONFIG, [ADD],
        update=append_to(history), max_steps=5, tracer=tracer,
    )

    lines = [json.loads(line) for line in (tmp_path / "s" / "run-0001.jsonl").read_text().splitlines()]
    assert [line["type"] for line in lines] == ["run", "step", "step", "end"]
    assert lines[-1] == {"type": "end", "stop_reason": "done", "steps": 2}

    loaded = load_session(tmp_path / "s")
    assert loaded.initial["config"]["model"] == "x"
    assert loaded.initial["window"][0]["content"] == "sys"
    trace = loaded.runs["run-0001"]
    assert trace.meta == {"condition": "a", "seed": 3}
    assert trace.steps == list(result.steps)  # lossless: reasoning, tool calls, error extras
    assert trace.steps[0].tool_results[1].extras == {"error": True}
    assert trace.stop_reason == "done"


def test_create_refuses_existing_directory(tmp_path):
    Session.create(tmp_path / "s", {})
    with pytest.raises(FileExistsError):
        Session.create(tmp_path / "s", {})


def test_open_requires_a_session(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        Session.open(tmp_path / "empty")


def test_open_continues_numbering(tmp_path):
    Session.create(tmp_path / "s", {}).tracer()
    session = Session.open(tmp_path / "s")
    assert session.tracer().name == "run-0002"
    assert session.tracer().name == "run-0003"


def test_named_runs_and_name_validation(tmp_path):
    session = Session.create(tmp_path / "s", {})
    assert session.tracer("trial-a_model.s3").name == "trial-a_model.s3"
    with pytest.raises(FileExistsError):
        session.tracer("trial-a_model.s3")
    with pytest.raises(ValueError):
        session.tracer("../escape")
    # named runs don't disturb automatic numbering
    assert session.tracer().name == "run-0001"


async def test_parallel_runs_get_distinct_files(tmp_path):
    session = Session.create(tmp_path / "s", {})

    async def _one(i):
        window, history = _setup()
        await run(
            ScriptedClient(_done(str(i))), window, CONFIG,
            update=append_to(history), max_steps=1, tracer=session.tracer(meta={"i": i}),
        )

    await asyncio.gather(*(_one(i) for i in range(5)))
    loaded = load_session(tmp_path / "s")
    assert sorted(loaded.runs) == [f"run-{n:04d}" for n in range(1, 6)]
    assert sorted(t.meta["i"] for t in loaded.runs.values()) == list(range(5))


async def test_max_steps_is_recorded(tmp_path):
    window, history = _setup()
    tracer = Session.create(tmp_path / "s", {}).tracer()
    await run(
        ScriptedClient(_call_reply()), window, CONFIG, [ADD],
        update=append_to(history), max_steps=1, tracer=tracer,
    )
    assert load_trace(tracer.path).stop_reason == "max_steps"


async def test_step_is_recorded_even_if_update_raises(tmp_path):
    window, _ = _setup()
    tracer = Session.create(tmp_path / "s", {}).tracer()

    def _broken(w, step):
        raise RuntimeError("update failed")

    with pytest.raises(RuntimeError):
        await run(ScriptedClient(_done()), window, CONFIG, update=_broken, max_steps=1, tracer=tracer)
    trace = load_trace(tracer.path)
    assert len(trace.steps) == 1 and trace.stop_reason is None  # no end line: unfinished


def test_standalone_tracer_without_session(tmp_path):
    tracer = Tracer(tmp_path / "solo.jsonl", {"note": "manual"})
    step = Step(assistant=_done("hi"))
    tracer.step(step)
    window, _ = _setup()
    tracer.end(RunResult(window=window, steps=(step,), stop_reason="done"))
    trace = load_trace(tmp_path / "solo.jsonl")
    assert trace.name == "solo" and trace.meta == {"note": "manual"} and trace.steps == [step]


def test_load_trace_requires_run_header(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"type": "end", "stop_reason": "done", "steps": 0}) + "\n")
    with pytest.raises(ValueError):
        load_trace(path)


class FailingClient:
    def __init__(self, exc):
        self.exc = exc

    async def complete(self, window, config, tools=None, *, replay_all_reasoning=True):
        raise self.exc


@pytest.mark.parametrize(
    ("exc", "name"),
    [
        (ProviderError("HTTP 400: bad model", status_code=400), "ProviderError"),
        (KeyboardInterrupt(), "KeyboardInterrupt"),
        (asyncio.CancelledError(), "CancelledError"),
    ],
)
async def test_run_failure_is_recorded_and_reraised(tmp_path, exc, name):
    window, history = _setup()
    session = Session.create(tmp_path / "s", {})
    tracer = session.tracer(meta={"m": 1})
    with pytest.raises(type(exc)) as info:
        await run(
            FailingClient(exc), window, CONFIG, update=append_to(history), max_steps=3,
            tracer=tracer,
        )
    assert info.value is exc  # re-raised unchanged

    loaded = load_session(tmp_path / "s")
    trace = loaded.runs["run-0001"]
    assert trace.error is not None and trace.error.error == name
    assert trace.stop_reason is None
    assert list(loaded.runs) == ["run-0001"]  # errors.jsonl is not mistaken for a run
    [record] = loaded.errors
    assert record.run == "run-0001" and record.error == name
    assert "Traceback" in record.traceback


async def test_errors_from_several_runs_collect_in_one_file(tmp_path):
    session = Session.create(tmp_path / "s", {})
    for _ in range(2):
        window, history = _setup()
        with pytest.raises(ProviderError):
            await run(
                FailingClient(ProviderError("down")), window, CONFIG,
                update=append_to(history), max_steps=1, tracer=session.tracer(),
            )
    assert [e.run for e in load_session(tmp_path / "s").errors] == ["run-0001", "run-0002"]


def test_errors_is_a_reserved_run_name(tmp_path):
    session = Session.create(tmp_path / "s", {})
    with pytest.raises(ConfigurationError):
        session.tracer("errors")


def test_session_errors_are_typed(tmp_path):
    Session.create(tmp_path / "s", {})
    with pytest.raises(SessionExistsError):
        Session.create(tmp_path / "s", {})
    (tmp_path / "plain").mkdir()
    with pytest.raises(NotASessionError):
        Session.open(tmp_path / "plain")
    session = Session.open(tmp_path / "s")
    session.tracer("x")
    with pytest.raises(RunExistsError):
        session.tracer("x")
