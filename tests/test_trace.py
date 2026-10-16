import asyncio
import json

import pytest
from fakes import FailingClient, ScriptedClient, agent, model_for
from pydantic import BaseModel

from protonous.client import OpenrouterConfig
from protonous.context import Message, ReasoningDetail, Role, Step, ToolCall
from protonous.errors import (
    ConfigurationError,
    NotASessionError,
    ProviderError,
    RunExistsError,
    SessionExistsError,
)
from protonous.layers import ContextWindow, SlidingWindowLayer, StaticLayer
from protonous.tools import Tool
from protonous.trace import Session, Tracer, load_session, load_trace

CONFIG = OpenrouterConfig(model="x")


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
    steps, _ = await agent(
        model_for(ScriptedClient(_call_reply(), _done())), window, history, [ADD], tracer=tracer
    )

    lines = [json.loads(line) for line in (tmp_path / "s" / "run-0001.jsonl").read_text().splitlines()]
    assert [line["type"] for line in lines] == ["run", "step", "step", "end"]
    assert lines[-1] == {"type": "end", "stop_reason": "done", "steps": 2}

    loaded = load_session(tmp_path / "s")
    assert loaded.initial["config"]["model"] == "x"
    assert loaded.initial["window"][0]["content"] == "sys"
    trace = loaded.runs["run-0001"]
    assert trace.meta == {"condition": "a", "seed": 3}
    assert trace.steps == steps  # lossless: reasoning, tool calls, error extras
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
        await agent(
            model_for(ScriptedClient(_done(str(i)))), window, history,
            tracer=session.tracer(meta={"i": i}),
        )

    await asyncio.gather(*(_one(i) for i in range(5)))
    loaded = load_session(tmp_path / "s")
    assert sorted(loaded.runs) == [f"run-{n:04d}" for n in range(1, 6)]
    assert sorted(t.meta["i"] for t in loaded.runs.values()) == list(range(5))


async def test_max_steps_is_recorded(tmp_path):
    window, history = _setup()
    tracer = Session.create(tmp_path / "s", {}).tracer()
    await agent(
        model_for(ScriptedClient(_call_reply())), window, history, [ADD], max_steps=1,
        tracer=tracer,
    )
    assert load_trace(tracer.path).stop_reason == "max_steps"


async def test_recorded_steps_survive_a_later_failure(tmp_path):
    window, _ = _setup()
    tracer = Session.create(tmp_path / "s", {}).tracer()
    model = model_for(ScriptedClient(_done()))
    with pytest.raises(RuntimeError), tracer.recording():
        tracer.step(await model.step(window))
        raise RuntimeError("the caller's own code failed")
    trace = load_trace(tracer.path)
    assert len(trace.steps) == 1 and trace.stop_reason is None
    assert trace.error is not None and trace.error.message == "the caller's own code failed"


def test_standalone_tracer_without_session(tmp_path):
    tracer = Tracer(tmp_path / "solo.jsonl", {"note": "manual"})
    step = Step(assistant=_done("hi"))
    tracer.step(step)
    tracer.end("done")
    trace = load_trace(tmp_path / "solo.jsonl")
    assert trace.name == "solo" and trace.meta == {"note": "manual"} and trace.steps == [step]


def test_load_trace_requires_run_header(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"type": "end", "stop_reason": "done", "steps": 0}) + "\n")
    with pytest.raises(ValueError):
        load_trace(path)


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
        await agent(model_for(FailingClient(exc)), window, history, tracer=tracer)
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
            await agent(
                model_for(FailingClient(ProviderError("down"))), window, history,
                tracer=session.tracer(),
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


async def test_usage_is_traced_per_step_and_summed(tmp_path):
    from protonous.context import Usage

    first = Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120, cost=0.001)
    second = Usage(prompt_tokens=180, completion_tokens=10, total_tokens=190, cost=0.002)
    call = _call_reply().model_copy(update={"usage": first})
    done = _done().model_copy(update={"usage": second})

    window, history = _setup()
    session = Session.create(tmp_path / "s", {})
    steps, _ = await agent(
        model_for(ScriptedClient(call, done)), window, history, [ADD], tracer=session.tracer()
    )
    assert [s.usage for s in steps] == [first, second]

    loaded = load_session(tmp_path / "s")
    trace = loaded.runs["run-0001"]
    assert [s.usage for s in trace.steps] == [first, second]  # survives the JSONL round-trip
    assert trace.usage.total_tokens == 310 and trace.usage.cost == pytest.approx(0.003)
    assert loaded.usage == trace.usage


def test_unmodelled_reasoning_fields_survive_a_trace(tmp_path):
    reply = Message(
        role=Role.ASSISTANT, content="ok",
        reasoning_details=(ReasoningDetail(type="reasoning.text", text="hm", provider_hint=7),),
    )
    tracer = Tracer(tmp_path / "r.jsonl")
    tracer.step(Step(assistant=reply))
    loaded = load_trace(tmp_path / "r.jsonl").steps[0].assistant.reasoning_details[0]
    assert loaded.model_extra == {"provider_hint": 7}
