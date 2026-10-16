import pytest
from fakes import FailingClient, ScriptedClient, agent, model_for

from protonous.context import Message, Role, ToolCall
from protonous.errors import (
    ProviderError,
    StreamIncompleteError,
    ToolDefinitionError,
    ToolPairingError,
)
from protonous.layers import ContextWindow, SlidingWindowLayer, StaticLayer
from protonous.tools import Toolbox, tool


def _say(text):
    return Message(role=Role.ASSISTANT, content=text)


def _calls(*calls):
    """calls: (id, name, raw JSON arguments) triples."""
    return Message(
        role=Role.ASSISTANT,
        content=None,
        tool_calls=tuple(
            ToolCall(id=cid, function_name=name, function_arguments=args)
            for cid, name, args in calls
        ),
    )


@tool
async def add(a: int, b: int) -> str:
    """Add two integers."""
    return str(a + b)


def _setup():
    history = SlidingWindowLayer()
    history.add(Message(role=Role.USER, content="go"))
    window = ContextWindow(layers=[StaticLayer([Message(role=Role.SYSTEM, content="sys")]), history])
    return window, history


# -- Model.call ----------------------------------------------------------------------


async def test_call_returns_the_assembled_reply_and_leaves_the_window_alone():
    window, _ = _setup()
    client = ScriptedClient(Message(role=Role.ASSISTANT, content="hello", reasoning="hm"))
    reply = await model_for(client).call(window)
    assert reply.content == "hello" and reply.reasoning == "hm"
    assert client.seen == [(["sys", "go"], [])]  # no tools offered
    assert [m.content for m in window.view()] == ["sys", "go"]  # nothing added


async def test_call_offers_tools_but_does_not_run_them():
    window, _ = _setup()
    client = ScriptedClient(_calls(("c1", "add", '{"a": 1, "b": 2}')))
    reply = await model_for(client).call(window, [add])
    assert client.seen[0][1] == ["add"]
    assert reply.tool_calls[0].function_arguments == '{"a": 1, "b": 2}'


async def test_unpaired_window_raises_before_calling_the_model():
    window, history = _setup()
    history.add(_calls(("c1", "add", "{}")))  # a call with no result
    client = ScriptedClient(_say("never"))
    with pytest.raises(ToolPairingError):
        await model_for(client).call(window)
    assert client.seen == []


# -- Model.stream ----------------------------------------------------------------------


async def test_stream_yields_chunks_then_has_the_reply():
    window, _ = _setup()
    stream = model_for(ScriptedClient(_say("hello there"))).stream(window)
    chunks = [chunk async for chunk in stream]
    assert "".join(c.content or "" for c in chunks) == "hello there"
    assert chunks[-1].stop_reason == "stop"
    assert stream.reply.content == "hello there"
    assert [c async for c in stream] == []  # read once; iterating again yields nothing


async def test_stream_step_runs_the_tools_of_the_streamed_reply():
    window, _ = _setup()
    stream = model_for(ScriptedClient(_calls(("c1", "add", '{"a": 2, "b": 3}')))).stream(
        window, [add]
    )
    fragments = [c.tool_calls[0].function_arguments async for c in stream if c.tool_calls]
    assert "".join(fragments) == '{"a": 2, "b": 3}'
    step = await stream.step()
    assert step.tool_results[0].content == "5"


async def test_collect_and_step_read_whatever_is_left():
    window, _ = _setup()
    stream = model_for(ScriptedClient(_say("hello there"))).stream(window)
    await anext(stream)  # read one chunk only
    assert (await stream.collect()).content == "hello there"


async def test_reply_before_the_end_raises():
    window, _ = _setup()
    stream = model_for(ScriptedClient(_say("hello"))).stream(window)
    with pytest.raises(StreamIncompleteError):
        _ = stream.reply


async def test_closing_early_closes_the_connection_and_leaves_no_reply():
    window, _ = _setup()
    client = ScriptedClient(_say("a long answer the user stopped"))
    async with model_for(client).stream(window) as stream:
        async for _chunk in stream:
            break  # e.g. the user pressed stop
    assert client.closed == 1
    with pytest.raises(StreamIncompleteError, match="closed"):
        _ = stream.reply
    with pytest.raises(StreamIncompleteError):
        await stream.step()


async def test_async_with_after_a_full_read_keeps_the_reply():
    window, _ = _setup()
    client = ScriptedClient(_say("hi"))
    async with model_for(client).stream(window) as stream:
        await stream.collect()
    assert stream.reply.content == "hi" and client.closed == 0


async def test_a_failed_stream_has_no_reply():
    window, _ = _setup()
    stream = model_for(FailingClient(ProviderError("down"))).stream(window)
    with pytest.raises(ProviderError):
        await stream.collect()
    with pytest.raises(StreamIncompleteError, match="failed"):
        await stream.collect()  # a later read doesn't turn a failure into a partial reply


async def test_unpaired_window_raises_when_the_stream_is_created():
    window, history = _setup()
    history.add(_calls(("c1", "add", "{}")))
    client = ScriptedClient(_say("never"))
    with pytest.raises(ToolPairingError):
        model_for(client).stream(window)
    assert client.seen == []


# -- Model.step ----------------------------------------------------------------------


async def test_step_runs_tool_calls_in_order_and_pairs_results():
    window, _ = _setup()
    client = ScriptedClient(
        _calls(("c1", "add", '{"a": 20, "b": 22}'), ("c2", "add", '{"a": 1, "b": 1}'))
    )
    step = await model_for(client).step(window, [add])
    assert [(r.tool_call_id, r.content) for r in step.tool_results] == [("c1", "42"), ("c2", "2")]
    assert step.messages == (step.assistant, *step.tool_results)


async def test_step_without_tool_calls_has_no_results():
    window, _ = _setup()
    step = await model_for(ScriptedClient(_say("done"))).step(window, [add])
    assert step.tool_results == () and step.messages == (step.assistant,)


async def test_tool_errors_come_back_as_results():
    window, _ = _setup()
    client = ScriptedClient(_calls(("c1", "nope", "{}")))
    step = await model_for(client).step(window, [add])
    assert step.tool_results[0].content.startswith("Error: unknown tool")
    assert step.tool_results[0].extras == {"error": True}


async def test_step_accepts_a_toolbox_and_rejects_duplicate_names():
    box = Toolbox()

    @box.tool
    async def double(n: int) -> str:
        """Double a number."""
        return str(2 * n)

    window, _ = _setup()
    step = await model_for(ScriptedClient(_calls(("c1", "double", '{"n": 21}')))).step(window, box)
    assert step.tool_results[0].content == "42"
    with pytest.raises(ToolDefinitionError):
        await model_for(ScriptedClient(_say("x"))).step(window, [add, add])


# -- a loop built from the blocks ----------------------------------------------------


async def test_reference_loop_feeds_results_back_and_stops_when_answered():
    window, history = _setup()
    client = ScriptedClient(_calls(("c1", "add", '{"a": 2, "b": 3}')), _say("5"))
    steps, stop = await agent(model_for(client), window, history, [add])
    assert stop == "done" and len(steps) == 2
    assert client.seen[1][0] == ["sys", "go", None, "5"]  # the model saw its call + the result


async def test_reference_loop_stops_at_max_steps():
    window, history = _setup()
    client = ScriptedClient(*[_calls((f"c{i}", "add", '{"a": 1, "b": 1}')) for i in range(3)])
    steps, stop = await agent(model_for(client), window, history, [add], max_steps=3)
    assert stop == "max_steps" and len(steps) == 3
