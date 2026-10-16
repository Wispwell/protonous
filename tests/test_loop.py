import pytest
from pydantic import BaseModel

from protonous.client import OpenrouterConfig
from protonous.context import Message, Role, ToolCall
from protonous.errors import ToolDefinitionError, ToolPairingError
from protonous.layers import ContextWindow, SlidingWindowLayer, StaticLayer
from protonous.loop import Step, append_to, check_pairing, run
from protonous.tools import Tool

CONFIG = OpenrouterConfig(model="x")


class ScriptedClient:
    """Returns the scripted replies in order and records what each call saw."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []  # (message contents in the window, tool names offered)
        self.replay_flags = []

    async def complete(self, window, config, tools=None, *, replay_all_reasoning=True):
        self.replay_flags.append(replay_all_reasoning)
        self.seen.append(
            ([m.content for m in window.view()], [t.name for t in tools or []])
        )
        return self.replies.pop(0)


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


class AddArgs(BaseModel):
    a: int
    b: int


async def _add(args):
    return str(args.a + args.b)


ADD = Tool(name="add", description="Add two ints.", args=AddArgs, fn=_add)


def _setup():
    history = SlidingWindowLayer()
    history.add(Message(role=Role.USER, content="go"))
    window = ContextWindow(layers=[StaticLayer([Message(role=Role.SYSTEM, content="sys")]), history])
    return window, history


async def test_answer_without_tools_is_one_step_done():
    window, history = _setup()
    client = ScriptedClient(_say("hi"))
    result = await run(client, window, CONFIG, update=append_to(history), max_steps=5)
    assert result.stop_reason == "done" and len(result.steps) == 1
    assert [m.content for m in history.view()] == ["go", "hi"]
    assert client.seen[0][1] == []  # no tools offered


async def test_tool_call_executes_and_result_is_paired():
    window, history = _setup()
    client = ScriptedClient(_calls(("c1", "add", '{"a": 2, "b": 3}')), _say("5"))
    result = await run(client, window, CONFIG, [ADD], update=append_to(history), max_steps=5)
    assert result.stop_reason == "done" and len(result.steps) == 2
    tool_msg = result.steps[0].tool_results[0]
    assert tool_msg.role is Role.TOOL and tool_msg.content == "5" and tool_msg.tool_call_id == "c1"
    assert client.seen[0][1] == ["add"]
    assert client.seen[1][0] == ["sys", "go", None, "5"]  # the model saw its call + the result


async def test_multiple_calls_run_in_order():
    window, history = _setup()
    order = []

    async def _record(args):
        order.append(args.a)
        return "ok"

    tool = Tool(name="add", description="", args=AddArgs, fn=_record)
    client = ScriptedClient(
        _calls(("c1", "add", '{"a": 1, "b": 0}'), ("c2", "add", '{"a": 2, "b": 0}')), _say("done")
    )
    result = await run(client, window, CONFIG, [tool], update=append_to(history), max_steps=5)
    assert order == [1, 2]
    assert [r.tool_call_id for r in result.steps[0].tool_results] == ["c1", "c2"]


async def test_tool_error_result_goes_back_to_the_model():
    window, history = _setup()
    client = ScriptedClient(_calls(("c1", "nope", "{}")), _say("sorry"))
    result = await run(client, window, CONFIG, [ADD], update=append_to(history), max_steps=5)
    assert result.stop_reason == "done"
    assert client.seen[1][0][-1].startswith("Error: unknown tool")


async def test_max_steps_stops_the_loop():
    window, history = _setup()
    client = ScriptedClient(*[_calls((f"c{i}", "add", '{"a": 1, "b": 1}')) for i in range(3)])
    result = await run(client, window, CONFIG, [ADD], update=append_to(history), max_steps=3)
    assert result.stop_reason == "max_steps" and len(result.steps) == 3


async def test_update_controls_the_next_window():
    window, _history = _setup()
    replacement = ContextWindow(layers=[StaticLayer([Message(role=Role.USER, content="fresh")])])
    steps_seen = []

    def _update(w, step):
        steps_seen.append(step)
        return replacement

    client = ScriptedClient(_calls(("c1", "add", '{"a": 1, "b": 1}')), _say("ok"))
    result = await run(client, window, CONFIG, [ADD], update=_update, max_steps=5)
    assert client.seen[1][0] == ["fresh"]  # step 2 used what update returned
    assert result.window is replacement
    assert all(isinstance(s, Step) for s in steps_seen)


async def test_duplicate_tool_names_rejected():
    window, history = _setup()
    with pytest.raises(ToolDefinitionError):
        await run(
            ScriptedClient(), window, CONFIG, [ADD, ADD], update=append_to(history), max_steps=1
        )


async def test_unpaired_window_raises_before_calling_the_model():
    window, history = _setup()
    history.add(_calls(("c1", "add", "{}")))  # a call with no result
    client = ScriptedClient(_say("never"))
    with pytest.raises(ToolPairingError):
        await run(client, window, CONFIG, [ADD], update=append_to(history), max_steps=1)
    assert client.seen == []


def test_append_to_rejects_foreign_layer():
    window, _ = _setup()
    update = append_to(SlidingWindowLayer())
    with pytest.raises(ValueError):
        update(window, Step(assistant=_say("x"), tool_results=()))


def _result(call_id):
    return Message(role=Role.TOOL, content="r", tool_call_id=call_id)


def test_check_pairing_accepts_well_formed_turns():
    check_pairing(
        [
            Message(role=Role.USER, content="go"),
            _calls(("c1", "add", "{}"), ("c2", "add", "{}")),
            _result("c2"),
            _result("c1"),
            _say("done"),
        ]
    )


@pytest.mark.parametrize(
    "messages",
    [
        [_result("c1")],  # orphan result
        [_calls(("c1", "add", "{}"))],  # call never answered
        [_calls(("c1", "add", "{}")), _say("x"), _result("c1")],  # answered too late
        [_calls(("c1", "add", "{}")), _result("c1"), _result("c1")],  # answered twice
    ],
)
def test_check_pairing_rejects(messages):
    with pytest.raises(ToolPairingError):
        check_pairing(messages)


async def test_replay_all_reasoning_is_passed_to_the_client():
    window, history = _setup()
    client = ScriptedClient(_say("a"))
    await run(client, window, CONFIG, update=append_to(history), max_steps=1)
    window, history = _setup()
    client2 = ScriptedClient(_say("a"))
    await run(
        client2, window, CONFIG, update=append_to(history), max_steps=1,
        replay_all_reasoning=False,
    )
    assert client.replay_flags == [True] and client2.replay_flags == [False]


async def test_run_accepts_a_toolbox():
    from protonous.tools import Toolbox

    box = Toolbox()

    @box.tool
    async def add(a: int, b: int) -> str:
        """Add."""
        return str(a + b)

    window, history = _setup()
    client = ScriptedClient(_calls(("c1", "add", '{"a": 2, "b": 3}')), _say("5"))
    result = await run(client, window, CONFIG, box, update=append_to(history), max_steps=5)
    assert client.seen[0][1] == ["add"]
    assert result.steps[0].tool_results[0].content == "5"
