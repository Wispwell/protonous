from types import SimpleNamespace

from protonous.context import Message, ReasoningDetail, Role, ToolCall, ToolSpec
from protonous.layers import ContextWindow, SlidingWindowLayer, StaticLayer
from protonous.normalizers import (
    normalize_chunk,
    normalize_in,
    normalize_tools,
)


def test_normalize_in_flatten_and_mapping():
    history = SlidingWindowLayer()
    for msg in (
        Message(role=Role.USER, content="hi"),
        Message(
            role=Role.ASSISTANT,
            content=None,
            tool_calls=(ToolCall(id="c1", function_name="f", function_arguments="{}"),),
        ),
        Message(role=Role.TOOL, content="42", tool_call_id="c1"),
    ):
        history.add(msg)
    ctx = ContextWindow(layers=[StaticLayer([Message(role=Role.SYSTEM, content="sys")]), history])
    assert normalize_in(ctx) == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "content": "42", "tool_call_id": "c1"},
    ]


def _window(*messages):
    return ContextWindow(layers=[StaticLayer(messages)])


def test_normalize_in_round_trips_reasoning_details():
    rd = (
        ReasoningDetail(type="reasoning.text", index=0, text="think"),
        ReasoningDetail(type="reasoning.encrypted", index=1, data="opaque"),
    )
    out = normalize_in(
        _window(Message(role=Role.ASSISTANT, content="ok", reasoning="think", reasoning_details=rd))
    )
    assert out[0]["reasoning_details"] == [
        {"type": "reasoning.text", "index": 0, "text": "think"},
        {"type": "reasoning.encrypted", "index": 1, "data": "opaque"},
    ]
    assert "reasoning" not in out[0]  # details win; the raw string isn't sent twice


def test_normalize_in_falls_back_to_raw_reasoning_string():
    out = normalize_in(_window(Message(role=Role.ASSISTANT, content="ok", reasoning="raw")))
    assert out[0]["reasoning"] == "raw" and "reasoning_details" not in out[0]


def test_normalize_in_never_sends_reasoning_on_non_assistant():
    out = normalize_in(_window(Message(role=Role.USER, content="hi", reasoning="x")))
    assert out == [{"role": "user", "content": "hi"}]


def test_normalize_tools_shape():
    spec = ToolSpec(
        name="add",
        description="Add two ints.",
        parameters={"type": "object", "properties": {"a": {"type": "integer"}}},
    )
    assert normalize_tools([spec]) == [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two ints.",
                "parameters": {"type": "object", "properties": {"a": {"type": "integer"}}},
            },
        }
    ]


def test_normalize_chunk_tool_call_fragment():
    tc = SimpleNamespace(
        id="c1", type="function", index=0,
        function=SimpleNamespace(name="f", arguments='{"a":'),
    )
    delta = SimpleNamespace(
        role=None, content=None, tool_calls=[tc], reasoning=None, reasoning_details=None
    )
    chunk = SimpleNamespace(
        id="gen-5", usage=None, choices=[SimpleNamespace(finish_reason=None, delta=delta)]
    )
    c = normalize_chunk(chunk)
    assert c.tool_calls[0].index == 0 and c.tool_calls[0].function_name == "f"
    assert c.tool_calls[0].function_arguments == '{"a":'


def test_normalize_chunk_reasoning_details_fragment():
    delta = SimpleNamespace(
        role=None, content=None, tool_calls=None, reasoning="th",
        reasoning_details=[{"type": "reasoning.text", "index": 0, "text": "th"}],
    )
    chunk = SimpleNamespace(
        id="gen-6", usage=None, choices=[SimpleNamespace(finish_reason=None, delta=delta)]
    )
    c = normalize_chunk(chunk)
    assert c.reasoning == "th" and c.reasoning_details[0].text == "th"


def test_normalize_chunk_delta():
    delta = SimpleNamespace(
        role="assistant", content="He", tool_calls=None, reasoning=None, reasoning_details=None
    )
    chunk = SimpleNamespace(
        id="gen-3", usage=None, choices=[SimpleNamespace(finish_reason=None, delta=delta)]
    )
    c = normalize_chunk(chunk)
    assert c.role is Role.ASSISTANT and c.content == "He" and c.provider_id == "gen-3"


def test_normalize_chunk_usage_only_tail():
    chunk = SimpleNamespace(
        id="gen-4",
        choices=[],
        usage=SimpleNamespace(
            completion_tokens=2,
            prompt_tokens=1,
            total_tokens=3,
            completion_tokens_details=None,
            prompt_tokens_details=None,
        ),
    )
    c = normalize_chunk(chunk)
    assert c.content is None and c.role is None
    assert c.usage.completion_tokens == 2


def test_normalize_in_replays_only_latest_reasoning_when_asked():
    window = _window(
        Message(role=Role.ASSISTANT, content="1", reasoning="old"),
        Message(role=Role.USER, content="u"),
        Message(role=Role.ASSISTANT, content="2", reasoning="new"),
        Message(role=Role.USER, content="u2"),
    )
    every = normalize_in(window)
    assert every[0]["reasoning"] == "old" and every[2]["reasoning"] == "new"
    latest = normalize_in(window, replay_all_reasoning=False)
    assert "reasoning" not in latest[0] and latest[2]["reasoning"] == "new"
