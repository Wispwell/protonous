import pytest
from pydantic import ValidationError

from protonous.context import Message, MessageChunk, ReasoningDetail, Role, ToolCall, Usage


def test_message_defaults_and_frozen():
    m = Message(role=Role.USER, content="hi")
    assert m.id and m.timestamp.tzinfo is not None
    assert m.tool_calls == () and m.extras == {}
    with pytest.raises(ValidationError):
        m.content = "x"


def test_toolcall_parsed_arguments():
    tc = ToolCall(id="c", function_name="f", function_arguments='{"a": 1}')
    assert tc.parsed_arguments == {"a": 1}


def test_chunk_add_content_and_scalars_last_non_null():
    a = MessageChunk(role=Role.ASSISTANT, content="He")
    b = MessageChunk(content="llo", stop_reason="stop", usage=Usage(completion_tokens=2))
    f = a + b
    assert f.content == "Hello"
    assert f.role == Role.ASSISTANT  # b.role is None, so a's role survives
    assert f.stop_reason == "stop"
    assert f.usage.completion_tokens == 2


def test_chunk_add_tool_calls_merge_by_index():
    a = MessageChunk(
        tool_calls=(ToolCall(index=0, id="c1", function_name="f", function_arguments='{"a":'),)
    )
    b = MessageChunk(tool_calls=(ToolCall(index=0, function_arguments="1}"),))
    tc = (a + b).tool_calls[0]
    assert tc.id == "c1" and tc.function_name == "f" and tc.function_arguments == '{"a":1}'


def test_chunk_add_reasoning_details_merge_by_index():
    a = MessageChunk(reasoning_details=(ReasoningDetail(type="reasoning.text", index=0, text="I "),))
    b = MessageChunk(reasoning_details=(ReasoningDetail(type="reasoning.text", index=0, text="think"),))
    assert (a + b).reasoning_details[0].text == "I think"


def test_chunk_convert_keeps_usage_and_tuples():
    usage = Usage(prompt_tokens=10, completion_tokens=3, total_tokens=13, cost=0.002)
    m = MessageChunk(role=Role.ASSISTANT, content="hi", usage=usage).convert()
    assert m.role == Role.ASSISTANT and m.content == "hi" and m.usage == usage
    assert isinstance(m.tool_calls, tuple)


def test_usage_adds_field_by_field_and_none_counts_as_nothing():
    a = Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120, cost=0.01)
    b = Usage(prompt_tokens=150, completion_tokens=30, total_tokens=180, cached_tokens=90)
    total = a + b
    assert (total.prompt_tokens, total.completion_tokens, total.total_tokens) == (250, 50, 300)
    assert total.cached_tokens == 90 and total.cost == 0.01
    assert total.reasoning_tokens is None  # nobody reported it
    assert sum([a, b], Usage()) == total
    assert Usage() + Usage() == Usage()


# -- Step, check_pairing ---------------------------------------------------------------


from protonous.context import Step, check_pairing
from protonous.errors import ToolPairingError


def _call(*ids):
    return Message(
        role=Role.ASSISTANT, content=None,
        tool_calls=tuple(ToolCall(id=i, function_name="f", function_arguments="{}") for i in ids),
    )


def _result(call_id):
    return Message(role=Role.TOOL, content="r", tool_call_id=call_id)


def test_step_messages_and_usage():
    reply = _call("c1").model_copy(update={"usage": Usage(total_tokens=5)})
    step = Step(assistant=reply, tool_results=(_result("c1"),))
    assert step.messages == (reply, step.tool_results[0])
    assert step.usage.total_tokens == 5
    assert Step(assistant=Message(role=Role.ASSISTANT, content="x")).usage == Usage()


def test_check_pairing_accepts_well_formed_turns():
    check_pairing(
        [Message(role=Role.USER, content="go"), _call("c1", "c2"), _result("c2"), _result("c1"),
         Message(role=Role.ASSISTANT, content="done")]
    )


@pytest.mark.parametrize(
    "messages",
    [
        [_result("c1")],  # orphan result
        [_call("c1")],  # call never answered
        [_call("c1"), Message(role=Role.ASSISTANT, content="x"), _result("c1")],  # too late
        [_call("c1"), _result("c1"), _result("c1")],  # answered twice
    ],
)
def test_check_pairing_rejects(messages):
    with pytest.raises(ToolPairingError):
        check_pairing(messages)


def test_reasoning_merge_keeps_unmodelled_fields():
    a = MessageChunk(reasoning_details=(
        ReasoningDetail(type="reasoning.text", index=0, text="I ", trace_id="t1"),))
    b = MessageChunk(reasoning_details=(
        ReasoningDetail(type="reasoning.text", index=0, text="think", signature="s"),))
    merged = (a + b).reasoning_details[0]
    assert merged.text == "I think" and merged.signature == "s"
    assert merged.model_extra == {"trace_id": "t1"}
