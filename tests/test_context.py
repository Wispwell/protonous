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


def test_chunk_convert_sets_token_count_and_tuples():
    m = MessageChunk(role=Role.ASSISTANT, content="hi", usage=Usage(completion_tokens=3)).convert()
    assert m.role == Role.ASSISTANT and m.content == "hi" and m.token_count == 3
    assert isinstance(m.tool_calls, tuple)
