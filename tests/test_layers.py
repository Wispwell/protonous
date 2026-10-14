import pytest

from protonous.context import Message, Role, ToolCall
from protonous.layers import (
    ContextWindow,
    LayerError,
    SingleMessageLayer,
    SlidingWindowLayer,
    StaticLayer,
)


def _user(text):
    return Message(role=Role.USER, content=text)


def _call(call_id):
    return Message(
        role=Role.ASSISTANT,
        content=None,
        tool_calls=(ToolCall(id=call_id, function_name="f", function_arguments="{}"),),
    )


def _result(call_id):
    return Message(role=Role.TOOL, content="r", tool_call_id=call_id)


def _contents(layer):
    return [m.content if m.role is not Role.TOOL else f"tool:{m.tool_call_id}" for m in layer.view()]


def test_static_layer_is_fixed():
    layer = StaticLayer([_user("a"), _user("b")])
    assert _contents(layer) == ["a", "b"]
    with pytest.raises(LayerError):
        layer.add(_user("c"))


def test_single_message_layer_keeps_latest():
    layer = SingleMessageLayer()
    assert layer.view() == []
    layer.add(_user("a"))
    layer.add(_user("b"))
    assert _contents(layer) == ["b"]


def test_single_message_layer_rejects_tool_traffic():
    layer = SingleMessageLayer()
    with pytest.raises(LayerError):
        layer.add(_call("c1"))
    with pytest.raises(LayerError):
        layer.add(_result("c1"))


def test_sliding_window_unbounded_keeps_everything():
    layer = SlidingWindowLayer()
    for i in range(50):
        layer.add(_user(str(i)))
    assert len(layer.view()) == 50


def test_sliding_window_evicts_oldest_plain_messages():
    layer = SlidingWindowLayer(max_messages=2)
    for text in ("a", "b", "c"):
        layer.add(_user(text))
    assert _contents(layer) == ["b", "c"]


def test_sliding_window_evicts_whole_turns():
    layer = SlidingWindowLayer(max_messages=3)
    for msg in (_call("c1"), _result("c1"), _result("c1b"), _user("x")):
        layer.add(msg)
    # 4 > 3: the first turn is the call plus both of its tool results, evicted together
    assert _contents(layer) == ["x"]


def test_sliding_window_never_leaves_an_orphan_tool_result():
    layer = SlidingWindowLayer(max_messages=2)
    for msg in (_user("a"), _call("c1"), _result("c1")):
        layer.add(msg)
    assert layer.view()[0].role is not Role.TOOL
    assert _contents(layer) == [None, "tool:c1"]


def test_sliding_window_keeps_newest_turn_even_if_oversized():
    layer = SlidingWindowLayer(max_messages=1)
    for msg in (_call("c1"), _result("c1")):
        layer.add(msg)
    assert len(layer.view()) == 2


def test_sliding_window_rejects_bad_cap():
    with pytest.raises(ValueError):
        SlidingWindowLayer(max_messages=0)


def test_context_window_concatenates_layers_in_order():
    state = SingleMessageLayer()
    state.add(_user("state"))
    history = SlidingWindowLayer()
    history.add(_user("h1"))
    window = ContextWindow(layers=[StaticLayer([_user("sys")]), state, history])
    assert [m.content for m in window.view()] == ["sys", "state", "h1"]
