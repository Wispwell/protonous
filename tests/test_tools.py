import pytest
from pydantic import BaseModel, ValidationError

from protonous.context import Role, ToolCall
from protonous.errors import ToolDefinitionError
from protonous.tools import Tool, by_name, execute


class AddArgs(BaseModel):
    a: int
    b: int


async def _add(args):
    return str(args.a + args.b)


ADD = Tool(name="add", description="Add two ints.", args=AddArgs, fn=_add)


def _call(name, args, call_id="c1"):
    return ToolCall(id=call_id, function_name=name, function_arguments=args)


def test_spec_comes_from_args_schema():
    spec = ADD.spec
    assert spec.name == "add" and spec.description == "Add two ints."
    assert set(spec.parameters["properties"]) == {"a", "b"}
    assert spec.parameters["required"] == ["a", "b"]


def test_tool_is_frozen():
    with pytest.raises(ValidationError):
        ADD.name = "other"


def test_by_name_rejects_duplicates():
    assert by_name([ADD]) == {"add": ADD}
    with pytest.raises(ToolDefinitionError):
        by_name([ADD, ADD])


async def test_execute_success():
    msg = await execute(by_name([ADD]), _call("add", '{"a": 2, "b": 3}'))
    assert msg.role is Role.TOOL and msg.content == "5" and msg.tool_call_id == "c1"
    assert msg.extras == {}


async def test_execute_empty_arguments_mean_empty_object():
    class NoArgs(BaseModel):
        pass

    async def _ping(args):
        return "pong"

    tool = Tool(name="ping", description="", args=NoArgs, fn=_ping)
    assert (await execute(by_name([tool]), _call("ping", ""))).content == "pong"


@pytest.mark.parametrize(
    ("name", "args", "expected"),
    [
        ("nope", "{}", "unknown tool 'nope' (available: add)"),
        ("add", "{not json", "arguments are not valid JSON"),
        ("add", '{"a": "x", "b": 1}', "invalid arguments"),
    ],
)
async def test_model_mistakes_become_error_results(name, args, expected):
    msg = await execute(by_name([ADD]), _call(name, args))
    assert msg.content.startswith("Error: ") and expected in msg.content
    assert msg.extras == {"error": True} and msg.tool_call_id == "c1"


async def test_tool_exception_becomes_error_result():
    async def _boom(args):
        raise RuntimeError("disk full")

    tool = Tool(name="add", description="", args=AddArgs, fn=_boom)
    msg = await execute(by_name([tool]), _call("add", '{"a": 1, "b": 1}'))
    assert msg.content == "Error: RuntimeError: disk full"


async def test_tool_returning_non_string_raises():
    async def _bad(args):
        return 5

    tool = Tool(name="add", description="", args=AddArgs, fn=_bad)
    with pytest.raises(ToolDefinitionError):
        await execute(by_name([tool]), _call("add", '{"a": 1, "b": 1}'))
