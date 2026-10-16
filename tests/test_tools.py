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


# -- @tool -------------------------------------------------------------------------

from typing import Annotated

from pydantic import Field

from protonous.tools import tool


@tool
async def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)


@tool
async def greet(name: Annotated[str, Field(description="Who to greet.")], loud: bool = False) -> str:
    """Greet someone."""
    text = f"hello {name}"
    return text.upper() if loud else text


class Point(BaseModel):
    x: int
    y: int


@tool
async def norm1(p: Point) -> str:
    """Manhattan length of a point."""
    assert isinstance(p, Point)  # nested models arrive as models, not dicts
    return str(abs(p.x) + abs(p.y))


@tool(name="ping", description="Check the connection.")
async def _ping() -> str:
    return "pong"


def test_decorator_reads_name_docstring_and_signature():
    assert isinstance(add, Tool)
    spec = add.spec
    assert spec.name == "add" and spec.description == "Add two numbers."
    assert spec.parameters["properties"] == {
        "a": {"title": "A", "type": "integer"},
        "b": {"title": "B", "type": "integer"},
    }
    assert spec.parameters["required"] == ["a", "b"]
    assert spec.parameters["additionalProperties"] is False


def test_decorator_defaults_and_field_descriptions():
    params = greet.spec.parameters
    assert params["required"] == ["name"]
    assert params["properties"]["name"]["description"] == "Who to greet."
    assert params["properties"]["loud"]["default"] is False


def test_decorator_overrides():
    assert (_ping.name, _ping.description) == ("ping", "Check the connection.")


async def test_decorated_tools_execute_with_keywords():
    registry = by_name([add, greet, norm1, _ping])
    assert (await execute(registry, _call("add", '{"a": 2, "b": 3}'))).content == "5"
    assert (await execute(registry, _call("greet", '{"name": "kim"}'))).content == "hello kim"
    assert (await execute(registry, _call("greet", '{"name": "kim", "loud": true}'))).content == "HELLO KIM"
    assert (await execute(registry, _call("norm1", '{"p": {"x": -2, "y": 3}}'))).content == "5"
    assert (await execute(registry, _call("ping", ""))).content == "pong"


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ('{"a": 1, "b": 2, "c": 3}', "Extra inputs are not permitted"),
        ('{"a": "x", "b": 2}', "invalid arguments"),
        ('{"a": 1}', "Field required"),
    ],
)
async def test_decorated_tool_argument_errors_reach_the_model(args, expected):
    msg = await execute(by_name([add]), _call("add", args))
    assert msg.extras == {"error": True} and expected in msg.content


def _sync(a: int) -> str:
    """Not async."""
    return ""


async def _no_doc(a: int) -> str:
    return ""


async def _unannotated(a) -> str:
    """Missing a type."""
    return ""


async def _varargs(*a: int) -> str:
    """Varargs."""
    return ""


async def _kwargs(**a: int) -> str:
    """Kwargs."""
    return ""


async def _positional(a: int, /) -> str:
    """Positional-only."""
    return ""


@pytest.mark.parametrize(
    ("fn", "expected"),
    [
        (_sync, "must be an async function"),
        (_no_doc, "needs a docstring"),
        (_unannotated, "needs a type annotation"),
        (_varargs, "plain named parameter"),
        (_kwargs, "plain named parameter"),
        (_positional, "plain named parameter"),
    ],
)
def test_decorator_rejects_bad_definitions(fn, expected):
    with pytest.raises(ToolDefinitionError, match=expected):
        tool(fn)


# -- Toolbox -----------------------------------------------------------------------

from protonous.tools import Toolbox


def test_toolbox_collects_decorated_tools_in_order():
    box = Toolbox()

    @box.tool
    async def first(x: int) -> str:
        """First."""
        return str(x)

    @box.tool(name="second_tool", description="Second.")
    async def second() -> str:
        return "2"

    assert isinstance(first, Tool) and isinstance(second, Tool)
    assert [t.name for t in box] == ["first", "second_tool"]
    assert len(box) == 2 and box[0] is first and box[-1] is second
    assert repr(box) == "Toolbox(['first', 'second_tool'])"


def test_toolbox_rejects_duplicate_names():
    box = Toolbox([ADD])
    with pytest.raises(ToolDefinitionError, match="already has a tool named 'add'"):

        @box.tool
        async def add(a: int) -> str:
            """Clash."""
            return ""

    assert len(box) == 1


def test_one_tool_in_several_toolboxes():
    base, extended = Toolbox([ADD]), Toolbox([ADD])

    @extended.tool
    async def extra() -> str:
        """Extra."""
        return ""

    assert [t.name for t in base] == ["add"]
    assert [t.name for t in extended] == ["add", "extra"]


async def test_toolbox_works_as_the_registry():
    box = Toolbox()

    @box.tool
    async def double(n: int) -> str:
        """Double a number."""
        return str(2 * n)

    msg = await execute(by_name(box), _call("double", '{"n": 21}'))
    assert msg.content == "42"


async def test_run_tools_runs_every_call_in_order():
    from protonous.context import Message
    from protonous.tools import run_tools

    reply = Message(
        role=Role.ASSISTANT, content=None,
        tool_calls=(_call("add", '{"a": 1, "b": 2}', "c1"), _call("add", '{"a": 3, "b": 4}', "c2")),
    )
    results = await run_tools([ADD], reply)
    assert [(r.tool_call_id, r.content) for r in results] == [("c1", "3"), ("c2", "7")]
    assert await run_tools([ADD], Message(role=Role.ASSISTANT, content="no tools")) == ()
