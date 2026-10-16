# Tools

## Defining tools

An agent's tools live in a `Toolbox`. Decorating a function defines the tool and
registers it:

```python
from typing import Annotated
from pydantic import Field
from protonous.tools import Toolbox

tools = Toolbox()

@tools.tool
async def read_file(
    path: Annotated[str, Field(description="File to read.")],
    max_bytes: int = 10_000,
) -> str:
    """Read a text file."""
    return Path(path).read_text()[:max_bytes]

step = await model.step(window, tools)
```

The decorator turns an async function into a `Tool`:

- **name**: the function's name. **description**: its docstring. Both can be
  overridden: `@tools.tool(name="read", description="...")`.
- **arguments**: built from the signature. Every parameter needs a type annotation;
  a default makes it optional; `Annotated[T, Field(description=...)]` documents it
  for the model. Any type Pydantic understands works, nested models included.
- The model's arguments are validated against that signature before the function
  runs, and unknown arguments are rejected. The function is then called with
  keyword arguments and must return a `str`.

A definition that can't work fails immediately, at decoration time, with
`ToolDefinitionError`: a non-async function, no docstring (and no `description=`),
an unannotated parameter, or `*args` / `**kwargs` / positional-only parameters.

## Toolboxes

A `Toolbox` is a sequence of tools in registration order, so it goes anywhere a
list of tools does. Names are unique within a toolbox; registering a second tool
with the same name raises `ToolDefinitionError`.

One toolbox per agent keeps each agent's tool set explicit, and makes tool sets easy
to vary: `toolbox.add(tool)` registers an existing tool, and the same tool can sit in
several toolboxes.

```python
base = Toolbox()

@base.tool
async def read_file(path: str) -> str:
    """Read a text file."""
    ...

extended = Toolbox(base)      # starts with base's tools

@extended.tool
async def write_file(path: str, text: str) -> str:
    """Write a text file."""
    ...
```

There is deliberately no global registry: tools never leak between agents, test
files or imported modules.

`@tool` (from `protonous.tools`) defines a tool the same way without registering it
anywhere.

## The explicit form

The decorators are shorthand for constructing a `Tool`. To reuse an existing
Pydantic model as the arguments, build one directly:

```python
class ReadFile(BaseModel):
    path: str

async def read_file(args: ReadFile) -> str:
    return Path(args.path).read_text()

Tool(name="read_file", description="Read a text file.", args=ReadFile, fn=read_file)
```

Here `fn` receives the validated model instance. `tool.spec` is the
[`ToolSpec`](context.md#messages) the model sees; `Tool` is a frozen Pydantic model.

## Execution

`execute(registry, call)` runs one `ToolCall` and returns its tool-result `Message`.
`by_name(tools)` builds the registry and rejects duplicate names, and
`run_tools(tools, reply)` runs every tool call of a reply in order. `model.step`
uses them; you rarely call them directly.

What the model gets back:

| Situation | Tool result content |
|---|---|
| success | whatever the function returned |
| unknown tool name | `Error: unknown tool 'x' (available: a, b)` |
| arguments are not JSON | `Error: arguments are not valid JSON: ...` |
| arguments fail validation (wrong type, missing, unknown) | `Error: invalid arguments: ...` |
| the function raises | `Error: <ExceptionType>: <message>` |

Error results carry `extras={"error": True}`. These are model mistakes or world
failures, and the agent is meant to see them and react, so they are returned, not
raised.

Programmer mistakes raise `ToolDefinitionError` instead: duplicate tool names, or a
function that returns something other than a `str`.
