# Tools

A `Tool` is a [`ToolSpec`](context.md#messages) plus the code behind it.

```python
class ReadFile(BaseModel):
    path: str = Field(description="File to read.")

async def read_file(args: ReadFile) -> str:
    return Path(args.path).read_text()

tool = Tool(name="read_file", description="Read a text file.", args=ReadFile, fn=read_file)
tool.spec   # ToolSpec; parameters = ReadFile.model_json_schema()
```

- `args` is a Pydantic model. Its JSON Schema is what the model sees, and the model's
  arguments are validated against it before `fn` runs.
- `fn` is async, receives the validated `args` instance, and returns a `str`.
- `Tool` is a frozen Pydantic model.

## Execution

`execute(registry, call)` runs one `ToolCall` and returns its tool-result `Message`.
`by_name(tools)` builds the registry and rejects duplicate names. The loop does both;
you rarely call them directly.

What the model gets back:

| Situation | Tool result content |
|---|---|
| success | whatever `fn` returned |
| unknown tool name | `Error: unknown tool 'x' (available: a, b)` |
| arguments are not JSON | `Error: arguments are not valid JSON: ...` |
| arguments fail validation | `Error: invalid arguments: ...` |
| `fn` raises | `Error: <ExceptionType>: <message>` |

Error results carry `extras={"error": True}`. These are model mistakes or world
failures, and the agent is meant to see them and react, so they are returned, not
raised.

Programmer mistakes raise `ToolDefinitionError` instead: duplicate tool names, or a
`fn` that returns something other than a `str`.
