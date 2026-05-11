# Local Python Agent with Tools — Part 3
> Subject/Topic: Build a local agent in Python with tools Part 3

## Overview

This project builds a local Python AI agent that can:

- Chat with a local OpenAI-compatible LLM endpoint
- Inject dynamic runtime context into every request
- Register Python functions as tools
- Convert Python type hints into JSON schema
- Let the LLM decide when to call tools
- Execute tools locally
- Send tool results back to the LLM
- Repeat this process until the model gives a final answer

The key idea is that a real AI agent is not just one model response. It is a loop:

```text
User message
    ↓
LLM receives message + context + tool schemas
    ↓
LLM either replies normally or requests a tool call
    ↓
Agent executes the tool locally
    ↓
Tool result is added back into the message history
    ↓
LLM sees the result and produces the final answer
```

This is the basic form of an agentic loop.

---

## What This Agent Can Do

The example agent includes three tools:

| Tool | Purpose |
|---|---|
| `add(a, b)` | Adds two numbers |
| `multiply(a, b)` | Multiplies two numbers |
| `secret()` | Returns a secret value |

It also injects dynamic context on every request:

```text
Current date and time
Current user
```

The system prompt tells the model to include the current date/time in every answer.

---

## Project Code

```python
import requests
from dataclasses import dataclass, field
from charset_normalizer.md import annotations
from rich.console import Console
import datetime
import getpass
import inspect
import json
from typing import Callable, Any, Annotated, get_origin, get_args, Union, Final
```

Main classes:

| Class | Responsibility |
|---|---|
| `Tools` | Registers tools, generates schemas, executes tool calls |
| `Agent` | Manages messages, context injection, LLM calls, and the tool loop |

---

## Core Concept: Tool Calling Loop

The agent becomes “agentic” because of this loop inside `Agent.chat()`:

```python
while True:
    # 1. Send messages + tools to model
    # 2. Read model response
    # 3. Check if model requested tool calls
    # 4. If no tool calls, return final response
    # 5. If tool calls exist, execute them
    # 6. Add tool results back to messages
    # 7. Loop again
```

In plain English:

```text
Ask the model what to do.
If the model asks for a tool, run the tool.
Give the tool result back to the model.
Repeat until the model gives a normal final answer.
```

---

## `Tools` Class

The `Tools` class is responsible for:

1. Storing registered Python functions
2. Creating JSON schemas from function signatures
3. Returning schemas to send to the LLM
4. Executing tool calls returned by the LLM

```python
@dataclass
class Tools:
    TOOL_SCHEMA_ATTR: Final[str] = "__tool_schema__"
    tools: dict[str, Callable[..., Any]] = field(default_factory=dict)
```

### Important Fields

| Field | Meaning |
|---|---|
| `TOOL_SCHEMA_ATTR` | Custom attribute name used to store schema on a function |
| `tools` | Dictionary mapping tool names to Python functions |

Example internal structure:

```python
self.tools = {
    "add": add,
    "multiply": multiply,
    "secret": secret,
}
```

---

## `__tool_schema__`

`__tool_schema__` is not a built-in Python feature. It is a custom attribute name created by this project.

The code attaches a generated schema directly onto the function object:

```python
setattr(func, "__tool_schema__", schema)
```

Later, the schema is retrieved with:

```python
getattr(func, "__tool_schema__", None)
```

This allows each registered tool function to carry its own OpenAI-style schema.

---

## `Final[str]`

```python
TOOL_SCHEMA_ATTR: Final[str] = "__tool_schema__"
```

`Final[str]` means this variable is intended to stay as a string constant and should not be reassigned.

It does not enforce the value at runtime, but type checkers can warn you if you try to reassign it.

---

## Annotation to JSON Schema

The key method is:

```python
def _annotation_to_schema(annotation: Any) -> dict[str, Any]:
```

This converts Python type hints into JSON schema fragments that the LLM can understand.

For example:

```python
name: str
age: int
tags: list[str]
```

Becomes:

```json
{
  "name": {"type": "string"},
  "age": {"type": "number"},
  "tags": {
    "type": "array",
    "items": {"type": "string"}
  }
}
```

This is needed because the LLM tool API does not directly understand Python type hints. It understands JSON schema.

---

## Supported Type Conversions

| Python annotation | JSON schema output |
|---|---|
| `str` | `{"type": "string"}` |
| `int` | `{"type": "number"}` |
| `float` | `{"type": "number"}` |
| `bool` | `{"type": "boolean"}` |
| `dict` | `{"type": "object"}` |
| `list` | `{"type": "array"}` |
| `list[str]` | `{"type": "array", "items": {"type": "string"}}` |
| `Union[str, None]` | `{"type": "string"}` |
| `Annotated[int, "First number"]` | `{"type": "number", "description": "First number"}` |

---

## Why Both `annotation` and `origin` Are Needed

```python
origin = get_origin(annotation)
```

Some annotations are simple:

```python
str
int
bool
list
```

For these, the annotation itself is enough.

But some annotations are generic or special typing objects:

```python
list[str]
dict[str, int]
Annotated[int, "First number"]
Union[str, None]
```

For these, Python stores the outer type in `origin`.

Example:

```python
get_origin(list[str])     # list
get_args(list[str])       # (str,)
```

So:

```python
annotation is list
```

handles plain untyped lists.

```python
origin is list
```

handles typed lists like `list[str]`.

---

## `Annotated`

`Annotated` allows you to attach metadata to a type hint.

Example:

```python
a: Annotated[int, "First number"]
```

This means:

- Base type: `int`
- Metadata: `"First number"`

The project uses the first metadata item as the parameter description:

```python
if meta:
    description = str(meta[0])
```

So this function parameter:

```python
a: Annotated[int, "First number"]
```

becomes:

```json
{
  "type": "number",
  "description": "First number"
}
```

---

## `schema_for_callable()`

```python
def schema_for_callable(cls, func: Callable[..., Any]) -> dict[str, Any]:
```

This method builds a full OpenAI-style tool schema from a Python function.

It uses:

```python
inspect.signature(func)
```

to inspect function parameters.

It uses:

```python
inspect.get_annotations(func)
```

to read the type hints.

Example tool:

```python
@agent.tool
def add(
    a: Annotated[int, "First number"],
    b: Annotated[int, "Second number"]
) -> dict[str, int]:
    """Add two numbers together."""
    return {"result": a + b}
```

Generated schema:

```json
{
  "type": "function",
  "function": {
    "name": "add",
    "description": "Add two numbers together.",
    "parameters": {
      "type": "object",
      "properties": {
        "a": {
          "type": "number",
          "description": "First number"
        },
        "b": {
          "type": "number",
          "description": "Second number"
        }
      },
      "required": ["a", "b"],
      "additionalProperties": false
    },
    "strict": true
  }
}
```

---

## Function Description vs Parameter Description

There are two different sources of description:

| Description type | Source |
|---|---|
| Tool/function description | Function docstring via `func.__doc__` |
| Parameter description | `Annotated` metadata |

Example:

```python
def add(
    a: Annotated[int, "First number"],
    b: Annotated[int, "Second number"]
) -> dict[str, int]:
    """Add two numbers together."""
```

This becomes:

```text
Function description: Add two numbers together.
Parameter a description: First number
Parameter b description: Second number
```

---

## `register()`

```python
def register(self, func: Callable[..., Any]) -> Callable[..., Any]:
```

This method registers a function as a tool.

It does three things:

1. Checks whether the function already has a schema
2. Creates and attaches the schema if missing
3. Stores the function in the tool registry

```python
if getattr(func, self.TOOL_SCHEMA_ATTR, None) is None:
    setattr(func, self.TOOL_SCHEMA_ATTR, self.schema_for_callable(func))

self.tools[func.__name__] = func
return func
```

It returns the original function, so it can be used as a decorator:

```python
@agent.tool
def multiply(a: int, b: int):
    return {"result": a * b}
```

This is different from a wrapper-style decorator.

```text
timer_dec(func)  → returns enhanced_fn
register(func)   → returns original func
```

---

## `get_schemas()`

```python
def get_schemas(self) -> list[dict[str, Any]]:
```

This collects all schemas from registered tools.

```python
for fn in self.tools.values():
    s = getattr(fn, self.TOOL_SCHEMA_ATTR, None)
    if s is not None:
        out.append(s)
```

The result is sent to the LLM as:

```python
api_kwargs["tools"] = tool_schemas
```

---

## `execute()`

```python
def execute(self, tool_call: dict[str, Any]) -> dict[str, Any]:
```

This method executes a tool call produced by the LLM.

Example tool call from the model:

```json
{
  "id": "call_123",
  "type": "function",
  "function": {
    "name": "add",
    "arguments": "{\"a\": 2, \"b\": 3}"
  }
}
```

Execution flow:

```text
1. Get function name from tool_call["function"]["name"]
2. Find the Python function in self.tools
3. Parse JSON arguments
4. Call the Python function using fn(**args)
5. Return the result as a dictionary
```

Code:

```python
args = json.loads(fn_payload.get("arguments") or "{}")
result = fn(**args)
```

If the function returns a dictionary, it returns it directly.

If the function returns another type, it wraps it:

```python
{"result": result}
```

---

## `Agent` Class

The `Agent` class manages:

1. System prompt
2. Model endpoint
3. Tool registry
4. Dynamic context functions
5. Message history
6. Chat loop

```python
@dataclass
class Agent:
    system_prompt: str = "You are a helpful assistant"
    model: str = "qwen/qwen3.5-9b"
    base_url: str = "http://127.0.0.1:1234/v1"
    api_key: str = field(default="NO_API_KEY", repr=False)
    tools: Tools = field(default_factory=Tools)
    contexts: dict[str, Callable[[], str]] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)
```

---

## Why `field(default_factory=Tools)` Is Used

```python
tools: Tools = field(default_factory=Tools)
```

This creates a new `Tools()` instance for each `Agent()` instance.

This is important because each agent should have its own independent tool registry.

Good:

```python
agent1 = Agent()
agent2 = Agent()
```

Each agent gets its own tools.

Avoid this pattern:

```python
tools: Tools = Tools()
```

because it can accidentally share mutable state across instances.

---

## `@agent.tool`

```python
def tool(self, func: Callable[..., Any]) -> Callable[..., Any]:
    return self.tools.register(func)
```

This lets you register tools directly on that specific agent instance:

```python
@agent.tool
def add(a: int, b: int):
    return {"result": a + b}
```

This is instance-bound registration.

Meaning:

```text
This tool belongs to this agent only.
```

That is cleaner than using one global registry shared by all agents.

---

## `@agent.context`

```python
def context(self, func: Callable[[], str]) -> Callable[[], str]:
    self.contexts[func.__name__] = func
    return func
```

This registers a context function.

Example:

```python
@agent.context
def user_context() -> str:
    return (
        f"Current date and time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Current user: {getpass.getuser()}\n"
    )
```

The context function is called on every chat request.

This makes the context dynamic.

---

## Dynamic Context Injection

Inside `chat()`:

```python
context_content = "\n\n".join(
    f"<context>\n<{n}>{fn()}</{n}>\n</context>"
    for n, fn in self.contexts.items()
)
```

This executes all registered context functions and injects their output as a system message.

Example generated context:

```xml
<context>
<user_context>
Current date and time: 2026-05-09 14:30:00
Current user: yenlim
</user_context>
</context>
```

This is useful because local LLMs do not automatically know real-time information like:

- Current date
- Current time
- Current user
- Runtime environment
- App-specific state

---

## Message Flow

The model receives messages like this:

```python
prefix + self.messages
```

Where `prefix` contains:

```python
[
    {"role": "system", "content": self.system_prompt},
    {"role": "system", "content": context_content},
]
```

And `self.messages` contains the conversation history:

```python
[
    {"role": "user", "content": "10 * 80 million"},
    {"role": "assistant", "content": None, "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "call_123", "content": "{...}"},
    {"role": "assistant", "content": "The result is 800 million."}
]
```

---

## Tool Call Message Format

When the assistant asks for a tool, it returns something like:

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_123",
      "type": "function",
      "function": {
        "name": "multiply",
        "arguments": "{\"a\": 10, \"b\": 80000000}"
      }
    }
  ]
}
```

The agent then executes the tool and appends a tool result message:

```json
{
  "role": "tool",
  "tool_call_id": "call_123",
  "content": "{\"result\": 800000000}"
}
```

Then the model is called again so it can produce the final human-readable answer.

---

## Debug Output

This version includes debug prints inside `chat()`:

```python
print("\n================ DEBUG =================")
print("RAW CONTEXT:", json.dumps(context_content, indent=2))
print("RAW MESSAGE:", json.dumps(message, indent=2))
print("TOOL CALLS:", json.dumps(tool_calls, indent=2))
print("========================================\n")
```

This helps you see:

- What context is being injected
- What raw message the model returned
- Whether the model produced tool calls
- Which tool the model selected
- What arguments the model generated

This is useful for debugging why a tool was or was not called.

---

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install requests rich charset-normalizer
```

### 3. Run a local OpenAI-compatible model server

This code expects a local endpoint like:

```text
http://127.0.0.1:1234/v1
```

This is commonly used by tools like LM Studio when running in OpenAI-compatible server mode.

### 4. Run the app

```bash
python main.py
```

---

## Example Usage

```text
You: What is 10 * 80 million?
Assistant: The result is 800 million. Current date/time: 2026-05-09 14:30:00
```

The expected internal flow is:

```text
User asks calculation
↓
LLM sees multiply tool schema
↓
LLM returns tool call: multiply(a=10, b=80000000)
↓
Agent executes multiply
↓
Agent sends result back to LLM
↓
LLM gives final answer
```

---

## Important Limitation

Tool calling only works if the local model and API layer support OpenAI-style `tool_calls`.

If the model does not support tool calling, it may return plain text instead of structured tool calls.

In that case, you would need to emulate tool selection manually by prompting the model to return JSON and then parsing it yourself.

---

## Possible Improvements

### 1. Replace `print()` with logging

Instead of raw prints, use Python’s `logging` module.

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug("RAW MESSAGE: %s", json.dumps(message, indent=2))
```

### 2. Improve dict schema handling

Current code simplifies typed dictionaries:

```python
elif origin is dict:
    schema = {"type": "object"}
```

A better version could inspect key and value types:

```python
elif origin is dict:
    key_type, value_type = get_args(annotation)
    if key_type is not str:
        raise ValueError("JSON object keys must be strings")

    schema = {
        "type": "object",
        "additionalProperties": Tools._annotation_to_schema(value_type),
    }
```

### 3. Improve integer handling

Currently both `int` and `float` map to:

```json
{"type": "number"}
```

A stricter schema could map:

```python
int   -> {"type": "integer"}
float -> {"type": "number"}
```

### 4. Use Pydantic for schema generation

Pydantic can generate JSON schema automatically:

```python
from pydantic import BaseModel

class SearchFlightsArgs(BaseModel):
    airport: str
    limit: int = 10

print(SearchFlightsArgs.model_json_schema())
```

This can reduce the need to manually convert annotations.

---

## Key Learning Points

- An AI agent is not only a chatbot; it is a loop that can call tools and use the results.
- Python functions can be registered as LLM tools.
- Python type hints must be converted into JSON schema for tool calling.
- `Annotated` is useful for adding parameter descriptions.
- `inspect.signature()` reads function parameters.
- `inspect.get_annotations()` reads function type hints.
- `getattr()` and `setattr()` can attach custom metadata to functions.
- `@agent.tool` is an instance-bound decorator.
- `@agent.context` injects dynamic runtime information.
- Tool calls must be added back into the conversation using the correct message format.
- The loop continues until the model stops requesting tools.

---

## Mental Model

Think of the agent like this:

```text
Agent = LLM + memory + context + tools + loop
```

The LLM decides what should happen.

The Python agent controls what is allowed to happen.

Tools are the bridge between model reasoning and real program execution.
