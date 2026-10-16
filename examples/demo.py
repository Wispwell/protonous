"""protonous end to end: tools, layers, a model, a hand-written loop, a trace.

The agent gets a small arithmetic task and a notepad. A state layer at the end of
the window shows it the notepad's current contents, refreshed by the loop after
every step. The run is traced into a session directory, then loaded back and
summarized.

    cp .env.example .env            # add OPENROUTER_API_KEY
    python examples/demo.py <model>   # e.g. a reasoning model with tool support
"""
from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime
from typing import Annotated

from pydantic import Field

from protonous.cli import render_session
from protonous.client import OpenrouterClient, OpenrouterConfig
from protonous.context import Message, Role
from protonous.layers import ContextWindow, SingleMessageLayer, SlidingWindowLayer, StaticLayer
from protonous.model import Model
from protonous.settings import get_settings
from protonous.tools import Toolbox
from protonous.trace import Session, load_session

SYSTEM = (
    "You are a careful assistant. Use the tools for every calculation, and record "
    "results in the notepad before giving your final answer."
)
TASK = "Compute (17 + 25) * 3. Write the result to the notepad, then state it."


# -- tools ------------------------------------------------------------------------

notepad: list[str] = []
tools = Toolbox()


@tools.tool
async def add(a: float, b: float) -> str:
    """Add two numbers."""
    return str(a + b)


@tools.tool
async def multiply(a: float, b: float) -> str:
    """Multiply two numbers."""
    return str(a * b)


@tools.tool
async def write_note(text: Annotated[str, Field(description="What to write down.")]) -> str:
    """Write a line to the notepad."""
    notepad.append(text)
    return f"noted ({len(notepad)} note(s) total)"


# -- context ----------------------------------------------------------------------

def notepad_state() -> Message:
    lines = "\n".join(f"- {line}" for line in notepad) or "(empty)"
    return Message(role=Role.SYSTEM, content=f"Notepad contents:\n{lines}")


async def main(model_id: str) -> None:
    history = SlidingWindowLayer()
    history.add(Message(role=Role.USER, content=TASK))
    state = SingleMessageLayer()  # last, so refreshing it never invalidates the cached prefix
    state.add(notepad_state())
    window = ContextWindow(
        layers=[StaticLayer([Message(role=Role.SYSTEM, content=SYSTEM)]), history, state]
    )

    config = OpenrouterConfig(model=model_id, max_tokens=2048, reasoning_effort="low")
    model = Model(OpenrouterClient(get_settings()), config)
    session = Session.create(
        f"sessions/demo-{datetime.now(UTC):%Y%m%d-%H%M%S}",
        {"config": config, "window": window.view(), "tools": [t.spec for t in tools]},
    )
    tracer = session.tracer(meta={"task": TASK})

    # -- the agent loop: ours to write ----------------------------------------------
    stop = "max_steps"
    with tracer.recording():
        for _ in range(10):
            step = await model.step(window, tools)
            tracer.step(step)
            history.extend(step.messages)  # the step goes into history
            state.add(notepad_state())  # and the notepad view is refreshed
            if not step.tool_results:  # answered without tools: done
                stop = "done"
                break
        tracer.end(stop)

    # -- read the session back (same view as `protonous replay <session_id>`) --
    print("\n".join(render_session(session.path.name, load_session(session.path))))
    print(f"\nnotepad: {notepad}")
    print(f"stop reason: {stop}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python examples/demo.py <openrouter-model-id>")
    asyncio.run(main(sys.argv[1]))
