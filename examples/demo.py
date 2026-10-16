"""protonous end to end: tools, layers, a custom update, a session, a trace.

The agent gets a small arithmetic task and a notepad. A state layer shows it the
notepad's current contents every step, refreshed by the update callback. The run
is traced into a session directory, then loaded back and summarized.

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
from protonous.loop import Step, append_to, run
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


async def main(model: str) -> None:
    state = SingleMessageLayer()
    state.add(notepad_state())
    history = SlidingWindowLayer(max_messages=None)
    history.add(Message(role=Role.USER, content=TASK))
    window = ContextWindow(
        layers=[StaticLayer([Message(role=Role.SYSTEM, content=SYSTEM)]), state, history]
    )

    remember = append_to(history)

    def update(window: ContextWindow, step: Step) -> ContextWindow:
        window = remember(window, step)  # the step goes into history
        state.add(notepad_state())  # and the notepad view is refreshed
        return window

    config = OpenrouterConfig(model=model, max_tokens=2048, reasoning_effort="low")
    session = Session.create(
        f"sessions/demo-{datetime.now(UTC):%Y%m%d-%H%M%S}",
        {"config": config, "window": window.view(), "tools": [t.spec for t in tools]},
    )

    result = await run(
        OpenrouterClient(get_settings()),
        window,
        config,
        tools,
        update=update,
        max_steps=10,
        tracer=session.tracer(meta={"task": TASK}),
    )

    # -- read the session back (same view as `protonous replay <session_id>`) --
    print("\n".join(render_session(session.path.name, load_session(session.path))))
    print(f"\nnotepad: {notepad}")
    print(f"stop reason: {result.stop_reason}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python examples/demo.py <openrouter-model-id>")
    asyncio.run(main(sys.argv[1]))
