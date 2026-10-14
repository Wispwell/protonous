"""The `protonous` command.

    protonous replay <session_id> [--root DIR] [--full]

`replay` prints a session as one line per run and one line per event (what the
agent thinks, each tool call with its result, its final answer), so a session can
be followed without reading the raw JSONL. Text is cut to one line unless --full.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .errors import ProtonousError
from .trace import ERRORS_FILE, SessionTrace, Trace, load_session

WIDTH = 160


def render_session(session_id: str, session: SessionTrace, *, full: bool = False) -> list[str]:
    created = session.created.strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        f"session {session_id} · created {created} · {len(session.runs)} run(s)",
        f"initial: {', '.join(session.initial) or '(none)'}",
    ]
    for trace in session.runs.values():
        lines.append("")
        lines.extend(render_run(trace, full=full))
    if session.errors:
        lines.append("")
        lines.append(f"{len(session.errors)} error(s); tracebacks in {ERRORS_FILE}")
    return lines


def render_run(trace: Trace, *, full: bool = False) -> list[str]:
    def cut(text: str) -> str:
        return _cut(text, full)

    if trace.error is not None:
        status = f"ERROR {trace.error.error}: {trace.error.message}"
    else:
        status = trace.stop_reason or "unfinished"
    lines = [cut(f"{trace.name} · {status} · {len(trace.steps)} step(s)")]
    if trace.meta:
        lines.append(cut(f"  meta: {json.dumps(trace.meta, ensure_ascii=False)}"))
    for i, step in enumerate(trace.steps):
        reply = step.assistant
        if reply.reasoning and reply.reasoning.strip():
            lines.append(cut(f"  [{i}] thinks: {reply.reasoning.strip()}"))
        for call, result in zip(reply.tool_calls, step.tool_results, strict=False):
            mark = "✗" if result.extras.get("error") else "→"
            lines.append(
                cut(f"  [{i}] {call.function_name}({call.function_arguments}) {mark} {result.content}")
            )
        if reply.content and reply.content.strip():
            lines.append(cut(f"  [{i}] says: {reply.content.strip()}"))
    return lines


def _cut(text: str, full: bool) -> str:
    """One line, at most WIDTH characters; the leading indent is kept."""
    if full:
        return text
    body = text.lstrip(" ")
    flat = " " * (len(text) - len(body)) + " ".join(body.split())
    return flat if len(flat) <= WIDTH else flat[: WIDTH - 1] + "…"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="protonous")
    commands = parser.add_subparsers(dest="command", required=True)
    replay = commands.add_parser("replay", help="print a session, one line per event")
    replay.add_argument("session_id", help="session directory name under --root")
    replay.add_argument("--root", default="sessions", help="sessions directory (default: sessions)")
    replay.add_argument("--full", action="store_true", help="don't cut long text")
    args = parser.parse_args(argv)

    path = Path(args.root) / args.session_id
    try:
        session = load_session(path)
    except FileNotFoundError:
        print(f"protonous: no session at {path}", file=sys.stderr)
        return 1
    except ProtonousError as e:
        print(f"protonous: {e}", file=sys.stderr)
        return 1
    print("\n".join(render_session(args.session_id, session, full=args.full)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
