"""The `protonous` command.

    protonous replay <session_id> [--root DIR] [--full] [--follow [--interval S]]

`replay` prints a session as one line per run and one line per event (what the
agent thinks, each tool call with its result, its final answer), so a session can
be followed without reading the raw JSONL. Text is cut to one line unless --full.

With --follow it keeps watching: every event so far, then each new one as runs
write it (runs that start later included), one per line and prefixed with its run,
until Ctrl-C. Events arrive per completed step.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

from .context import Step, Usage
from .errors import ProtonousError
from .trace import (
    ERRORS_FILE,
    RunEnded,
    RunFailed,
    RunStarted,
    SessionFollower,
    SessionTrace,
    StepRecorded,
    Trace,
    load_session,
)

WIDTH = 160


def render_session(session_id: str, session: SessionTrace, *, full: bool = False) -> list[str]:
    created = session.created.strftime("%Y-%m-%d %H:%M:%S %Z")
    runs = f"{len(session.runs)} run(s)"
    lines = [
        f"session {session_id} · created {created} · {runs}{_usage_text(session.usage)}",
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
    lines = [cut(f"{trace.name} · {status} · {len(trace.steps)} step(s){_usage_text(trace.usage)}")]
    if trace.meta:
        lines.append(cut(f"  meta: {json.dumps(trace.meta, ensure_ascii=False)}"))
    for i, step in enumerate(trace.steps):
        lines.extend(cut(f"  {line}") for line in render_step(i, step))
    return lines


def render_step(index: int, step: Step) -> list[str]:
    """One step's events: what the agent thinks, each tool call with its result
    (`→`, or `✗` for an error result), and what it says. Not cut."""
    reply = step.assistant
    lines = []
    if reply.reasoning and reply.reasoning.strip():
        lines.append(f"[{index}] thinks: {reply.reasoning.strip()}")
    for call, result in zip(reply.tool_calls, step.tool_results, strict=False):
        mark = "✗" if result.extras.get("error") else "→"
        lines.append(f"[{index}] {call.function_name}({call.function_arguments}) {mark} {result.content}")
    if reply.content and reply.content.strip():
        lines.append(f"[{index}] says: {reply.content.strip()}")
    return lines


def follow(
    path: Path,
    *,
    full: bool = False,
    interval: float = 1.0,
    polls: int | None = None,
    out: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Print a session's events as they are written, until interrupted (or after
    `polls` polls, for tests)."""
    follower = SessionFollower(path)
    steps: dict[str, int] = {}
    usage: dict[str, Usage] = {}

    def done(run: str) -> str:
        return f"{steps.get(run, 0)} step(s){_usage_text(usage.get(run, Usage()))}"

    count = 0
    while polls is None or count < polls:
        for event in follower.poll():
            match event:
                case RunStarted(run=run, meta=meta):
                    meta_text = f" · meta {json.dumps(meta, ensure_ascii=False)}" if meta else ""
                    out(_cut(f"{run} · started{meta_text}", full))
                case StepRecorded(run=run, index=index, step=step):
                    steps[run] = steps.get(run, 0) + 1
                    usage[run] = usage.get(run, Usage()) + step.usage
                    for line in render_step(index, step):
                        out(_cut(f"{run} {line}", full))
                case RunEnded(run=run, stop_reason=stop_reason):
                    out(_cut(f"{run} · {stop_reason} · {done(run)}", full))
                case RunFailed(run=run, error=error):
                    out(_cut(f"{run} · ERROR {error.error}: {error.message} · {done(run)}", full))
        count += 1
        if polls is None or count < polls:
            sleep(interval)


def _usage_text(usage: Usage) -> str:
    """` · 3,812 tokens (3,500 in, 312 out) · $0.0021`, or "" if nothing was reported."""
    parts = []
    if usage.total_tokens is not None:
        detail = ""
        if usage.prompt_tokens is not None and usage.completion_tokens is not None:
            detail = f" ({usage.prompt_tokens:,} in, {usage.completion_tokens:,} out)"
        parts.append(f"{usage.total_tokens:,} tokens{detail}")
    if usage.cost is not None:
        parts.append(f"${usage.cost:.4f}")
    return "".join(f" · {part}" for part in parts)


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
    replay.add_argument(
        "-f", "--follow", action="store_true", help="keep watching for new events until Ctrl-C"
    )
    replay.add_argument(
        "--interval", type=float, default=1.0, help="seconds between checks with --follow"
    )
    args = parser.parse_args(argv)

    path = Path(args.root) / args.session_id
    if args.follow:
        try:
            print(f"following {path} (Ctrl-C to stop)")
            follow(path, full=args.full, interval=args.interval)
        except ProtonousError as e:
            print(f"protonous: {e}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            pass
        return 0
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
