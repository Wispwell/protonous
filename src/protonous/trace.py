"""Sessions and run traces.

A session is a directory: the initial data shared by its runs, one JSONL file per
run (one agent loop, however the caller writes it), and one file collecting every
run's failures.

    <session>/
      session.json          {"created": ..., "initial": {...}}
      run-0001.jsonl        one file per run:
        {"type": "run", "name": "run-0001", "meta": {...}}     the run's own metadata
        {"type": "step", "index": 0, "step": {...}}             a serialized Step
        {"type": "end", "stop_reason": "done", "steps": 3}      tracer.end(), or
        {"type": "error", "error": "ProviderError", "message": "..."}   raised in recording()
      errors.jsonl          {"run", "time", "error", "message", "traceback"} per failure

Every line is appended and the file closed immediately, so a crash loses nothing.
A run file with neither an end nor an error line was killed outright.
"""
from __future__ import annotations

import json
import re
import traceback
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from .context import Step, Usage
from .errors import (
    ConfigurationError,
    NotASessionError,
    RunExistsError,
    SessionExistsError,
    TraceFormatError,
)

SESSION_FILE = "session.json"
ERRORS_FILE = "errors.jsonl"
_AUTO_NAME = re.compile(r"run-(\d+)")
_SAFE_NAME = re.compile(r"[A-Za-z0-9._-]+")


class RunError(BaseModel):
    error: str  # exception type name, e.g. "ProviderError", "KeyboardInterrupt"
    message: str


class ErrorRecord(RunError):
    run: str
    time: datetime
    traceback: str


class Trace(BaseModel):
    name: str
    meta: dict[str, Any]
    steps: list[Step]
    stop_reason: str | None  # None: the run didn't finish
    error: RunError | None  # set if the run raised inside tracer.recording()

    @property
    def usage(self) -> Usage:
        """Summed over the recorded steps (a failed run's partial reply isn't one)."""
        return sum((step.usage for step in self.steps), Usage())


class SessionTrace(BaseModel):
    created: datetime
    initial: dict[str, Any]
    runs: dict[str, Trace]  # by run name, sorted
    errors: list[ErrorRecord]  # in the order they happened

    @property
    def usage(self) -> Usage:
        """Summed over every run in the session."""
        return sum((trace.usage for trace in self.runs.values()), Usage())


class Tracer:
    """Writes one run's file. The file must not exist yet: one file, one run.
    `meta` is anything JSON-serializable (Pydantic models included). Failures also
    go, with their traceback, to errors.jsonl next to the run file."""

    def __init__(self, path: str | Path, meta: Mapping[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.name = self.path.stem
        self._index = 0
        try:
            self._write({"type": "run", "name": self.name, "meta": dict(meta or {})}, mode="x")
        except FileExistsError as e:
            raise RunExistsError(f"{self.path}: run already exists") from e

    def step(self, step: Step) -> None:
        self._write({"type": "step", "index": self._index, "step": step})
        self._index += 1

    def end(self, stop_reason: str) -> None:
        """Mark the run finished, for whatever reason the caller's loop stopped."""
        self._write({"type": "end", "stop_reason": stop_reason, "steps": self._index})

    @contextmanager
    def recording(self) -> Iterator[None]:
        """protonous's one exception handler. Anything that escapes the block
        (including Ctrl-C and cancellation) is recorded, then re-raised unchanged."""
        try:
            yield
        except BaseException as e:
            self.error(e)
            raise

    def error(self, exc: BaseException) -> None:
        summary = {"error": type(exc).__name__, "message": str(exc)}
        self._write({"type": "error", **summary})
        record = {
            "run": self.name,
            "time": datetime.now(UTC),
            **summary,
            "traceback": "".join(traceback.format_exception(exc)),
        }
        _append(self.path.parent / ERRORS_FILE, record)

    def _write(self, record: dict[str, Any], mode: str = "a") -> None:
        _append(self.path, record, mode)


class Session:
    """A directory of runs sharing initial data. Use `create` to start one and
    `open` to add runs to an existing one."""

    def __init__(self, path: Path) -> None:  # use create() / open()
        self.path = path

    @classmethod
    def create(cls, path: str | Path, initial: Mapping[str, Any]) -> Session:
        """Start a new session. Fails if the directory already exists."""
        path = Path(path)
        try:
            path.mkdir(parents=True)
        except FileExistsError as e:
            raise SessionExistsError(f"{path}: already exists") from e
        record = {"created": datetime.now(UTC), "initial": dict(initial)}
        (path / SESSION_FILE).write_text(_dumps(record, indent=2) + "\n", encoding="utf-8")
        return cls(path)

    @classmethod
    def open(cls, path: str | Path) -> Session:
        """Open an existing session to add runs. Fails if it has no session.json."""
        path = Path(path)
        if not (path / SESSION_FILE).is_file():
            raise NotASessionError(f"{path}: not a session (no {SESSION_FILE})")
        return cls(path)

    def tracer(self, name: str | None = None, meta: Mapping[str, Any] | None = None) -> Tracer:
        """A Tracer for a new run file. Unnamed runs get the next free run-NNNN,
        claimed atomically, so runs started in parallel never share a file."""
        if name is not None:
            if not _SAFE_NAME.fullmatch(name):
                raise ConfigurationError(
                    f"run name {name!r}: use letters, digits, '.', '_' or '-'"
                )
            if f"{name}.jsonl" == ERRORS_FILE:
                raise ConfigurationError(f"run name {name!r} is reserved")
            return Tracer(self.path / f"{name}.jsonl", meta)
        number = self._next_number()
        while True:
            try:
                return Tracer(self.path / f"run-{number:04d}.jsonl", meta)
            except RunExistsError:
                number += 1

    def _next_number(self) -> int:
        taken = [
            int(m.group(1))
            for p in self.path.glob("run-*.jsonl")
            if (m := _AUTO_NAME.fullmatch(p.stem))
        ]
        return max(taken, default=0) + 1


def load_trace(path: str | Path) -> Trace:
    name: str | None = None
    meta: dict[str, Any] = {}
    steps: list[Step] = []
    stop_reason: str | None = None
    error: RunError | None = None
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            match record["type"]:
                case "run":
                    name, meta = record["name"], record["meta"]
                case "step":
                    steps.append(Step.model_validate(record["step"]))
                case "end":
                    stop_reason = record["stop_reason"]
                case "error":
                    error = RunError(error=record["error"], message=record["message"])
    if name is None:
        raise TraceFormatError(f"{path}: no run header line")
    return Trace(name=name, meta=meta, steps=steps, stop_reason=stop_reason, error=error)


def load_session(path: str | Path) -> SessionTrace:
    path = Path(path)
    record = json.loads((path / SESSION_FILE).read_text(encoding="utf-8"))
    run_files = sorted(p for p in path.glob("*.jsonl") if p.name != ERRORS_FILE)
    runs = {trace.name: trace for trace in map(load_trace, run_files)}
    errors_path = path / ERRORS_FILE
    errors = (
        [ErrorRecord.model_validate_json(line) for line in errors_path.read_text().splitlines()]
        if errors_path.is_file()
        else []
    )
    return SessionTrace(
        created=record["created"], initial=record["initial"], runs=runs, errors=errors
    )


def _append(path: Path, record: dict[str, Any], mode: str = "a") -> None:
    with path.open(mode, encoding="utf-8") as f:
        f.write(_dumps(record) + "\n")


def _dumps(record: dict[str, Any], indent: int | None = None) -> str:
    return json.dumps(to_jsonable_python(record), ensure_ascii=False, indent=indent)
