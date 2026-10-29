"""Monotonic progress accounting and durable JSON artifacts without ML imports."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, TextIO


def to_jsonable(value: Any) -> Any:
    """Convert manifest values explicitly while rejecting nonfinite statistics."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON artifacts cannot contain NaN or infinite values.")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON artifact mapping keys must be strings.")
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    raise TypeError(f"Unsupported JSON artifact value: {type(value).__name__}")


def atomic_write_json(path: str | Path, value: Any) -> Path:
    """Flush a complete manifest to a sibling temporary file before replacing it."""

    target = Path(path)
    encoded = json.dumps(to_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                         prefix=f".{target.name}.", suffix=".tmp", delete=False) as output:
            temporary = output.name
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return target


def append_jsonl(path: str | Path, value: Any) -> Path:
    """Append one prevalidated event to a single-process suite's durable event log."""

    target = Path(path)
    encoded = json.dumps(to_jsonable(value), sort_keys=True, allow_nan=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
    return target


def _duration(seconds: float | None) -> str:
    """Render an unknown ETA distinctly from a measured zero-second duration."""

    if seconds is None:
        return "unknown"
    rounded = max(0, int(round(seconds)))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class ProgressTracker:
    """Measure one run or the whole suite using an independently owned clock."""

    def __init__(self, total_units: float, label: str, *, unit: str = "examples",
                 event_path: str | Path | None = None, stream: TextIO | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        """Start a tracker without claiming throughput before useful work occurs."""

        if isinstance(total_units, bool) or not isinstance(total_units, (int, float)) or not math.isfinite(total_units) or total_units <= 0:
            raise ValueError("total_units must be positive and finite.")
        if not isinstance(label, str) or not label.strip() or not isinstance(unit, str) or not unit.strip():
            raise ValueError("label and unit must be nonempty strings.")
        self.total_units = float(total_units)
        self.label = label
        self.unit = unit
        self.event_path = None if event_path is None else Path(event_path)
        self.stream = sys.stdout if stream is None else stream
        self._clock = clock
        self._started = self._read_clock()
        self._last_time = self._started
        self._ended: float | None = None
        self.completed = 0.0
        self.status = "running"

    def _read_clock(self) -> float:
        """Reject invalid injected clocks to protect timing and ETA provenance."""

        value = self._clock()
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("The monotonic clock must return a finite number.")
        return float(value)

    def snapshot(self) -> dict[str, Any]:
        """Return measured timings; an ETA exists only after positive progress."""

        now = self._ended if self._ended is not None else self._read_clock()
        if now < self._last_time:
            raise ValueError("The monotonic clock moved backwards.")
        self._last_time = now
        elapsed = now - self._started
        throughput = self.completed / elapsed if self.completed > 0 and elapsed > 0 else None
        remaining = max(0.0, self.total_units - self.completed)
        eta = remaining / throughput if throughput is not None else None
        if self.status == "completed":
            eta = 0.0
        elif self.status == "failed":
            eta = None
        return {
            "label": self.label, "unit": self.unit, "status": self.status,
            "completed": self.completed, "total_units": self.total_units,
            "fraction": self.completed / self.total_units,
            "elapsed_seconds": elapsed, "throughput_per_second": throughput,
            "eta_seconds": eta, "timing_kind": "measured",
        }

    def _emit(self, event: dict[str, Any]) -> dict[str, Any]:
        """Record and display exactly the same measured progress snapshot."""

        to_jsonable(event)
        if self.event_path is not None:
            append_jsonl(self.event_path, event)
        rate = event["throughput_per_second"]
        rendered_rate = "unknown" if rate is None else f"{rate:.3f} {self.unit}/s"
        print(
            f"[{self.label}] {100 * event['fraction']:.1f}% "
            f"{event['completed']:g}/{event['total_units']:g} {self.unit} "
            f"elapsed={_duration(event['elapsed_seconds'])} "
            f"throughput={rendered_rate} ETA={_duration(event['eta_seconds'])} "
            f"status={event['status']}", file=self.stream, flush=True,
        )
        return event

    def update(self, completed: float, *, emit: bool = True, **metadata: Any) -> dict[str, Any]:
        """Set absolute completed work without allowing regressions or overflow."""

        if self.status != "running":
            raise RuntimeError("A finished progress tracker cannot be updated.")
        if isinstance(completed, bool) or not isinstance(completed, (int, float)) or not math.isfinite(completed):
            raise ValueError("completed must be finite.")
        if completed < self.completed or completed > self.total_units:
            raise ValueError("completed must be nondecreasing and no greater than total_units.")
        previous = self.completed
        self.completed = float(completed)
        try:
            event = self.snapshot()
            if set(metadata).intersection(event):
                raise ValueError("Metadata must not overwrite measured progress fields.")
            event.update(metadata)
            to_jsonable(event)
        except Exception:
            self.completed = previous
            raise
        return self._emit(event) if emit else event

    def advance(self, units: float = 1, *, emit: bool = True, **metadata: Any) -> dict[str, Any]:
        """Advance by a measured number of completed examples, steps, or runs."""

        if isinstance(units, bool) or not isinstance(units, (int, float)) or not math.isfinite(units) or units < 0:
            raise ValueError("units must be finite and nonnegative.")
        return self.update(self.completed + units, emit=emit, **metadata)

    def finish(self, status: str = "completed", *, emit: bool = True, **metadata: Any) -> dict[str, Any]:
        """Freeze run timing on completion or failure and persist its final event."""

        if status not in ("completed", "failed"):
            raise ValueError("Final status must be completed or failed.")
        if self.status != "running":
            raise RuntimeError("A progress tracker can only finish once.")
        now = self._read_clock()
        if now < self._last_time:
            raise ValueError("The monotonic clock moved backwards.")
        previous_completed = self.completed
        if status == "completed":
            self.completed = self.total_units
        self.status, self._ended = status, now
        try:
            event = self.snapshot()
            if set(metadata).intersection(event):
                raise ValueError("Metadata must not overwrite measured progress fields.")
            event.update(metadata)
            to_jsonable(event)
        except Exception:
            self.completed, self.status, self._ended = previous_completed, "running", None
            raise
        return self._emit(event) if emit else event
