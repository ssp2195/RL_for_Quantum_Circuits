"""Shared deadline and status helpers for the qualification runner."""
from __future__ import annotations

import json
import time
from typing import Any


def _status(payload: dict[str, Any]) -> None:
    print("STATUS " + json.dumps(payload, sort_keys=True), flush=True)


def _deadline_reached(cpu_start: float, wall_start: float, limit: float) -> bool:
    return max(time.process_time() - cpu_start, time.perf_counter() - wall_start) >= limit


__all__ = ["_deadline_reached", "_status"]
