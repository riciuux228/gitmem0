"""Lightweight in-memory metrics collector for GitMem0 daemon.

Zero external dependencies. Thread-safe. Tracks per-action:
- Request counts
- Latency (avg, p95, min, max)
- Error counts
- Uptime
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict


class MetricsCollector:
    """In-memory metrics with sliding window latency tracking."""

    MAX_SAMPLES = 1000  # per-action latency window

    def __init__(self) -> None:
        self._start_time = time.monotonic()
        self._counters: dict[str, int] = defaultdict(int)
        self._latencies: dict[str, list[float]] = defaultdict(list)
        self._errors: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def record(self, action: str, latency_ms: float, success: bool) -> None:
        """Record one request's metrics."""
        with self._lock:
            self._counters[action] += 1
            if not success:
                self._errors[action] += 1
            buf = self._latencies[action]
            buf.append(latency_ms)
            # Sliding window: keep only last MAX_SAMPLES
            if len(buf) > self.MAX_SAMPLES:
                self._latencies[action] = buf[-self.MAX_SAMPLES:]

    def snapshot(self) -> dict:
        """Return current metrics snapshot."""
        with self._lock:
            uptime = time.monotonic() - self._start_time
            total_requests = sum(self._counters.values())
            total_errors = sum(self._errors.values())

            per_action = {}
            for action in sorted(self._counters.keys()):
                count = self._counters[action]
                errors = self._errors.get(action, 0)
                lats = self._latencies.get(action, [])
                if lats:
                    sorted_lats = sorted(lats)
                    n = len(sorted_lats)
                    avg_ms = sum(sorted_lats) / n
                    p95_idx = min(int(n * 0.95), n - 1)
                    p95_ms = sorted_lats[p95_idx]
                    per_action[action] = {
                        "count": count,
                        "errors": errors,
                        "avg_ms": round(avg_ms, 2),
                        "p95_ms": round(p95_ms, 2),
                        "min_ms": round(sorted_lats[0], 2),
                        "max_ms": round(sorted_lats[-1], 2),
                    }
                else:
                    per_action[action] = {
                        "count": count,
                        "errors": errors,
                    }

            return {
                "uptime_seconds": round(uptime, 1),
                "total_requests": total_requests,
                "total_errors": total_errors,
                "per_action": per_action,
            }

    def reset(self) -> None:
        """Reset all counters."""
        with self._lock:
            self._start_time = time.monotonic()
            self._counters.clear()
            self._latencies.clear()
            self._errors.clear()
