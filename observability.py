"""
Instrumentation for serve.py — unit 9.

Two things live here, and nothing else does:

  1. Structured logging. One JSON line per request, to stdout. Not a file —
     a host doesn't give you a disk worth trusting, but every one of them
     captures stdout and lets you search it. `docker logs`, `heroku logs`,
     Render's log stream, `journalctl` — they all just read stdout.

  2. In-memory metrics. Counts by outcome, latency percentiles, and the
     model-call/token/cache numbers `generate.py` already tracks, pulled
     into one place a host can show you without you grepping logs by hand.
     This is one process with one worker (see serve.py's `_one_at_a_time`
     note), so a plain lock is enough — no Redis, no Prometheus client,
     nothing you'd need a second service to run.

Logging and metrics are kept separate on purpose: a log line answers "what
happened on request abc123?"; metrics answer "how is the service doing right
now?". Conflating them is how you end up grepping a log file to answer a
question a dashboard should answer in one glance.
"""

import json
import logging
import sys
import threading
import time
import uuid

# ─── Logging ──────────────────────────────────────────────────────────────────


class _JSONFormatter(logging.Formatter):
    """One JSON object per line. No multi-line stack traces to grep around."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
        }
        # `extra={...}` on the logging call lands on the record as attributes.
        # `getattr(record, "fields", None)` is how we pull that dict back out.
        fields = getattr(record, "fields", None)
        if fields:
            payload.update(fields)
        else:
            payload["message"] = record.getMessage()
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging() -> logging.Logger:
    """
    Call this once, at import time. Returns the logger `serve.py` logs through.

    Writes to stdout, not stderr — stdout is what a host's log pipeline
    captures as "the app's output"; stderr is where this repo's existing
    `[rate limit]` / `[serve]` notices already go, and mixing structured JSON
    into that stream would make both harder to read.
    """
    logger = logging.getLogger("fitfindr")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JSONFormatter())
        logger.addHandler(handler)
    return logger


def log_event(logger: logging.Logger, **fields) -> None:
    """Log one structured event. Every field you pass becomes a JSON key."""
    logger.info("", extra={"fields": fields})


def new_request_id() -> str:
    """Short, unique enough to grep for across a log stream, not a UUID wall."""
    return uuid.uuid4().hex[:12]


# ─── Metrics ──────────────────────────────────────────────────────────────────


class Metrics:
    """
    Counters and latencies for one running process.

    `snapshot()` is what `/metrics` returns. Percentiles are computed from the
    last `_MAX_SAMPLES` request durations — enough to be meaningful, small
    enough that memory doesn't grow across a long-lived process.
    """

    _MAX_SAMPLES = 500

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._outcomes: dict[str, int] = {}
        self._durations_ms: list[float] = []
        self._model_calls = 0
        self._cache_hits = 0
        self._prompt_tokens = 0
        self._output_tokens = 0

    def record(
        self,
        outcome: str,
        duration_ms: float,
        model_calls: int = 0,
        cache_hits: int = 0,
        prompt_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        with self._lock:
            self._outcomes[outcome] = self._outcomes.get(outcome, 0) + 1
            self._durations_ms.append(duration_ms)
            if len(self._durations_ms) > self._MAX_SAMPLES:
                self._durations_ms.pop(0)
            self._model_calls += model_calls
            self._cache_hits += cache_hits
            self._prompt_tokens += prompt_tokens
            self._output_tokens += output_tokens

    def snapshot(self) -> dict:
        with self._lock:
            durations = sorted(self._durations_ms)
            total = sum(self._outcomes.values())
            return {
                "uptime_seconds": round(time.time() - self._started_at, 1),
                "requests_total": total,
                "requests_by_outcome": dict(self._outcomes),
                "latency_ms": _percentiles(durations),
                "model_calls_total": self._model_calls,
                "cache_hits_total": self._cache_hits,
                "tokens_total": {
                    "prompt": self._prompt_tokens,
                    "output": self._output_tokens,
                },
            }


def _percentiles(sorted_values: list[float]) -> dict:
    if not sorted_values:
        return {"p50": None, "p95": None, "max": None, "samples": 0}

    def pick(pct: float) -> float:
        idx = min(len(sorted_values) - 1, int(len(sorted_values) * pct))
        return round(sorted_values[idx], 1)

    return {
        "p50": pick(0.50),
        "p95": pick(0.95),
        "max": round(sorted_values[-1], 1),
        "samples": len(sorted_values),
    }
