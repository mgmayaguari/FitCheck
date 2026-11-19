"""
Guardrails in front of the agent — unit 9.

Two different problems, two different fixes. Conflating them is how you end
up with one knob that doesn't actually solve either one.

  1. A single client sending requests faster than the app can answer them.
     `serve.py` runs one agent call at a time on purpose (see its
     `_one_at_a_time` note — the free tier doesn't have room for two MCP
     subprocesses at once, and `generate.py`'s pacing assumes one caller).
     That means a script firing ten requests a second doesn't get ten fast
     failures — it gets ten requests queued behind each other, and every
     real user queues behind all of them. The fix is `IPRateLimiter`: reject
     a client going too fast before it ever reaches the lock. Cheap, no
     agent run, no model call spent on a request that was never going to be
     the one we answer.

  2. A single request that never comes back — a stuck model call, a stuck
     MCP subprocess. Without a bound, the caller's HTTP connection just hangs
     and the lock never frees. `run_with_timeout` bounds what the *caller*
     waits for by running the work in a background thread and giving up on
     waiting past a deadline.

     What it does NOT do: free the lock early. The lock models something
     real — one agent run fits in the free tier's memory; two don't — so
     letting a second run start while the first is still (maybe) alive would
     double the MCP process count and let two threads stomp on
     `generate.py`'s shared pacing state. A request that times out gets an
     honest answer ("this is taking too long") while the original work either
     finishes on its own or stays stuck — and if it's truly stuck, the
     service is still blocked until it isn't. A client-side timeout is not a
     fix for that; it's a bound on how long one caller is kept waiting to
     find out. The real fix is a work queue the server owns and can cancel,
     which is a bigger change than this unit calls for — see the README.
"""

import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeoutError

import config


class IPRateLimiter:
    """
    Sliding-window request count per client IP.

    Same idea as `generate.py`'s own `_call_times` pacing, just keyed by
    caller instead of applied globally: a deque of recent hit timestamps per
    key, trimmed to the window on every check.
    """

    def __init__(self, limit: int, window_seconds: float = 60.0):
        self._limit = limit
        self._window = window_seconds
        self._lock = threading.Lock()
        self._hits: dict[str, deque] = defaultdict(deque)
        self._checks_since_sweep = 0

    def allow(self, key: str) -> tuple[bool, float]:
        """
        Record one attempt from `key`. Returns (allowed, retry_after_seconds).

        retry_after is 0.0 when allowed — how long until the oldest hit in
        this window ages out, otherwise.
        """
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self._window:
                hits.popleft()

            if len(hits) >= self._limit:
                return False, max(self._window - (now - hits[0]), 0.1)

            hits.append(now)
            self._checks_since_sweep += 1
            if self._checks_since_sweep >= 200:
                self._sweep(now)
            return True, 0.0

    def _sweep(self, now: float) -> None:
        """
        Drop IPs with nothing left in their window.

        Without this, a long-lived process accumulates one deque per IP it
        has ever seen and never frees any of them — a slow memory leak
        rather than a DoS, but a leak all the same on a process meant to
        stay up. Called from inside `allow()`'s lock, every 200 checks rather
        than every one, so it doesn't cost every request a full-dict scan.
        """
        self._checks_since_sweep = 0
        stale = [
            k for k, hits in self._hits.items()
            if not hits or now - hits[-1] > self._window
        ]
        for k in stale:
            del self._hits[k]


# One shared pool for agent runs that might time out. Bounded, not unlimited:
# a run that times out from the caller's point of view keeps using a worker
# thread until it actually finishes (see the module docstring). Four workers
# means at most four such orphaned runs can be in flight before a fifth
# `/ask` call has to wait for a thread, same as it would wait for the
# `_one_at_a_time` lock anyway — this doesn't add a new failure mode, it
# bounds an existing one.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="agent-run")


def run_with_timeout(fn, timeout_seconds: float, *args, **kwargs):
    """
    Run fn(*args, **kwargs) and wait up to timeout_seconds for it.

    Raises TimeoutError if it doesn't finish in time. The call is NOT
    cancelled — it keeps running on its worker thread, and its result (or
    exception) is simply never collected. That's a deliberate trade-off, not
    an oversight: see the module docstring.
    """
    future = _executor.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout_seconds)
    except _FutureTimeoutError:
        raise TimeoutError(
            f"Timed out after {timeout_seconds:.0f}s waiting for a response."
        ) from None
