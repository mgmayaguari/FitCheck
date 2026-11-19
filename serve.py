#!/usr/bin/env python3
"""
FitFindr over HTTP. ← UNIT 9

    python serve.py                       run it on your machine
    gunicorn serve:app                    run it the way a host runs it

Everything else in this repo exits after one command. A host has nothing to
keep running, so a command line cannot be deployed. This file is the smallest
thing that can: it starts a web server, waits, and hands each request to the
same `run_agent()` your CLI calls. **No new agent logic lives here.** If your
loop works from `python app.py ask`, it works from here; if it doesn't, fix it
there, not in this file.

Two routes:

    POST /ask       {"query": "vintage graphic tee under $30",
                     "wardrobe": {...}}     ← wardrobe is optional
    GET  /health    is it awake?

`/ask` returns the session dict from `run_agent()` as JSON, unchanged — the
same keys you print from the command line, including `error`. Read
`session["error"]` first, exactly as you do everywhere else.

It runs **one request at a time**, and one route is deliberately slower than a
web server has to be. The reason is worth reading before you deploy: see the
note above `_one_at_a_time` below — it also decides which gunicorn worker you
can use.

Instrumentation (unit 9's follow-along, done): every `/ask` call is timed and
logged as one structured JSON line to stdout — that's what a host's log
pipeline actually captures, so that's where it goes, not a local file. A
`/metrics` route exposes the running totals a host dashboard would want:
request counts by outcome, latency percentiles, model calls, cache hits,
tokens. See `observability.py` for why it's split into logging (what happened
on *this* request) versus metrics (how is the service doing *right now*).

Two guardrails on top of that, in `limits.py`: a per-IP rate limit (a script
firing requests back to back otherwise makes every other caller queue behind
it — see `_one_at_a_time` below) and a per-request timeout (a caller gets a
bounded wait instead of hanging forever on a stuck model call). Neither one
is a cap on this app pretending to be busier than it is; both exist because
`/ask` answers one caller at a time, so one bad caller can otherwise degrade
the service for everyone else, not just themselves.
"""

import os
import sys
import threading
import time

# config first, and before anything that talks to the model: it loads your .env
# and fixes the Windows console encoding for every module that imports it.
import config  # noqa: F401

from flask import Flask, jsonify, request, send_from_directory

import generate
from agent import run_agent
from generate import ModelUnavailable, QuotaGuard
from limits import IPRateLimiter, run_with_timeout
from observability import Metrics, configure_logging, log_event, new_request_id
from utils.data_loader import get_example_wardrobe

app = Flask(__name__)
_logger = configure_logging()
_metrics = Metrics()
_rate_limiter = IPRateLimiter(config.RATE_LIMIT_PER_MINUTE)


# ── one request at a time ─────────────────────────────────────────────────────
# Here's why this exists.
#
# A web server will happily run two requests at once. Two things in this repo
# assume otherwise:
#
#   • If you moved a tool onto MCP in unit 4, `mcp_client.call_tool` starts
#     `mcp_server.py` as a separate process over stdio, one per call. Two
#     overlapping requests means two extra Python interpreters, and the free
#     tier you're deploying to has 512 MB for all of it. That is the reason
#     that would still be here even if everything else were perfect.
#   • `generate.py`'s pacing window and its call counter are plain module
#     globals — deliberately, because that file is teaching pacing and quota,
#     not locking. Two threads sharing them means the per-minute limiter is
#     counting something that isn't true.
#
# What the lock covers is exactly the agent run: the part that spawns
# processes and calls the model. Reading the request, rejecting a bad one, and
# turning the answer into JSON all happen outside it — so a request with a
# malformed body comes straight back with a 400 instead of queueing behind
# somebody else's agent.
#
# Serialising the runs costs you nothing here — this is one user asking one
# question — and it keeps the deployed behaviour identical to the CLI's. When
# that stops being an acceptable trade, the fix is a real queue, not a bigger
# lock.
#
# The same constraint decides your start command, and this is the part that
# bites in deployment rather than on your machine. `mcp_client.call_tool`
# wraps `asyncio.run()`, and `asyncio.run()` refuses to start when an event
# loop is already running. An async worker always has one running. So under
# `gunicorn -k gevent` (or `-k eventlet`, or `serve:app` wrapped for an ASGI
# server like uvicorn) EVERY MCP call fails with "asyncio.run() cannot be
# called from a running event loop" — not some of them under load, all of
# them, every time, in a place your laptop never showed you.
#
#     gunicorn serve:app              ← right: the DEFAULT sync worker
#     gunicorn -k gevent serve:app    ← breaks every MCP call you make

_one_at_a_time = threading.Lock()


def _client_ip() -> str:
    """
    The caller's IP, for rate limiting.

    Render (like most hosts) sits in front of this app as a reverse proxy, so
    `request.remote_addr` is Render's own edge, not the caller. The real
    client IP is the first entry in `X-Forwarded-For`, which the proxy sets —
    a direct caller can't forge the proxy's own header the same way, since
    Render appends to it rather than passing through whatever a client sent.
    Falls back to `remote_addr` for local runs, where there's no proxy at all.
    """
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _run_agent_locked(query: str, wardrobe: dict) -> dict:
    """The actual agent run, holding `_one_at_a_time` for its whole duration."""
    with _one_at_a_time:
        return run_agent(query, wardrobe)


@app.get("/")
def index():
    """
    The terminal-style frontend — static/index.html, a plain fetch() against
    /ask. No build step, no framework: it's one file, served as-is. Flask
    already serves everything else under static/ at /static/<path>; this
    just answers the bare "/" a browser actually requests.
    """
    return send_from_directory(app.static_folder, "index.html")


@app.get("/health")
def health():
    """Is the server awake? A host pings something like this to find out."""
    return jsonify({"status": "ok"})


@app.get("/metrics")
def metrics():
    """
    Running totals for this process: request counts by outcome, latency
    percentiles, model calls, cache hits, tokens. See observability.py.

    This is one process's view, not a fleet's — it resets on restart, and a
    host that runs more than one instance of this app would need something
    that isn't a module-level dict. Fine for this deployment: serve.py is
    built to answer one request at a time on one instance (see
    `_one_at_a_time` above), so there's only ever one of these to look at.
    """
    return jsonify(_metrics.snapshot())


@app.post("/ask")
def ask():
    """
    Run the agent on one query.

    In:   {"query": "...", "wardrobe": {...}}   wardrobe optional — leave it
                                                out and you get the same
                                                example wardrobe the CLI uses.
    Out:  the session dict from run_agent(), as JSON.
    """
    request_id = new_request_id()
    started = time.monotonic()
    client_ip = _client_ip()

    # The cheapest possible rejection, before any JSON parsing or agent work:
    # a client going too fast is turned away here, never reaches the lock,
    # and costs nothing but a dict lookup.
    allowed, retry_after = _rate_limiter.allow(client_ip)
    if not allowed:
        duration_ms = (time.monotonic() - started) * 1000
        log_event(
            _logger,
            event="ask",
            request_id=request_id,
            outcome="rate_limited",
            status=429,
            duration_ms=round(duration_ms, 1),
            client_ip=client_ip,
            retry_after=round(retry_after, 1),
        )
        _metrics.record("rate_limited", duration_ms)
        response = jsonify({
            "error": f"Too many requests. Try again in {retry_after:.0f}s.",
            "request_id": request_id,
        })
        response.status_code = 429
        response.headers["Retry-After"] = str(int(retry_after) + 1)
        return response

    payload = request.get_json(silent=True) or {}
    query = (payload.get("query") or "").strip()
    if not query:
        duration_ms = (time.monotonic() - started) * 1000
        log_event(
            _logger,
            event="ask",
            request_id=request_id,
            outcome="bad_request",
            status=400,
            duration_ms=round(duration_ms, 1),
        )
        _metrics.record("bad_request", duration_ms)
        # 400 rather than a stack trace: the caller can act on this one.
        return jsonify({"error": "Send JSON with a 'query' field, e.g. "
                                 '{"query": "vintage graphic tee under $30"}',
                         "request_id": request_id}), 400

    wardrobe = payload.get("wardrobe")
    if not isinstance(wardrobe, dict):
        wardrobe = get_example_wardrobe()

    # A queued request should look like a queued request. Without this the
    # second caller just sits there, and "slow" and "waiting its turn" are
    # indistinguishable from the outside — which is how people conclude the
    # deployment is broken when it is behaving exactly as designed.
    waited_for_lock = _one_at_a_time.locked()
    if waited_for_lock:
        print(
            "  [serve] another request is still running. This one is waiting "
            "its turn — serve.py handles one at a time, on purpose.",
            file=sys.stderr,
            flush=True,
        )

    calls_before = generate.call_count()
    tokens_before = generate.token_counts()
    cache_hits_before = generate.cache_hit_count()

    status = 200
    outcome = "completed"
    error_detail = None
    session = None

    try:
        session = run_with_timeout(
            _run_agent_locked, config.REQUEST_TIMEOUT_SECONDS, query, wardrobe,
        )
        if session["error"]:
            # A clean "nothing matched" or "model unavailable" message
            # from run_agent is a normal outcome of the loop, not a server
            # failure — it's still a 200, just one worth telling apart
            # from a fit card in the numbers.
            outcome = (
                "model_unavailable"
                if "model" in session["error"].lower()
                else "no_match"
            )
    except TimeoutError:
        # The agent run itself is NOT stopped — see limits.py. This request
        # simply stops waiting for it and tells the caller so, honestly,
        # instead of leaving their connection open with nothing to show for
        # it. The model-call/token deltas logged below may undercount this
        # request's real cost: the run can still be making calls on its
        # worker thread after we've already answered.
        status, outcome = 504, "timeout"
        error_detail = (
            f"The agent didn't finish within {config.REQUEST_TIMEOUT_SECONDS:g}s. "
            "It may still be running in the background, but this request has "
            "given up waiting on it. Try again, or with a narrower query."
        )
    except QuotaGuard as exc:
        status, outcome, error_detail = 503, "quota_exceeded", str(exc)
    except ModelUnavailable as exc:
        status, outcome, error_detail = 503, "model_unavailable", str(exc)
    except Exception as exc:  # noqa: BLE001 — a 500 with a sentence beats a hang
        status, outcome = 500, "server_error"
        error_detail = f"{type(exc).__name__}: {exc}"
        _logger.exception("unhandled error in /ask", extra={"fields": {
            "event": "ask_error", "request_id": request_id,
        }})

    # Outside the lock: building a response and logging are not work the next
    # caller should be waiting on.
    duration_ms = (time.monotonic() - started) * 1000
    calls_made = generate.call_count() - calls_before
    cache_hits_made = generate.cache_hit_count() - cache_hits_before
    tokens_after = generate.token_counts()
    prompt_tokens_made = tokens_after["prompt"] - tokens_before["prompt"]
    output_tokens_made = tokens_after["output"] - tokens_before["output"]

    log_event(
        _logger,
        event="ask",
        request_id=request_id,
        outcome=outcome,
        status=status,
        duration_ms=round(duration_ms, 1),
        waited_for_lock=waited_for_lock,
        query_chars=len(query),
        wardrobe_items=len(wardrobe.get("items", [])),
        model_calls=calls_made,
        cache_hits=cache_hits_made,
        prompt_tokens=prompt_tokens_made,
        output_tokens=output_tokens_made,
        relaxed=session["relaxed"] if session else [],
    )
    _metrics.record(
        outcome,
        duration_ms,
        model_calls=calls_made,
        cache_hits=cache_hits_made,
        prompt_tokens=prompt_tokens_made,
        output_tokens=output_tokens_made,
    )

    if error_detail is not None:
        return jsonify({"error": error_detail, "request_id": request_id}), status

    body = dict(session)
    body["request_id"] = request_id
    return jsonify(body), status


if __name__ == "__main__":
    # PORT and 0.0.0.0 are not decoration. A host tells you which port to
    # listen on through PORT, and it reaches your process from outside the
    # container — bind 127.0.0.1 and nothing but the container itself can
    # connect, which looks exactly like a broken app from the browser.
    port = int(os.getenv("PORT", "5000"))

    # Never on by default. Flask's debugger executes code typed into a browser,
    # and a deployed app with it on is a stranger's shell. Set AI201_DEBUG=1 in
    # your own terminal when you want the reloader.
    debug = os.getenv("AI201_DEBUG", "0") == "1"

    print(f"\nFitFindr is listening on http://localhost:{port}")
    print("It runs one request at a time — a second one waits for the first.")
    print("Ask it something (single quotes, as always):\n")
    print(
        f"  curl -X POST http://localhost:{port}/ask \\\n"
        f"    -H 'Content-Type: application/json' \\\n"
        f"    -d '{{\"query\": \"vintage graphic tee under $30\"}}'\n"
    )
    print("Ctrl-C to stop.\n")

    app.run(host="0.0.0.0", port=port, debug=debug)
