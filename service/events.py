"""Thread-safe Server-Sent Events hub for the service.

Pushes state changes to connected browsers so the frontend never has to poll.
Jobs run in worker threads while the SSE stream lives on the FastAPI asyncio
loop, so the bridge uses plain ``queue.Queue`` objects guarded by a lock rather
than loop-bound scheduling.

Public surface:

- ``subscribe()`` / ``unsubscribe(q)``: open / close a per-client queue.
- ``broadcast(payload)``: enqueue an event to every subscriber (safe to call
  from any thread; each ``payload`` must carry a ``"type"`` used as the SSE
  event name).
- ``frames(q)``: async iterator yielding well-formed SSE frames for one client,
  with a short heartbeat while idle and clean shutdown on cancellation.
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading

_subs: list[queue.Queue] = []
_subs_lock = threading.Lock()

HEARTBEAT_SECONDS = 1.0
# An SSE connection is only kept alive by traffic. While no events flow,
# emit an SSE comment every PING_EVERY idle ticks so browsers and any proxy
# in between never time the connection out (EventSource ignores comments).
PING_EVERY = 15
# Hard cap on queued events per client: a slow/blocked connection drains only
# the latest events once it resumes, never an unbounded backlog.
_MAX_QUEUED = 100


def subscribe() -> queue.Queue:
    q: queue.Queue = queue.Queue(maxsize=_MAX_QUEUED)
    with _subs_lock:
        _subs.append(q)
    return q


def unsubscribe(q: queue.Queue) -> None:
    with _subs_lock:
        if q in _subs:
            _subs.remove(q)


def broadcast(payload: dict) -> None:
    """Fan out an event to every subscriber. ``payload`` is mutated by the
    caller (it is the source of the SSE event name), so a copy is queued."""
    data = dict(payload)
    with _subs_lock:
        subs = list(_subs)
    for q in subs:
        try:
            q.put_nowait(data)
        except queue.Full:
            try:
                q.get_nowait()  # drop the oldest so the newest still lands
                q.put_nowait(data)
            except queue.Empty:
                pass


def _format(payload: dict) -> str:
    etype = str(payload.get("type", "message"))
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {etype}\ndata: {data}\n\n"


async def frames(q: queue.Queue):
    """Async generator of SSE frames for one subscriber queue.

    Yields an event as soon as it is queued; otherwise sleeps a short
    heartbeat (an SSE connection is only kept alive by traffic, so the idle
    interval prevents proxies from timing it out). Ends when the caller
    cancels the generator (client disconnect).
    """
    while True:
        try:
            payload = q.get_nowait()
        except queue.Empty:
            for _ in range(PING_EVERY):
                await asyncio.sleep(HEARTBEAT_SECONDS)
                try:
                    payload = q.get_nowait()
                    break
                except queue.Empty:
                    payload = None
            if payload is None:
                yield ": ping\n\n"  # keep-alive comment, ignored by EventSource
                continue
        yield _format(payload)
        if q.qsize() > 0:
            # Drain any events that arrived while we were still in the loop so
            # the burst reaches the client in one yield cycle.
            continue
        await asyncio.sleep(0.0)
