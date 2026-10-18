"""Small in-process limits for the public research endpoint."""

from __future__ import annotations

from collections import OrderedDict, deque
from threading import Lock
from time import monotonic

from fastapi import HTTPException, Request

RESEARCH_REQUESTS_PER_MINUTE = 10
RESEARCH_RATE_WINDOW_SECONDS = 60.0
MAX_RESEARCH_BODY_BYTES = 16 * 1024


class PerClientRateLimiter:
    def __init__(self, requests: int, window_seconds: float, max_clients: int = 1024) -> None:
        self.requests = requests
        self.window_seconds = window_seconds
        self.max_clients = max_clients
        self._requests: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = Lock()

    def check(self, client_key: str) -> None:
        now = monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            for key in list(self._requests):
                timestamps = self._requests[key]
                while timestamps and timestamps[0] <= cutoff:
                    timestamps.popleft()
                if not timestamps:
                    del self._requests[key]

            timestamps = self._requests.setdefault(client_key, deque())
            self._requests.move_to_end(client_key)
            if len(timestamps) >= self.requests:
                raise HTTPException(
                    status_code=429,
                    detail="Research request limit reached. Please wait before trying again.",
                )
            timestamps.append(now)
            if len(self._requests) > self.max_clients:
                self._requests.popitem(last=False)


_research_rate_limiter = PerClientRateLimiter(
    RESEARCH_REQUESTS_PER_MINUTE,
    RESEARCH_RATE_WINDOW_SECONDS,
)


def enforce_research_rate_limit(request: Request) -> None:
    client_key = request.client.host if request.client is not None else "unknown"
    _research_rate_limiter.check(client_key)