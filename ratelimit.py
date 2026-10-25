"""
ratelimit.py - simple in-memory sliding-window rate limiting.

Two limits protect the public demo:
- per visitor (IP address), so one person can't spam the API
- global, so the total number of LLM calls stays within the provider's free quota,
  even if someone spoofs IP addresses

In-memory state is fine for a single server instance. With several instances
behind a load balancer you'd keep the counters in a shared store like Redis.
"""
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from config import RATE_LIMIT_GLOBAL, RATE_LIMIT_PER_IP


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window = window_seconds
        self.hits: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()  # endpoints run in a thread pool

    def hit(self, key: str) -> int | None:
        """Records a request. Returns seconds to wait if the limit is exceeded, else None."""
        now = time.monotonic()
        with self.lock:
            timestamps = self.hits[key]
            while timestamps and now - timestamps[0] >= self.window:
                timestamps.popleft()
            if len(timestamps) >= self.limit:
                return int(self.window - (now - timestamps[0])) + 1
            timestamps.append(now)
            if len(self.hits) > 10_000:  # forget visitors with no recent requests
                for k in [k for k, v in self.hits.items() if not v]:
                    del self.hits[k]
            return None


per_ip = SlidingWindowLimiter(RATE_LIMIT_PER_IP, window_seconds=60)
global_limit = SlidingWindowLimiter(RATE_LIMIT_GLOBAL, window_seconds=3600)


def client_ip(request: Request) -> str:
    # Behind a hosting proxy (e.g. Render) the real visitor IP is in X-Forwarded-For.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request) -> None:
    """FastAPI dependency: raises 429 Too Many Requests when a limit is hit."""
    wait = per_ip.hit(client_ip(request))
    if wait is not None:
        raise HTTPException(
            429, f"Too many questions. Please wait {wait} s and try again.",
            headers={"Retry-After": str(wait)},
        )
    wait = global_limit.hit("all")
    if wait is not None:
        raise HTTPException(
            429, "The demo has reached its hourly limit. Please try again later.",
            headers={"Retry-After": str(wait)},
        )
