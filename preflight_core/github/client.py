"""GitHub REST client built for long, resumable backfills.

- Token bucket caps our own request rate (avoids GitHub's secondary rate limits).
- Waits out the primary rate limit instead of failing, and slows down before hitting it.
- ETag conditional requests: a 304 costs no rate-limit budget and returns the cached body.
- Retries transport errors and 5xx with exponential backoff.
"""

import gzip
import logging
import time
from collections.abc import Callable, Iterator
from typing import Any, Protocol
from urllib.parse import urlencode

import httpx
from sqlalchemy import Engine, func, select
from sqlalchemy.dialects.postgresql import insert

from preflight_core.db.schema import http_cache

log = logging.getLogger(__name__)

API_VERSION = "2022-11-28"


class GitHubError(Exception):
    def __init__(self, status: int, url: str, body: str) -> None:
        super().__init__(f"GitHub {status} for {url}: {body[:200]}")
        self.status = status
        self.url = url


class ETagCache(Protocol):
    def get(self, key: str) -> tuple[str, bytes] | None: ...
    def set(self, key: str, etag: str, body: bytes) -> None: ...


class MemoryETagCache:
    def __init__(self) -> None:
        self._d: dict[str, tuple[str, bytes]] = {}

    def get(self, key: str) -> tuple[str, bytes] | None:
        return self._d.get(key)

    def set(self, key: str, etag: str, body: bytes) -> None:
        self._d[key] = (etag, body)


class TokenBucket:
    def __init__(
        self,
        rate: float,
        capacity: float,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self._now = now
        self._sleep = sleep
        self._last = now()

    def acquire(self) -> None:
        t = self._now()
        self.tokens = min(self.capacity, self.tokens + (t - self._last) * self.rate)
        self._last = t
        if self.tokens < 1:
            wait = (1 - self.tokens) / self.rate
            self._sleep(wait)
            self._last = self._now()
            self.tokens = 1
        self.tokens -= 1


class GitHubClient:
    def __init__(
        self,
        token: str,
        base_url: str = "https://api.github.com",
        cache: ETagCache | None = None,
        rps: float = 10.0,
        max_retries: int = 5,
        min_remaining: int = 50,
        sleep: Callable[[float], None] = time.sleep,
        wall_clock: Callable[[], float] = time.time,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": API_VERSION}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        # httpx drops the Authorization header on cross-origin redirects (log/artifact blobs).
        self._http = httpx.Client(
            base_url=base_url,
            headers=headers,
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=True,
            transport=transport,
        )
        self.cache = cache
        self.max_retries = max_retries
        self.min_remaining = min_remaining
        self._sleep = sleep
        self._clock = wall_clock
        self._bucket = TokenBucket(rps, capacity=rps, sleep=sleep)
        self.remaining: int | None = None
        self.reset_at: float | None = None
        self.calls = 0

    def close(self) -> None:
        self._http.close()

    # -- core request loop -------------------------------------------------

    def _wait_until_reset(self, reason: str) -> None:
        wait = max(1.0, (self.reset_at or self._clock()) - self._clock() + 1)
        log.warning("rate limit wait", extra={"reason": reason, "wait_s": round(wait)})
        self._sleep(wait)
        self.remaining = None  # unknown until the next response

    def _update_limits(self, resp: httpx.Response) -> None:
        if "x-ratelimit-remaining" in resp.headers:
            self.remaining = int(resp.headers["x-ratelimit-remaining"])
        if "x-ratelimit-reset" in resp.headers:
            self.reset_at = float(resp.headers["x-ratelimit-reset"])

    def _request(self, url: str, params: dict[str, Any] | None, headers: dict[str, str]):
        attempt = 0
        while True:
            if self.remaining is not None and self.remaining <= self.min_remaining:
                self._wait_until_reset("low remaining budget")
            self._bucket.acquire()
            try:
                resp = self._http.get(url, params=params, headers=headers)
                self.calls += 1
            except httpx.TransportError as e:
                attempt = self._backoff(attempt, url, repr(e))
                continue
            self._update_limits(resp)

            if resp.status_code in (403, 429):
                if resp.headers.get("x-ratelimit-remaining") == "0":
                    self._wait_until_reset("primary rate limit")
                    continue
                if "retry-after" in resp.headers:
                    self._sleep(float(resp.headers["retry-after"]))
                    continue
                if "secondary rate limit" in resp.text.lower():
                    self._sleep(60)
                    continue
            if resp.status_code >= 500:
                attempt = self._backoff(attempt, url, f"HTTP {resp.status_code}")
                continue
            return resp

    def _backoff(self, attempt: int, url: str, why: str) -> int:
        if attempt >= self.max_retries:
            raise GitHubError(0, url, f"gave up after {attempt} retries: {why}")
        wait = min(60.0, 2.0**attempt)
        log.warning("retrying", extra={"url": url, "why": why, "wait_s": wait})
        self._sleep(wait)
        return attempt + 1

    # -- public API --------------------------------------------------------

    def get(self, path: str, params: dict[str, Any] | None = None) -> httpx.Response:
        """GET with ETag caching; on a 304 the cached body is returned as a 200."""
        key = path + ("?" + urlencode(sorted(params.items())) if params else "")
        headers: dict[str, str] = {}
        cached = self.cache.get(key) if self.cache else None
        if cached:
            headers["If-None-Match"] = cached[0]
        resp = self._request(path, params, headers)
        if resp.status_code == 304 and cached:
            return httpx.Response(
                200, content=cached[1], headers=resp.headers, request=resp.request
            )
        if resp.status_code >= 400:
            raise GitHubError(resp.status_code, key, resp.text)
        if self.cache and "etag" in resp.headers:
            self.cache.set(key, resp.headers["etag"], resp.content)
        return resp

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self.get(path, params).json()

    def paginate(
        self, path: str, params: dict[str, Any] | None = None, item_key: str | None = None
    ) -> Iterator[dict[str, Any]]:
        """Yield items across pages, following Link: rel=next."""
        params = {"per_page": 100, **(params or {})}
        url: str | None = path
        while url:
            resp = self.get(url, params)
            data = resp.json()
            yield from (data[item_key] if item_key else data)
            url = resp.links.get("next", {}).get("url")
            params = None  # the next URL already carries the query string

    def send(self, method: str, path: str, json: dict[str, Any]) -> dict[str, Any]:
        """POST/PATCH (e.g. Check Runs). Retries 5xx/transport errors; no caching."""
        attempt = 0
        while True:
            self._bucket.acquire()
            try:
                resp = self._http.request(method, path, json=json)
                self.calls += 1
            except httpx.TransportError as e:
                attempt = self._backoff(attempt, path, repr(e))
                continue
            self._update_limits(resp)
            if resp.status_code >= 500:
                attempt = self._backoff(attempt, path, f"HTTP {resp.status_code}")
                continue
            if resp.status_code >= 400:
                raise GitHubError(resp.status_code, path, resp.text)
            return resp.json() if resp.content else {}

    def download(self, path: str) -> bytes | None:
        """Fetch a blob (job log, artifact zip). None if expired or missing (404/410)."""
        resp = self._request(path, None, {})
        if resp.status_code in (404, 410):
            return None
        if resp.status_code >= 400:
            raise GitHubError(resp.status_code, path, resp.text)
        return resp.content


class PgETagCache:
    """ETag cache persisted in Postgres so it survives restarts.

    Stores whole response bodies: never use it for endpoints that return source code
    (commits, compare, contents), because their `patch` fields would be persisted.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._t = http_cache

    def get(self, key: str) -> tuple[str, bytes] | None:
        with self._engine.connect() as c:
            row = c.execute(
                select(self._t.c.etag, self._t.c.body).where(self._t.c.url == key)
            ).first()
        return (row.etag, gzip.decompress(row.body)) if row else None

    def set(self, key: str, etag: str, body: bytes) -> None:
        stmt = insert(self._t).values(url=key, etag=etag, body=gzip.compress(body))
        stmt = stmt.on_conflict_do_update(
            index_elements=[self._t.c.url],
            set_={"etag": stmt.excluded.etag, "body": stmt.excluded.body, "fetched_at": func.now()},
        )
        with self._engine.begin() as c:
            c.execute(stmt)
