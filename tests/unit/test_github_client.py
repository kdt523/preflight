import httpx
import pytest
import respx

from preflight_core.github.client import GitHubClient, GitHubError, MemoryETagCache, TokenBucket

API = "https://api.github.com"


class FakeClock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


@pytest.fixture
def clock():
    return FakeClock()


def make_client(clock, **kw):
    return GitHubClient(
        "tok", cache=kw.pop("cache", None), sleep=clock.sleep, wall_clock=clock.now, rps=1000, **kw
    )


@respx.mock
def test_sends_auth_and_api_version(clock):
    route = respx.get(f"{API}/repos/a/b").mock(return_value=httpx.Response(200, json={"id": 1}))
    assert make_client(clock).get_json("/repos/a/b") == {"id": 1}
    req = route.calls.last.request
    assert req.headers["authorization"] == "Bearer tok"
    assert req.headers["x-github-api-version"] == "2022-11-28"


@respx.mock
def test_etag_304_returns_cached_body(clock):
    route = respx.get(f"{API}/repos/a/b").mock(
        side_effect=[
            httpx.Response(200, json={"v": 1}, headers={"etag": 'W/"abc"'}),
            httpx.Response(304),
        ]
    )
    c = make_client(clock, cache=MemoryETagCache())
    assert c.get_json("/repos/a/b") == {"v": 1}
    assert c.get_json("/repos/a/b") == {"v": 1}
    assert route.calls[1].request.headers["if-none-match"] == 'W/"abc"'


@respx.mock
def test_waits_for_primary_rate_limit_reset(clock):
    reset = clock.t + 120
    respx.get(f"{API}/x").mock(
        side_effect=[
            httpx.Response(
                403,
                text="API rate limit exceeded",
                headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(int(reset))},
            ),
            httpx.Response(200, json={"ok": True}, headers={"x-ratelimit-remaining": "4999"}),
        ]
    )
    assert make_client(clock).get_json("/x") == {"ok": True}
    assert sum(clock.sleeps) >= 120


@respx.mock
def test_honours_retry_after_on_secondary_limit(clock):
    respx.get(f"{API}/x").mock(
        side_effect=[
            httpx.Response(429, headers={"retry-after": "7"}),
            httpx.Response(200, json={}),
        ]
    )
    make_client(clock).get_json("/x")
    assert 7 in clock.sleeps


@respx.mock
def test_slows_down_before_budget_runs_out(clock):
    reset = clock.t + 300
    respx.get(f"{API}/x").mock(
        return_value=httpx.Response(
            200,
            json={},
            headers={"x-ratelimit-remaining": "10", "x-ratelimit-reset": str(int(reset))},
        )
    )
    c = make_client(clock, min_remaining=50)
    c.get_json("/x")
    assert clock.sleeps == []
    c.get_json("/x")  # budget now known to be low -> waits for reset first
    assert sum(clock.sleeps) >= 300


@respx.mock
def test_retries_5xx_then_succeeds(clock):
    respx.get(f"{API}/x").mock(
        side_effect=[httpx.Response(502), httpx.Response(503), httpx.Response(200, json={})]
    )
    make_client(clock).get_json("/x")
    assert clock.sleeps == [1.0, 2.0]


@respx.mock
def test_retries_transport_errors_then_gives_up(clock):
    respx.get(f"{API}/x").mock(side_effect=httpx.ConnectError("boom"))
    with pytest.raises(GitHubError, match="gave up"):
        make_client(clock, max_retries=2).get_json("/x")


@respx.mock
def test_4xx_raises_without_retry(clock):
    route = respx.get(f"{API}/x").mock(return_value=httpx.Response(404, text="Not Found"))
    with pytest.raises(GitHubError) as e:
        make_client(clock).get_json("/x")
    assert e.value.status == 404
    assert route.call_count == 1


@respx.mock
def test_paginate_follows_link_header(clock):
    respx.get(f"{API}/repos/a/b/actions/runs", params={"page": "2"}).mock(
        return_value=httpx.Response(200, json={"workflow_runs": [{"id": 3}]})
    )
    respx.get(f"{API}/repos/a/b/actions/runs").mock(
        return_value=httpx.Response(
            200,
            json={"workflow_runs": [{"id": 1}, {"id": 2}]},
            headers={"link": f'<{API}/repos/a/b/actions/runs?per_page=100&page=2>; rel="next"'},
        )
    )
    ids = [
        r["id"]
        for r in make_client(clock).paginate("/repos/a/b/actions/runs", item_key="workflow_runs")
    ]
    assert ids == [1, 2, 3]


@respx.mock
def test_download_expired_log_returns_none(clock):
    respx.get(f"{API}/repos/a/b/actions/jobs/9/logs").mock(return_value=httpx.Response(410))
    assert make_client(clock).download("/repos/a/b/actions/jobs/9/logs") is None


@respx.mock
def test_download_follows_redirect_without_leaking_token(clock):
    respx.get(f"{API}/repos/a/b/actions/jobs/9/logs").mock(
        return_value=httpx.Response(302, headers={"location": "https://blob.example/log.txt"})
    )
    blob = respx.get("https://blob.example/log.txt").mock(
        return_value=httpx.Response(200, content=b"line1\nline2\n")
    )
    assert make_client(clock).download("/repos/a/b/actions/jobs/9/logs") == b"line1\nline2\n"
    assert "authorization" not in blob.calls.last.request.headers


def test_token_bucket_throttles_to_rate(clock):
    bucket = TokenBucket(rate=2, capacity=2, now=clock.now, sleep=clock.sleep)
    for _ in range(6):
        bucket.acquire()
    # 2 free from capacity, then 4 more at 2/s = 2 seconds
    assert sum(clock.sleeps) == pytest.approx(2.0)
