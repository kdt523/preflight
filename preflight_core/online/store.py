"""Redis online store: per-repo point-in-time state for serving.

Keys (prefix pf:)
  state:{repo}              zlib(JSON) HistoryState          build-model features
  tests:{repo}              zlib(JSON) TestHistory           per-test ranker features
  applied:{repo}:start      SET of unit keys whose start event was applied
  applied:{repo}:complete   SET of unit keys whose completion event was applied
  repo:{repo}               HASH config: default_branch, installation_id, blame_free

The nightly materializer rebuilds state from Postgres; the worker applies new events in
between. Writers hold a per-repo lock so two workers never lose each other's update.
"""

import json
import zlib
from contextlib import contextmanager, suppress
from typing import Any

import redis

from preflight_core.features.state import HistoryState
from preflight_core.models.test_ranker import TestHistory


def _pack(d: dict[str, Any]) -> bytes:
    return zlib.compress(json.dumps(d, separators=(",", ":")).encode(), 6)


def _unpack(b: bytes) -> dict[str, Any]:
    return json.loads(zlib.decompress(b))


class OnlineStore:
    def __init__(self, rds: redis.Redis, prefix: str = "pf") -> None:
        self.r = rds
        self.p = prefix

    def _k(self, *parts: str) -> str:
        return ":".join([self.p, *parts])

    # state ------------------------------------------------------------------------
    def get_state(self, repo: str) -> HistoryState:
        raw = self.r.get(self._k("state", repo))
        return HistoryState.from_dict(_unpack(raw)) if raw else HistoryState()

    def put_state(self, repo: str, state: HistoryState) -> None:
        self.r.set(self._k("state", repo), _pack(state.to_dict()))

    def get_tests(self, repo: str) -> TestHistory:
        raw = self.r.get(self._k("tests", repo))
        return TestHistory.from_dict(_unpack(raw)) if raw else TestHistory()

    def put_tests(self, repo: str, hist: TestHistory) -> None:
        self.r.set(self._k("tests", repo), _pack(hist.to_dict()))

    # idempotent event application -------------------------------------------------
    def was_applied(self, repo: str, kind: str, key: str) -> bool:
        return bool(self.r.sismember(self._k("applied", repo, kind), key))

    def mark_applied(self, repo: str, kind: str, keys: set[str] | list[str]) -> None:
        if keys:
            self.r.sadd(self._k("applied", repo, kind), *keys)

    def replace_snapshot(
        self, repo: str, state: HistoryState, started: set[str], completed: set[str]
    ) -> None:
        pipe = self.r.pipeline(transaction=True)
        pipe.set(self._k("state", repo), _pack(state.to_dict()))
        for kind, keys in (("start", started), ("complete", completed)):
            k = self._k("applied", repo, kind)
            pipe.delete(k)
            if keys:
                pipe.sadd(k, *keys)
        pipe.execute()

    @contextmanager
    def lock(self, repo: str, timeout: int = 30):
        lk = self.r.lock(self._k("lock", repo), timeout=timeout, blocking_timeout=timeout)
        if not lk.acquire():
            raise TimeoutError(f"could not lock online state for {repo}")
        try:
            yield
        finally:
            with suppress(redis.exceptions.LockError):  # lock expired while we held it
                lk.release()

    # repo config ------------------------------------------------------------------
    def repo_config(self, repo: str) -> dict[str, Any]:
        raw = {k.decode(): v.decode() for k, v in self.r.hgetall(self._k("repo", repo)).items()}
        return {
            "default_branch": raw.get("default_branch", "main"),
            "installation_id": int(raw["installation_id"]) if raw.get("installation_id") else None,
            "blame_free": raw.get("blame_free") == "1",
        }

    def set_repo_config(self, repo: str, **cfg: Any) -> None:
        vals = {
            k: ("1" if v is True else "0" if v is False else str(v))
            for k, v in cfg.items()
            if v is not None
        }
        if vals:
            self.r.hset(self._k("repo", repo), mapping=vals)
