"""Stream consumer: Redis Streams consumer group with retries and a dead-letter stream.

A message is acked only after its handler succeeds. Failures stay pending; after
IDLE_MS they are re-claimed and retried. After MAX_DELIVERIES attempts the message moves to
`<stream>:dlq` with the error, so one poison event never blocks the queue.

    python -m apps.worker.main
"""

import json
import logging
import os
import socket
import time

import httpx
import redis
from prometheus_client import start_http_server

from apps.worker.handlers import Handlers
from preflight_core.config import get_settings
from preflight_core.db.engine import get_engine
from preflight_core.github.app_auth import InstallationTokens, load_private_key
from preflight_core.labeling.log_classifier import LogClassifier
from preflight_core.logging import setup_logging
from preflight_core.observability.metrics import DLQ, EVENTS, QUEUE_LAG, QUEUE_PENDING
from preflight_core.online.store import OnlineStore
from preflight_core.storage.s3 import get_raw_store

log = logging.getLogger("worker")

GROUP = "workers"
MAX_DELIVERIES = 3
IDLE_MS = 60_000


class Consumer:
    def __init__(
        self, rds: redis.Redis, stream: str, handlers: Handlers, consumer: str | None = None
    ) -> None:
        self.r = rds
        self.stream = stream
        self.dlq = f"{stream}:dlq"
        self.h = handlers
        self.name = consumer or f"{socket.gethostname()}-{os.getpid()}"
        self.errors: dict[bytes, str] = {}
        try:
            self.r.xgroup_create(stream, GROUP, id="0", mkstream=True)
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    def _process(self, msg_id: bytes, fields: dict[bytes, bytes]) -> None:
        event = fields[b"event"].decode()
        ctx = {"delivery_id": fields.get(b"delivery_id", b"").decode(), "event": event}
        try:
            outcome = self.h.handle(event, json.loads(fields[b"payload"]))
            self.r.xack(self.stream, GROUP, msg_id)
            self.errors.pop(msg_id, None)
            EVENTS.labels(event, "ok").inc()
            log.info("processed", extra={**ctx, "outcome": outcome})
        except Exception as e:
            self.errors[msg_id] = repr(e)
            EVENTS.labels(event, "error").inc()
            log.exception("handler failed; will retry", extra=ctx)

    def reclaim(self) -> None:
        """Retry messages idle too long; dead-letter the ones that keep failing."""
        pending = [
            p
            for p in self.r.xpending_range(self.stream, GROUP, min="-", max="+", count=100)
            if p["time_since_delivered"] >= IDLE_MS
        ]
        for p in pending:
            msg_id = p["message_id"]
            if p["times_delivered"] >= MAX_DELIVERIES:
                entries = self.r.xrange(self.stream, msg_id, msg_id)
                if entries:
                    fields = entries[0][1]
                    self.r.xadd(
                        self.dlq,
                        {
                            **fields,
                            b"error": self.errors.get(msg_id, "unknown"),
                            b"original_id": msg_id,
                        },
                    )
                    DLQ.labels(fields.get(b"event", b"?").decode()).inc()
                    log.error("dead-lettered", extra={"id": msg_id.decode()})
                self.r.xack(self.stream, GROUP, msg_id)
                continue
            for mid, fields in self.r.xclaim(self.stream, GROUP, self.name, IDLE_MS, [msg_id]):
                self._process(mid, fields)

    def update_gauges(self) -> None:
        for g in self.r.xinfo_groups(self.stream):
            if g["name"] in (GROUP, GROUP.encode()):
                QUEUE_PENDING.set(g["pending"])
                QUEUE_LAG.set(g.get("lag") or 0)

    def run_once(self, block_ms: int = 5000) -> int:
        resp = self.r.xreadgroup(GROUP, self.name, {self.stream: ">"}, count=10, block=block_ms)
        n = 0
        for _, messages in resp or []:
            for msg_id, fields in messages:
                self._process(msg_id, fields)
                n += 1
        return n

    def run_forever(self) -> None:
        last_maint = 0.0
        while True:
            self.run_once()
            if time.monotonic() - last_maint > 15:
                self.reclaim()
                self.update_gauges()
                last_maint = time.monotonic()


def build_handlers() -> tuple[redis.Redis, str, Handlers]:
    s = get_settings()
    rds = redis.Redis.from_url(s.redis_url)
    try:
        key = load_private_key(
            s.github_app_private_key_path,
            s.github_app_private_key.get_secret_value() if s.github_app_private_key else None,
        )
        github_for = InstallationTokens(s.github_app_id, key, s.github_api_url).client
    except RuntimeError:
        log.warning("GitHub App not configured: events will retry, then dead-letter")

        def github_for(installation_id: int):
            raise RuntimeError("GitHub App not configured (see docs/github-app.md)")

    clf = LogClassifier.load("models/log_classifier.joblib")
    handlers = Handlers(
        engine=get_engine(),
        store=OnlineStore(rds),
        github_for=github_for,
        scorer=httpx.Client(
            base_url=s.scorer_url,
            headers={"Authorization": f"Bearer {s.scorer_api_token.get_secret_value()}"}
            if s.scorer_api_token
            else {},
        ),
        raw_store=get_raw_store(),
        salt=s.preflight_hash_salt.get_secret_value(),
        fail_open_timeout_s=s.fail_open_timeout_s,
        log_model=clf.predict if clf else None,
    )
    return rds, s.events_stream, handlers


def main() -> None:
    setup_logging()
    start_http_server(int(os.environ.get("METRICS_PORT", "9102")))
    rds, stream, handlers = build_handlers()
    Consumer(rds, stream, handlers).run_forever()


if __name__ == "__main__":
    main()
