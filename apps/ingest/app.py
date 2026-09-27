"""Webhook ingest: verify, dedup, enqueue. Does no scoring work so it always answers fast."""

import json
import logging
from typing import Any

import redis
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import Engine, text
from sqlalchemy.dialects.postgresql import insert

from apps.ingest.signature import verify
from preflight_core.db.schema import webhook_deliveries
from preflight_core.observability.metrics import WEBHOOKS

log = logging.getLogger("ingest")

# Events the predict/learn loops consume. Others are acknowledged and recorded, not enqueued.
HANDLED_EVENTS = {
    "pull_request",
    "push",
    "workflow_run",
    "workflow_job",
    "check_suite",
    "check_run",  # requested_action = the Useful / Not useful buttons
    "installation",
    "installation_repositories",
}
STREAM_MAXLEN = 100_000


def create_app(engine: Engine, rds: redis.Redis, secret: str, stream: str) -> FastAPI:
    app = FastAPI(title="preflight-ingest")

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, str]:
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        rds.ping()
        return {"status": "ready"}

    @app.post("/webhooks/github", status_code=202)
    async def github_webhook(
        request: Request,
        response: Response,
        x_github_event: str = Header(...),
        x_github_delivery: str = Header(...),
        x_hub_signature_256: str | None = Header(None),
    ) -> dict[str, Any]:
        body = await request.body()
        ctx = {"delivery_id": x_github_delivery, "event": x_github_event}
        if not verify(secret, body, x_hub_signature_256):
            log.warning("bad signature", extra=ctx)
            WEBHOOKS.labels(x_github_event, "bad_signature").inc()
            raise HTTPException(401, "invalid signature")
        if x_github_event == "ping":
            response.status_code = 200
            return {"status": "pong"}
        try:
            payload = json.loads(body)
        except ValueError as e:
            raise HTTPException(400, "body is not JSON") from e

        action = payload.get("action")
        repo = (payload.get("repository") or {}).get("full_name")
        ctx |= {"action": action, "repo": repo}

        # One transaction: if the enqueue fails, the dedup row rolls back and GitHub's
        # redelivery is accepted instead of being dropped as a duplicate.
        def record_and_enqueue() -> tuple[str, str | None]:
            with engine.begin() as c:
                inserted = c.execute(
                    insert(webhook_deliveries)
                    .values(
                        delivery_id=x_github_delivery,
                        event=x_github_event,
                        action=action,
                        repo_full_name=repo,
                    )
                    .on_conflict_do_nothing()
                    .returning(webhook_deliveries.c.delivery_id)
                ).first()
                if inserted is None:
                    return "duplicate", None
                if x_github_event not in HANDLED_EVENTS:
                    return "ignored", None
                msg_id = rds.xadd(
                    stream,
                    {
                        "delivery_id": x_github_delivery,
                        "event": x_github_event,
                        "action": action or "",
                        "repo": repo or "",
                        "payload": body,
                    },
                    maxlen=STREAM_MAXLEN,
                    approximate=True,
                )
                return "queued", str(msg_id)

        status, msg_id = await run_in_threadpool(record_and_enqueue)
        WEBHOOKS.labels(x_github_event, status).inc()
        if status == "duplicate":
            response.status_code = 200
        log.info(status, extra={**ctx, "stream_id": msg_id})
        return {"status": status}

    return app
