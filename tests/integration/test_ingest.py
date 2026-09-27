from pathlib import Path

import fakeredis
import pytest
import redis
from fastapi.testclient import TestClient
from sqlalchemy import text

from apps.ingest.app import create_app
from apps.ingest.signature import sign

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).parents[1] / "fixtures" / "webhooks"
SECRET = "test-secret"
STREAM = "gh:events:test"


@pytest.fixture
def rds():
    return fakeredis.FakeRedis()


@pytest.fixture
def client(db, rds):
    return TestClient(create_app(db, rds, SECRET, STREAM))


def post(client, fixture, delivery, event=None, secret=SECRET, body=None):
    body = body if body is not None else (FIXTURES / fixture).read_bytes()
    return client.post(
        "/webhooks/github",
        content=body,
        headers={
            "X-GitHub-Event": event or fixture.split(".")[0],
            "X-GitHub-Delivery": delivery,
            "X-Hub-Signature-256": sign(secret, body),
        },
    )


def test_valid_webhook_is_enqueued(client, rds, db):
    r = post(client, "pull_request.opened.json", "d-1")
    assert (r.status_code, r.json()) == (202, {"status": "queued"})
    [(_, msg)] = rds.xrange(STREAM)
    assert msg[b"event"] == b"pull_request" and msg[b"action"] == b"opened"
    assert msg[b"repo"] == b"acme/app" and msg[b"delivery_id"] == b"d-1"
    with db.connect() as c:
        rows = c.execute(text("SELECT event, repo_full_name FROM webhook_deliveries")).all()
    assert rows == [("pull_request", "acme/app")]


def test_redelivery_is_deduplicated(client, rds):
    assert post(client, "workflow_run.completed.json", "d-2").status_code == 202
    r = post(client, "workflow_run.completed.json", "d-2")
    assert (r.status_code, r.json()) == (200, {"status": "duplicate"})
    assert rds.xlen(STREAM) == 1


def test_bad_signature_is_rejected_and_not_recorded(client, rds, db):
    assert post(client, "push.json", "d-3", secret="wrong").status_code == 401
    assert rds.xlen(STREAM) == 0
    with db.connect() as c:
        assert c.execute(text("SELECT count(*) FROM webhook_deliveries")).scalar() == 0


def test_unhandled_event_is_recorded_not_enqueued(client, rds):
    assert post(client, "issues.opened.json", "d-4").json() == {"status": "ignored"}
    assert rds.xlen(STREAM) == 0


def test_ping(client):
    r = post(client, "ping.json", "d-5")
    assert (r.status_code, r.json()) == (200, {"status": "pong"})


def test_non_json_body_is_400(client):
    assert post(client, "x", "d-6", event="push", body=b"not json").status_code == 400


def test_enqueue_failure_rolls_back_so_redelivery_succeeds(db, rds):
    class DownRedis:
        def xadd(self, *a, **kw):
            raise redis.ConnectionError("redis down")

    down = TestClient(create_app(db, DownRedis(), SECRET, STREAM), raise_server_exceptions=False)
    assert post(down, "push.json", "d-7").status_code == 500
    up = TestClient(create_app(db, rds, SECRET, STREAM))
    assert post(up, "push.json", "d-7").json() == {"status": "queued"}  # not "duplicate"


def test_feedback_button_events_are_enqueued(client, rds):
    body = (
        b'{"action": "requested_action", "check_run": {"id": 1},'
        b' "repository": {"full_name": "acme/app"}}'
    )
    r = post(client, "x", "d-8", event="check_run", body=body)
    assert r.json() == {"status": "queued"} and rds.xlen(STREAM) == 1
