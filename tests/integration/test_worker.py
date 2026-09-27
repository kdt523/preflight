"""Worker handlers against real Postgres, fake Redis, a fake GitHub and the real scorer app."""

import json
import time
from pathlib import Path

import fakeredis
import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from apps.scorer.app import create_app
from apps.worker.handlers import Handlers
from apps.worker.main import Consumer
from preflight_core.features.state import build_state
from preflight_core.online.store import OnlineStore
from preflight_core.serving.service import ScoringService
from tests.integration.test_backfill import LOG, MemoryRawStore

pytestmark = pytest.mark.integration

FIX = Path(__file__).parents[1] / "fixtures" / "webhooks"
SHA = "6dcb09b5b57875f334f61aebed695e2e4193db5e"


class FakeGH:
    def __init__(self):
        self.sent: list[tuple[str, str, dict]] = []
        self.runs_for_sha = [
            {
                "status": "completed",
                "event": "pull_request",
                "name": "Tests",
                "path": ".github/workflows/tests.yml",
            }
        ]

    def get_json(self, path, params=None):
        if path.endswith(f"/commits/{SHA}"):
            return {
                "sha": SHA,
                "author": {"login": "dev1"},
                "commit": {"author": {"email": "d@x", "date": "2026-05-01T09:59:00Z"}},
                "parents": [{"sha": "p" * 40}],
                "stats": {"additions": 40, "deletions": 12},
                "files": [
                    {
                        "filename": "src/auth.py",
                        "additions": 40,
                        "deletions": 12,
                        "status": "modified",
                        "patch": "+SECRET",
                    }
                ],
            }
        if path.endswith("/actions/runs") and params and params.get("head_sha"):
            return {"workflow_runs": self.runs_for_sha}
        if path.endswith("/artifacts"):
            return {"artifacts": []}
        raise AssertionError(path)

    def paginate(self, path, params=None, item_key=None):
        return iter(
            [
                {
                    "id": 77,
                    "name": "test",
                    "conclusion": "failure",
                    "steps": [{"name": "Run pytest", "conclusion": "failure"}],
                }
            ]
        )

    def download(self, path):
        return LOG + b"\nFAILED tests/test_auth.py::test_login - assert 1 == 2"

    def send(self, method, path, json):
        self.sent.append((method, path, json))
        return {"id": 5000 + len(self.sent)}


@pytest.fixture
def env(db, synth_bundle, synth_units):
    rds = fakeredis.FakeRedis()
    store = OnlineStore(rds)
    state, s, c = build_state(synth_units, until=time.time())
    store.replace_snapshot("acme/app", state, s, c)
    svc = ScoringService(store, lambda: synth_bundle, "salt")
    gh = FakeGH()
    raw = MemoryRawStore()
    h = Handlers(db, store, lambda inst: gh, TestClient(create_app(svc, db)), raw, "salt")
    return h, gh, store, rds, db


def payload(name):
    return json.loads((FIX / name).read_text())


def q(db, sql):
    with db.connect() as c:
        return c.execute(text(sql)).all()


def test_pr_gets_a_neutral_check_and_a_logged_prediction(env):
    h, gh, _, _, db = env
    out = h.handle("pull_request", payload("pull_request.opened.json"))
    assert out.startswith("checked")
    method, path, body = gh.sent[-1]
    assert (method, path) == ("POST", "/repos/acme/app/check-runs")
    assert body["conclusion"] == "neutral" and body["head_sha"] == SHA
    rows = q(db, "SELECT band, check_run_id, pr_number, shadow FROM predictions")
    assert rows[0][1] == 5001 and rows[0][2] == 42 and rows[0][3] is False
    stored = json.dumps([list(r) for r in q(db, "SELECT * FROM predictions")], default=str)
    assert "SECRET" not in stored and "dev1" not in stored


def test_scorer_timeout_fails_open(env, db):
    h, gh, *_ = env

    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    h.scorer = httpx.Client(base_url="http://scorer", transport=httpx.MockTransport(slow))
    assert (
        h.handle("pull_request", payload("pull_request.opened.json")) == "fail-open: scorer timeout"
    )
    assert gh.sent[-1][2]["conclusion"] == "skipped"
    assert q(db, "SELECT band, fail_open_reason FROM predictions") == [
        ("skipped", "scorer timeout")
    ]


def test_feedback_vote_is_recorded_once_per_user(env, db):
    h, *_ = env
    h.handle("pull_request", payload("pull_request.opened.json"))
    vote = {
        "action": "requested_action",
        "check_run": {"id": 5001},
        "requested_action": {"identifier": "not_useful"},
        "sender": {"login": "rev"},
    }
    assert h.handle("check_run", vote) == "feedback not_useful"
    vote["requested_action"]["identifier"] = "useful"
    h.handle("check_run", vote)  # changed their mind
    assert q(db, "SELECT useful FROM feedback") == [(True,)]


def test_non_default_push_is_not_scored(env):
    h, gh, *_ = env
    p = payload("push.json")
    p["ref"] = "refs/heads/feature"
    assert h.handle("push", p) == "skipped push" and not gh.sent


def test_workflow_run_completed_learns_and_updates_online_state(env, db):
    h, gh, store, *_ = env
    before = store.get_state("acme/app")
    p = payload("workflow_run.completed.json")
    assert h.handle("workflow_run", p) == "learned failure"
    assert q(db, "SELECT failure_type, label_source FROM builds") == [("code_defect", "log")]
    assert q(db, "SELECT failed_step FROM jobs") == [("Run pytest",)]
    assert store.was_applied("acme/app", "start", SHA)
    assert store.was_applied("acme/app", "complete", SHA)
    after = store.get_state("acme/app")
    assert after.repo_fails == before.repo_fails + 1
    assert after.files["src/auth.py"][1] >= 1  # the file's failure count went up

    h.handle("workflow_run", p)  # redelivery: nothing double-counted
    assert store.get_state("acme/app").repo_fails == after.repo_fails


def test_poison_message_goes_to_dlq(env, monkeypatch):
    import apps.worker.main as wm

    h, _, _, rds, _ = env
    monkeypatch.setattr(wm, "IDLE_MS", 0)
    h.handle = lambda event, p: (_ for _ in ()).throw(RuntimeError("boom"))
    rds.xadd("gh:events:t", {"event": "push", "delivery_id": "d1", "payload": "{}"})
    c = Consumer(rds, "gh:events:t", h, consumer="c1")
    c.run_once(block_ms=10)
    for _ in range(4):
        c.reclaim()
    dlq = rds.xrange("gh:events:t:dlq")
    assert len(dlq) == 1 and b"boom" in dlq[0][1][b"error"]
    assert rds.xpending("gh:events:t", "workers")["pending"] == 0
