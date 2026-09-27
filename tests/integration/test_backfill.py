"""End-to-end backfill against a fake GitHub and the real (test) Postgres."""

import io
import zipfile
from datetime import UTC, datetime

import pytest
from sqlalchemy import text

from preflight_core.backfill.runner import BackfillConfig, Backfiller
from preflight_core.github.client import GitHubError
from preflight_core.labeling.labeler import relabel_all
from preflight_core.labeling.report import build_report

pytestmark = pytest.mark.integration

REPO = "acme/app"
SHA_A, SHA_B, SHA_C = "a" * 40, "b" * 40, "c" * 40


def run(id, sha, conclusion, *, attempt=1, event="pull_request", name="Tests", day="01", hour="10"):
    return {
        "id": id,
        "run_attempt": attempt,
        "workflow_id": 7,
        "name": name,
        "path": f".github/workflows/{name.lower()}.yml",
        "head_sha": sha,
        "head_branch": "x",
        "event": event,
        "pull_requests": [],
        "conclusion": conclusion,
        "created_at": f"2026-05-{day}T{hour}:00:00Z",
        "run_started_at": f"2026-05-{day}T{hour}:01:00Z",
        "updated_at": f"2026-05-{day}T{hour}:20:00Z",
    }


DAY1 = "2026-05-01T00:00:00Z..2026-05-01T23:59:59Z"
DAY2 = "2026-05-02T00:00:00Z..2026-05-02T23:59:59Z"
DAY2_AM = "2026-05-02T00:00:00Z..2026-05-02T11:59:59Z"
DAY2_PM = "2026-05-02T12:00:00Z..2026-05-02T23:59:59Z"

LOG = "\n".join(
    [
        "2026-05-01T10:02:00.0000000Z Traceback (most recent call last):",
        '2026-05-01T10:02:00.0000000Z   File "/w/app/src/auth.py", line 3, in login',
        "2026-05-01T10:02:00.0000000Z     return compute_secret(user)",
        "2026-05-01T10:02:00.0000000Z KeyError: 'user'",
    ]
).encode()


def junit_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "junit.xml",
            '<testsuite><testcase classname="t" name="a"><failure/></testcase>'
            '<testcase classname="t" name="b"/></testsuite>',
        )
    return buf.getvalue()


class FakeGitHub:
    def __init__(self):
        self.calls = 0
        self.log: list[str] = []
        self.runs = {
            DAY1: [
                run(100, SHA_A, "success", attempt=2),  # rerun: attempt 1 failed -> flaky later
                run(101, SHA_B, "failure", event="push", name="CI"),
                run(102, SHA_A, "success", name="Label PRs"),  # bot workflow, excluded
                run(103, SHA_A, "failure", event="schedule"),  # not a CI event, excluded
                run(104, SHA_A, None),  # still running, excluded
            ],
            DAY2_AM: [run(105, SHA_C, "success", day="02", hour="09")],
            DAY2_PM: [run(106, SHA_C, "failure", day="02", hour="15")],
        }

    def get_json(self, path, params=None):
        self.calls += 1
        self.log.append(path + (f"?{params['created']}" if params and "created" in params else ""))
        if path.endswith("/actions/runs"):
            created = params["created"]
            if created == DAY2:  # pretend the full day exceeds the 1000-result search cap
                return {"total_count": 1500, "workflow_runs": []}
            items = self.runs.get(created, []) if params.get("page", 1) == 1 else []
            return {"total_count": len(self.runs.get(created, [])), "workflow_runs": items}
        if path.endswith("/runs/100/attempts/1"):
            return run(100, SHA_A, "failure", attempt=1)
        if path.endswith(f"/commits/{SHA_A}") or path.endswith(f"/commits/{SHA_C}"):
            sha = path.rsplit("/", 1)[1]
            return {
                "sha": sha,
                "author": {"login": "dev1"},
                "commit": {"author": {"email": "d@x", "date": "2026-05-01T09:00:00Z"}},
                "parents": [{"sha": "p" * 40}],
                "stats": {"additions": 5, "deletions": 1},
                "files": [
                    {
                        "filename": "src/auth.py",
                        "additions": 5,
                        "deletions": 1,
                        "status": "modified",
                        "patch": "+SECRET_SOURCE",
                    }
                ],
            }
        if path.endswith(f"/commits/{SHA_B}"):
            raise GitHubError(422, path, "No commit found")
        if path.endswith("/runs/106/artifacts") or path.endswith("/runs/100/artifacts"):
            return {
                "artifacts": [
                    {"id": 9, "name": "junit-results", "expired": False},
                    {"id": 8, "name": "coverage-html", "expired": False},
                ]
            }
        if path.endswith("/artifacts"):
            return {"artifacts": []}
        raise AssertionError(f"unexpected GET {path}")

    def paginate(self, path, params=None, item_key=None):
        self.calls += 1
        self.log.append(path)
        jobs = {
            "/runs/100/attempts/1/jobs": [
                {
                    "id": 1000,
                    "name": "test",
                    "conclusion": "failure",
                    "steps": [{"name": "Run pytest", "conclusion": "failure"}],
                }
            ],
            "/runs/101/attempts/1/jobs": [
                {
                    "id": 1010,
                    "name": "build",
                    "conclusion": "failure",
                    "steps": [{"name": "Set up job", "conclusion": "failure"}],
                }
            ],
            "/runs/106/attempts/1/jobs": [
                {"id": 1060, "name": "test", "conclusion": "failure", "steps": []}
            ],
        }
        for suffix, items in jobs.items():
            if path.endswith(suffix):
                return iter(items)
        raise AssertionError(f"unexpected paginate {path}")

    def download(self, path):
        self.calls += 1
        self.log.append(path)
        if path.endswith("/jobs/1000/logs"):
            return LOG
        if path.endswith("/jobs/1010/logs"):
            return None  # expired
        if path.endswith("/jobs/1060/logs"):
            return b"FAILED tests/test_x.py::test_a - assert 1 == 2\nError: Process completed."
        if path.endswith("/artifacts/9/zip"):
            return junit_zip()
        raise AssertionError(f"unexpected download {path}")


class MemoryRawStore:
    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def put_bytes(self, key, data):
        self.objects[key] = data
        return key

    def get_bytes(self, key):
        return self.objects[key]

    def put_json(self, key, obj):
        import json

        return self.put_bytes(key, json.dumps(obj).encode())


def make(db, gh, raw):
    cfg = BackfillConfig(
        since=datetime(2026, 5, 1, tzinfo=UTC), until=datetime(2026, 5, 3, tzinfo=UTC)
    )
    return Backfiller(gh, db, raw, salt="s", cfg=cfg, now=lambda: datetime(2026, 5, 10, tzinfo=UTC))


def q(db, sql):
    with db.connect() as c:
        return c.execute(text(sql)).all()


def test_full_backfill(db):
    gh, raw = FakeGitHub(), MemoryRawStore()
    stats = make(db, gh, raw).run_repo(REPO, "main", junit=True)

    builds = q(
        db,
        "SELECT run_id, run_attempt, conclusion, rerun_of IS NOT NULL, log_available "
        "FROM builds ORDER BY run_id, run_attempt",
    )
    assert [(b[0], b[1], b[2]) for b in builds] == [
        (100, 1, "failure"),
        (100, 2, "success"),
        (101, 1, "failure"),
        (105, 1, "success"),
        (106, 1, "failure"),
    ]
    assert [b[3] for b in builds] == [False, True, False, False, False]  # attempt 2 -> attempt 1
    assert {b[0]: b[4] for b in builds if b[2] == "failure"} == {100: True, 101: False, 106: True}

    # window split happened because the full day was over the search cap
    assert f"/repos/{REPO}/actions/runs?{DAY2_AM}" in gh.log

    jobs = dict(q(db, "SELECT id, failed_step FROM jobs"))
    assert jobs == {1000: "Run pytest", 1010: "Set up job", 1060: None}
    log_text = raw.objects[f"logs/{REPO}/1000.txt"].decode()
    assert "KeyError: 'user'" in log_text and "compute_secret" not in log_text

    assert stats.commits == 2 and stats.commits_missing == 1
    stored = repr(q(db, "SELECT * FROM commits")) + repr(q(db, "SELECT * FROM file_changes"))
    assert "SECRET_SOURCE" not in stored and "dev1" not in stored

    tests = q(
        db,
        "SELECT b.run_id, t.test_id, t.status FROM test_results t "
        "JOIN builds b ON b.id = t.build_id ORDER BY 1, 2",
    )
    # run 100's artifacts are attached once (to one attempt), not to both attempts
    assert tests == [
        (100, "t::a", "failed"),
        (100, "t::b", "passed"),
        (106, "t::a", "failed"),
        (106, "t::b", "passed"),
    ]


def test_resume_does_not_refetch_or_duplicate(db):
    gh, raw = FakeGitHub(), MemoryRawStore()
    make(db, gh, raw).run_repo(REPO, "main", junit=True)
    first_calls = gh.calls
    make(db, gh, raw).run_repo(REPO, "main", junit=True)
    assert gh.calls == first_calls  # every stage resumed past its cursor
    assert q(db, "SELECT count(*) FROM builds")[0][0] == 5
    assert q(db, "SELECT count(*) FROM test_results")[0][0] == 4


def test_labels_and_report(db):
    raw = MemoryRawStore()
    make(db, FakeGitHub(), raw).run_repo(REPO, "main", junit=True)
    counts = relabel_all(db, raw)[REPO]
    assert dict(counts) == {"flaky": 1, "pass": 2, "infra": 1, "code_defect": 1}

    labels = q(
        db,
        "SELECT run_id, run_attempt, failure_type, label_source, label_reason "
        "FROM builds ORDER BY run_id, run_attempt",
    )
    assert labels == [
        (100, 1, "flaky", "rerun", "same sha/workflow passed later"),
        (100, 2, "pass", "conclusion", "success"),
        (101, 1, "infra", "step", "step:Set up job"),  # log expired -> step fallback
        (105, 1, "pass", "conclusion", "success"),
        (106, 1, "code_defect", "log", "pytest_failed"),  # earlier pass on same sha != flaky
    ]
    assert q(db, "SELECT failure_base_rate FROM repos")[0][0] == pytest.approx(1 / 3)

    report = build_report(db)
    assert "training rows (pass+code_defect): 3" in report
    assert "noisy failures removed         : 2 of 3 failures (66.7%)" in report


def test_runs_cursor_extends_backwards_and_forwards(db):
    def backfill(gh, since_day, until_day):
        cfg = BackfillConfig(
            since=datetime(2026, 5, since_day, tzinfo=UTC),
            until=datetime(2026, 5, until_day, tzinfo=UTC),
        )
        Backfiller(gh, db, MemoryRawStore(), salt="s", cfg=cfg).run_repo(
            REPO, "main", stages=("runs",)
        )

    def runs_queries(gh):
        return [p for p in gh.log if "/actions/runs?" in p]

    gh = FakeGitHub()
    backfill(gh, 2, 3)  # day 2 only
    assert all("2026-05-02" in p for p in runs_queries(gh))

    gh = FakeGitHub()
    backfill(gh, 1, 3)  # earlier since: fetches day 1 only
    assert runs_queries(gh) == [f"/repos/{REPO}/actions/runs?{DAY1}"]

    gh = FakeGitHub()
    backfill(gh, 1, 3)  # nothing new
    assert runs_queries(gh) == []
    assert q(db, "SELECT count(*) FROM builds")[0][0] == 5
    assert q(db, "SELECT cursor FROM backfill_state WHERE stage = 'runs'")[0][0] == {
        "from": "2026-05-01T00:00:00Z",
        "done_until": "2026-05-03T00:00:00Z",
    }
