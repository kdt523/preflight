from preflight_core.backfill.parse import (
    REDACTED,
    commit_to_rows,
    hash_author,
    is_ci_run,
    is_test_path,
    job_to_row,
    log_tail,
    redact_source,
    run_to_build,
)

RUN = {
    "id": 11,
    "run_attempt": 2,
    "workflow_id": 7,
    "name": "Tests",
    "path": ".github/workflows/tests.yml",
    "head_sha": "a" * 40,
    "head_branch": "fix-bug",
    "event": "pull_request",
    "pull_requests": [{"number": 42, "base": {"sha": "b" * 40}}],
    "created_at": "2026-05-01T10:00:00Z",
    "run_started_at": "2026-05-01T10:05:00Z",
    "updated_at": "2026-05-01T10:20:00Z",
    "conclusion": "failure",
}


def test_run_to_build():
    b = run_to_build(RUN, repo_id=1)
    assert (b["run_id"], b["run_attempt"], b["pr_number"], b["base_sha"]) == (11, 2, 42, "b" * 40)
    assert b["started_at"].isoformat() == "2026-05-01T10:05:00+00:00"


def test_fork_pr_has_no_pr_number():
    b = run_to_build({**RUN, "pull_requests": []}, repo_id=1)
    assert b["pr_number"] is None and b["base_sha"] is None


def test_ci_run_filter():
    assert is_ci_run(RUN)
    assert is_ci_run({**RUN, "event": "push", "name": "CI"})
    assert not is_ci_run({**RUN, "event": "schedule"})
    assert not is_ci_run({**RUN, "name": "Label PRs", "path": ".github/workflows/labeler.yml"})
    assert not is_ci_run({**RUN, "name": "Build docs", "path": ".github/workflows/docs.yml"})
    assert not is_ci_run({**RUN, "name": "CodeQL", "path": ".github/workflows/codeql.yml"})


def test_is_test_path():
    for p in [
        "tests/test_api.py",
        "pkg/tests/unit/x.py",
        "src/test_x.py",
        "a/b_test.py",
        "conftest.py",
        "web/app.spec.ts",
    ]:
        assert is_test_path(p), p
    for p in ["src/api.py", "testing_utils_notes.md", "docs/latest.rst", "contest.py"]:
        assert not is_test_path(p), p


def test_author_hash_is_salted_and_normalised():
    h = hash_author("Octocat", "salt")
    assert h == hash_author(" octocat ", "salt")
    assert h != hash_author("octocat", "other-salt")
    assert "octocat" not in h and len(h) == 64
    assert hash_author(None, "salt") is None


COMMIT = {
    "sha": "a" * 40,
    "author": {"login": "octocat"},
    "commit": {"author": {"email": "o@x.com", "date": "2026-05-01T09:00:00Z"}},
    "parents": [{"sha": "p" * 40}],
    "stats": {"additions": 12, "deletions": 3},
    "files": [
        {
            "filename": "src/app.py",
            "additions": 10,
            "deletions": 3,
            "status": "modified",
            "patch": "@@ -1 +1 @@\n-secret_code()\n+other_code()",
        },
        {
            "filename": "tests/test_app.py",
            "additions": 2,
            "deletions": 0,
            "status": "added",
            "patch": "+def test(): pass",
        },
    ],
}


def test_commit_rows_never_contain_source():
    row, files = commit_to_rows(COMMIT, repo_id=1, salt="s")
    assert (row["additions"], row["deletions"], row["files_changed"]) == (12, 3, 2)
    assert row["author_hash"] == hash_author("octocat", "s")
    assert not row["files_truncated"]
    assert [f["is_test"] for f in files] == [False, True]
    dumped = repr((row, files))
    assert "patch" not in dumped and "secret_code" not in dumped and "octocat" not in dumped


def test_commit_marks_truncated_file_list():
    files = [{"filename": f"f{i}.py", "additions": 1, "deletions": 0} for i in range(300)]
    row, _ = commit_to_rows({**COMMIT, "files": files}, repo_id=1, salt="s")
    assert row["files_truncated"]


def test_job_row_finds_failed_step():
    job = {
        "id": 5,
        "name": "test (3.12)",
        "conclusion": "failure",
        "runner_name": "GitHub Actions 3",
        "steps": [
            {"name": "Checkout", "conclusion": "success"},
            {"name": "Run pytest", "conclusion": "failure"},
            {"name": "Upload", "conclusion": "skipped"},
        ],
    }
    assert job_to_row(job, build_id=9)["failed_step"] == "Run pytest"


def test_log_tail_strips_timestamps_and_keeps_last_n():
    raw = "\n".join(f"2026-05-01T10:00:0{i % 10}.1234567Z line {i}" for i in range(300)).encode()
    tail = log_tail(raw, n=200)
    assert len(tail) == 200 and tail[0] == "line 100" and tail[-1] == "line 299"


def test_redact_source_keeps_errors_drops_code():
    lines = [
        "Traceback (most recent call last):",
        '  File "/home/runner/work/app/src/app.py", line 10, in handler',
        "    token = compute_secret(user)",
        "ValueError: bad user",
        "________________________ test_login ________________________",
        "    def test_login():",
        ">       assert login('x') == 200",
        "E       assert 500 == 200",
        "tests/test_app.py:12: AssertionError",
        "=================== short test summary info ===================",
        "    Error: The runner has received a shutdown signal.",
    ]
    out = redact_source(lines)
    text = "\n".join(out)
    assert (
        "compute_secret" not in text and "def test_login" not in text and "login('x')" not in text
    )
    assert "ValueError: bad user" in text and "E       assert 500 == 200" in text
    assert "tests/test_app.py:12: AssertionError" in text
    # indented runner message outside a pytest failure block survives
    assert "    Error: The runner has received a shutdown signal." in text
    assert out.count(REDACTED) == 2
