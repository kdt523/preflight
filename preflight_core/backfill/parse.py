"""Pure transforms from GitHub API payloads to table rows. No I/O, so all unit-testable."""

import hashlib
import hmac
import re
from datetime import datetime
from typing import Any

CI_EVENTS = {"push", "pull_request", "merge_group"}

# Workflows that are not CI for the code under change (bots, releases, docs, security scans).
NON_CI_WORKFLOW = re.compile(
    r"label|stale|triage|greet|welcome|release|publish|deploy|pages|\bdocs?\b|documentation"
    r"|codeql|scorecard|dependabot|backport|\block\b|\bcla\b|assign|milestone|comment"
    r"|benchmark|wheels?\b|upload|\bsync\b|bump|auto-?merge|notify|\bclose\b|size-?label",
    re.I,
)

TEST_PATH = re.compile(
    r"(^|/)(tests?|testing|__tests__|spec)/|(^|/)test_[^/]*\.py$|_tests?\.py$"
    r"|(^|/)conftest\.py$|\.(spec|test)\.[jt]sx?$",
    re.I,
)


def parse_ts(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def is_ci_workflow(name: str | None, path: str | None) -> bool:
    return not NON_CI_WORKFLOW.search(f"{name or ''} {path or ''}")


def is_ci_run(run: dict[str, Any]) -> bool:
    return run.get("event") in CI_EVENTS and is_ci_workflow(run.get("name"), run.get("path"))


def is_test_path(path: str) -> bool:
    return bool(TEST_PATH.search(path))


def hash_author(identity: str | None, salt: str) -> str | None:
    if not identity:
        return None
    return hmac.new(salt.encode(), identity.strip().lower().encode(), hashlib.sha256).hexdigest()


def run_to_build(run: dict[str, Any], repo_id: int) -> dict[str, Any]:
    prs = run.get("pull_requests") or []
    return {
        "repo_id": repo_id,
        "run_id": run["id"],
        "run_attempt": run.get("run_attempt") or 1,
        "workflow_id": run["workflow_id"],
        "workflow_name": run.get("name"),
        "sha": run["head_sha"],
        "head_branch": run.get("head_branch"),
        "event": run["event"],
        # Empty for PRs from forks; week 2 resolves those when building PR-level diffs.
        "pr_number": prs[0]["number"] if prs else None,
        "base_sha": prs[0]["base"]["sha"] if prs else None,
        "started_at": parse_ts(run.get("run_started_at") or run["created_at"]),
        "completed_at": parse_ts(run.get("updated_at")),
        "conclusion": run.get("conclusion"),
    }


MAX_COMMIT_FILES = 300  # GitHub returns at most 300 files on the commit endpoint's first page


def commit_to_rows(
    commit: dict[str, Any], repo_id: int, salt: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Commit payload -> (commits row, file_changes rows). `patch` (source code) is never read."""
    author = commit.get("author") or {}
    git_author = (commit.get("commit") or {}).get("author") or {}
    parents = commit.get("parents") or []
    files = commit.get("files") or []
    stats = commit.get("stats") or {}
    row = {
        "repo_id": repo_id,
        "sha": commit["sha"],
        "author_hash": hash_author(author.get("login") or git_author.get("email"), salt),
        "committed_at": parse_ts(git_author.get("date")),
        "parent_sha": parents[0]["sha"] if parents else None,
        "parent_count": len(parents),
        "additions": stats.get("additions", sum(f.get("additions", 0) for f in files)),
        "deletions": stats.get("deletions", sum(f.get("deletions", 0) for f in files)),
        "files_changed": len(files),
        "files_truncated": len(files) >= MAX_COMMIT_FILES,
    }
    file_rows = [
        {
            "repo_id": repo_id,
            "sha": commit["sha"],
            "path": f["filename"],
            "previous_path": f.get("previous_filename"),
            "added": f.get("additions", 0),
            "deleted": f.get("deletions", 0),
            "change_type": f.get("status", "modified"),
            "is_test": is_test_path(f["filename"]),
        }
        for f in files
    ]
    return row, file_rows


def job_to_row(job: dict[str, Any], build_id: int) -> dict[str, Any]:
    failed = next(
        (s["name"] for s in job.get("steps") or [] if s.get("conclusion") == "failure"), None
    )
    return {
        "id": job["id"],
        "build_id": build_id,
        "name": job["name"],
        "conclusion": job.get("conclusion"),
        "failed_step": failed,
        "runner_name": job.get("runner_name"),
        "started_at": parse_ts(job.get("started_at")),
        "completed_at": parse_ts(job.get("completed_at")),
    }


# -- logs ------------------------------------------------------------------

_TS_PREFIX = re.compile(r"^\d{4}-\d\d-\d\dT[\d:.]+Z ")
_PY_FRAME = re.compile(r'^\s*File ".*", line \d+')
_PYTEST_FAILURE_HEADER = re.compile(r"^_{3,} .+ _{3,}$")
_PYTEST_SECTION = re.compile(r"^={3,}")
REDACTED = "<source redacted>"


def log_tail(raw: bytes, n: int = 200) -> list[str]:
    lines = raw.decode("utf-8", errors="replace").splitlines()[-n:]
    return [_TS_PREFIX.sub("", ln) for ln in lines]


def redact_source(lines: list[str]) -> list[str]:
    """Drop lines that echo source code; keep errors, assertion diffs and runner messages.

    Redacts: the code line under a Python traceback frame, and source listings inside
    pytest failure blocks (lines starting with '>' or 4-space indent; 'E ' lines are kept).
    Best effort: the full raw log is never stored, only this redacted tail.
    """
    out: list[str] = []
    in_failure = after_frame = False
    for ln in lines:
        if _PYTEST_FAILURE_HEADER.match(ln):
            in_failure = True
        elif _PYTEST_SECTION.match(ln):
            in_failure = False
        is_src = (after_frame and ln.startswith("    ")) or (
            in_failure and (ln.startswith(">") or ln.startswith("    "))
        )
        after_frame = bool(_PY_FRAME.match(ln))
        if is_src:
            if not out or out[-1] != REDACTED:
                out.append(REDACTED)
        else:
            out.append(ln)
    return out
