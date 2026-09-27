"""Worker event handlers: the predict loop (PR/push -> Check Run) and the live learn loop
(workflow_run completed -> build, jobs, logs, label, online state).

Every handler is idempotent: GitHub redeliveries, stream retries and replays converge.
"""

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
from sqlalchemy import Engine, select, update
from sqlalchemy.dialects.postgresql import insert

from preflight_core.backfill import store as bstore
from preflight_core.backfill.parse import commit_to_rows, hash_author, is_ci_run, run_to_build
from preflight_core.backfill.runner import FAILED, BackfillConfig, Backfiller, Stats
from preflight_core.db.schema import builds, feedback, predictions, repos
from preflight_core.features.dataset import load_units
from preflight_core.github.client import GitHubClient
from preflight_core.labeling.labeler import relabel_repo
from preflight_core.models.test_ranker import TestRun, canonical_test_id
from preflight_core.observability.metrics import FAIL_OPEN
from preflight_core.online.store import OnlineStore
from preflight_core.serving.render import check_run_payload, skipped_payload

log = logging.getLogger("worker")

SCORE_ACTIONS = {"opened", "synchronize", "reopened", "ready_for_review"}
ZERO_SHA = "0" * 40


class Handlers:
    def __init__(
        self,
        engine: Engine,
        store: OnlineStore,
        github_for: Callable[[int], GitHubClient],
        scorer: httpx.Client,
        raw_store: Any,
        salt: str,
        fail_open_timeout_s: float = 2.0,
        log_model=None,
    ) -> None:
        self.engine = engine
        self.store = store
        self.github_for = github_for
        self.scorer = scorer
        self.raw = raw_store
        self.salt = salt
        self.timeout = fail_open_timeout_s
        self.log_model = log_model

    def handle(self, event: str, payload: dict[str, Any]) -> str:
        fn = {
            "pull_request": self.on_pull_request,
            "push": self.on_push,
            "workflow_run": self.on_workflow_run,
            "check_run": self.on_check_run,
            "installation": self.on_installation,
            "installation_repositories": self.on_installation,
        }.get(event)
        return fn(payload) if fn else "ignored"

    # -- repos -------------------------------------------------------------------

    def upsert_repo(self, repo: dict[str, Any], installation_id: int | None) -> int:
        stmt = insert(repos).values(
            full_name=repo["full_name"],
            default_branch=repo.get("default_branch") or "main",
            installation_id=installation_id,
        )
        set_ = {"installation_id": stmt.excluded.installation_id}
        if repo.get("default_branch"):
            set_["default_branch"] = stmt.excluded.default_branch
        with self.engine.begin() as c:
            rid = c.execute(
                stmt.on_conflict_do_update(index_elements=[repos.c.full_name], set_=set_).returning(
                    repos.c.id
                )
            ).scalar_one()
        self.store.set_repo_config(
            repo["full_name"],
            installation_id=installation_id,
            default_branch=repo.get("default_branch"),
        )
        return rid

    def on_installation(self, p: dict[str, Any]) -> str:
        inst = p["installation"]["id"]
        for r in p.get("repositories") or p.get("repositories_added") or []:
            self.upsert_repo({"full_name": r["full_name"]}, inst)
        return "repos registered"

    # -- predict loop --------------------------------------------------------------

    def on_pull_request(self, p: dict[str, Any]) -> str:
        if p.get("action") not in SCORE_ACTIONS or p["pull_request"].get("draft"):
            return "skipped action"
        pr = p["pull_request"]
        return self.score_and_check(
            p["repository"],
            p["installation"]["id"],
            pr["head"]["sha"],
            is_pr=True,
            is_default_branch=False,
            pr_number=pr["number"],
            base_sha=pr["base"]["sha"],
        )

    def on_push(self, p: dict[str, Any]) -> str:
        repo = p["repository"]
        branch = p["ref"].removeprefix("refs/heads/")
        # PR branches are scored via pull_request.synchronize; score default-branch pushes here.
        if p.get("deleted") or p["after"] == ZERO_SHA or branch != repo.get("default_branch"):
            return "skipped push"
        return self.score_and_check(
            repo, p["installation"]["id"], p["after"], is_pr=False, is_default_branch=True
        )

    def score_and_check(
        self,
        repo: dict[str, Any],
        installation_id: int,
        sha: str,
        *,
        is_pr: bool,
        is_default_branch: bool,
        pr_number: int | None = None,
        base_sha: str | None = None,
    ) -> str:
        full = repo["full_name"]
        repo_id = self.upsert_repo(repo, installation_id)
        gh = self.github_for(installation_id)
        try:
            commit = gh.get_json(f"/repos/{full}/commits/{sha}")
            row, files = commit_to_rows(commit, repo_id, self.salt)
            body = {
                "repo": full,
                "sha": sha,
                "is_pr": is_pr,
                "is_default_branch": is_default_branch,
                "base_sha": base_sha,
                "pr_number": pr_number,
                "author_hash": row["author_hash"],
                "truncated": row["files_truncated"],
                "rank_known_tests": True,
                "files": [
                    {
                        "path": f["path"],
                        "added": f["added"],
                        "deleted": f["deleted"],
                        "change_type": f["change_type"],
                    }
                    for f in files
                ],
            }
            r = self.scorer.post("/v1/score", json=body, timeout=self.timeout)
            r.raise_for_status()
            score = r.json()
        except httpx.TimeoutException:
            return self._fail_open(gh, full, repo_id, sha, pr_number, "scorer timeout")
        except Exception as e:  # fail-open on anything: CI must never wait on Preflight
            log.warning("scoring failed", extra={"repo": full, "sha": sha, "err": repr(e)})
            return self._fail_open(gh, full, repo_id, sha, pr_number, type(e).__name__)

        check = gh.send("POST", f"/repos/{full}/check-runs", check_run_payload(sha, score))
        if score.get("prediction_id"):
            with self.engine.begin() as c:
                c.execute(
                    update(predictions)
                    .where(predictions.c.id == score["prediction_id"])
                    .values(check_run_id=check.get("id"))
                )
        return f"checked {score['band']}"

    def _fail_open(
        self,
        gh: GitHubClient,
        full: str,
        repo_id: int,
        sha: str,
        pr_number: int | None,
        reason: str,
    ) -> str:
        FAIL_OPEN.labels(reason).inc()
        check = gh.send("POST", f"/repos/{full}/check-runs", skipped_payload(sha, reason))
        with self.engine.begin() as c:
            c.execute(
                insert(predictions).values(
                    repo_id=repo_id,
                    sha=sha,
                    pr_number=pr_number,
                    model_version="-",
                    model_key="-",
                    band="skipped",
                    ci_plan="full",
                    fail_open_reason=reason,
                    check_run_id=check.get("id"),
                )
            )
        return f"fail-open: {reason}"

    # -- feedback ----------------------------------------------------------------------

    def on_check_run(self, p: dict[str, Any]) -> str:
        if p.get("action") != "requested_action":
            return "ignored"
        ident = p["requested_action"]["identifier"]
        if ident not in {"useful", "not_useful"}:
            return "ignored"
        with self.engine.begin() as c:
            pid = c.execute(
                select(predictions.c.id).where(
                    predictions.c.check_run_id == p["check_run"]["id"],
                    predictions.c.shadow.is_(False),
                )
            ).scalar()
            if pid is None:
                return "unknown check run"
            stmt = insert(feedback).values(
                prediction_id=pid,
                user_hash=hash_author(p["sender"]["login"], self.salt),
                useful=ident == "useful",
            )
            c.execute(
                stmt.on_conflict_do_update(
                    index_elements=[feedback.c.prediction_id, feedback.c.user_hash],
                    set_={"useful": stmt.excluded.useful},
                )
            )
        return f"feedback {ident}"

    # -- learn loop ----------------------------------------------------------------------

    def on_workflow_run(self, p: dict[str, Any]) -> str:
        run = p["workflow_run"]
        if p.get("action") != "completed" or not is_ci_run(run) or not run.get("conclusion"):
            return "skipped run"
        repo = p["repository"]
        full = repo["full_name"]
        repo_id = self.upsert_repo(repo, p["installation"]["id"])
        gh = self.github_for(p["installation"]["id"])
        now = datetime.now(UTC)  # live runs are always inside log/artifact retention
        cfg = BackfillConfig(since=now, until=now, log_retention_days=36500)
        bf = Backfiller(gh, self.engine, self.raw, self.salt, cfg)
        stats = Stats()

        with self.engine.begin() as c:
            bstore.upsert_builds(c, [run_to_build(run, repo_id)])
            bstore.link_reruns(c, repo_id)
            b = c.execute(
                select(
                    builds.c.id,
                    builds.c.run_id,
                    builds.c.run_attempt,
                    builds.c.sha,
                    builds.c.started_at,
                ).where(
                    builds.c.repo_id == repo_id,
                    builds.c.run_id == run["id"],
                    builds.c.run_attempt == (run.get("run_attempt") or 1),
                )
            ).one()
        if run["conclusion"] in FAILED:
            bf.fetch_jobs(full, b, stats)
        bf.fetch_diff(repo_id, full, b.sha, stats)
        junit = bf.fetch_junit(full, b.run_id)
        if junit:
            with self.engine.begin() as c:
                bstore.insert_test_results(c, b.id, junit)
        relabel_repo(self.engine, self.raw, repo_id, self.log_model, sha=b.sha)
        self._update_online_state(
            gh, full, repo_id, repo.get("default_branch") or "main", b.sha, junit, run
        )
        return f"learned {run['conclusion']}"

    def _all_ci_runs_done(self, gh: GitHubClient, full: str, sha: str) -> bool:
        runs = gh.get_json(f"/repos/{full}/actions/runs", {"head_sha": sha, "per_page": 100})
        return all(
            r.get("status") == "completed" for r in runs.get("workflow_runs", []) if is_ci_run(r)
        )

    def _update_online_state(
        self,
        gh: GitHubClient,
        full: str,
        repo_id: int,
        default_branch: str,
        sha: str,
        junit: list[dict[str, Any]],
        run: dict[str, Any],
    ) -> None:
        units, _ = load_units(self.engine, repo_id, default_branch, sha=sha)
        if not units:
            return
        unit = units[0]
        done = unit.label == 1 or self._all_ci_runs_done(gh, full, sha)
        with self.store.lock(full):
            state = self.store.get_state(full)
            changed = False
            if not self.store.was_applied(full, "start", sha):
                state.apply_start(unit)
                self.store.mark_applied(full, "start", [sha])
                changed = True
            if done and unit.completed_at and not self.store.was_applied(full, "complete", sha):
                state.apply_completion(unit)
                self.store.mark_applied(full, "complete", [sha])
                changed = True
            if changed:
                self.store.put_state(full, state)
            if junit:
                hist = self.store.get_tests(full)
                hist.apply(
                    TestRun(
                        key=str(run["id"]),
                        as_of=unit.as_of,
                        completed_at=unit.completed_at or unit.as_of,
                        changed_files=[f.path for f in unit.files],
                        results={
                            canonical_test_id(r["test_id"]): (r["status"], r["duration_ms"])
                            for r in junit
                        },
                    )
                )
                self.store.put_tests(full, hist)
