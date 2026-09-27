"""Resumable, staged backfill of one repo.

Stages run in order; each keeps its own cursor in backfill_state and commits after every
unit of work, so killing the process loses at most one window / build.

  runs   workflow runs (CI events + CI workflows only), incl. earlier attempts of reruns
  jobs   jobs + failed step for failed builds; redacted log tail for failed jobs if still retained
  diffs  commit diff stats and paths (never patches) for every build SHA
  junit  JUnit XML from artifacts (only for repos that publish it)
"""

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import Engine, select

from preflight_core.backfill import store
from preflight_core.backfill.junit import junit_from_zip, merge_results
from preflight_core.backfill.parse import (
    commit_to_rows,
    is_ci_run,
    job_to_row,
    log_tail,
    redact_source,
    run_to_build,
)
from preflight_core.backfill.repo_selection import JUNIT_HINT
from preflight_core.db.schema import builds, commits
from preflight_core.github.client import GitHubClient, GitHubError

log = logging.getLogger(__name__)

FAILED = ("failure", "timed_out", "startup_failure")
SEARCH_CAP = 1000  # GitHub returns at most 1000 results for a filtered runs query
STAGES = ("runs", "jobs", "diffs", "junit")


@dataclass
class BackfillConfig:
    since: datetime
    until: datetime
    log_retention_days: int = 90
    artifact_retention_days: int = 90
    max_junit_runs: int = 1000
    min_window: timedelta = timedelta(hours=1)


@dataclass
class Stats:
    runs_seen: int = 0
    builds: int = 0
    jobs: int = 0
    logs: int = 0
    logs_expired: int = 0
    commits: int = 0
    commits_missing: int = 0
    junit_runs: int = 0
    notes: list[str] = field(default_factory=list)


class Backfiller:
    def __init__(
        self,
        gh: GitHubClient,
        engine: Engine,
        raw_store: Any,
        salt: str,
        cfg: BackfillConfig,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.gh, self.engine, self.raw, self.salt, self.cfg, self.now = (
            gh,
            engine,
            raw_store,
            salt,
            cfg,
            now,
        )

    def run_repo(
        self, full_name: str, default_branch: str, junit: bool = False, stages=STAGES
    ) -> Stats:
        stats = Stats()
        with self.engine.begin() as c:
            repo_id = store.upsert_repo(c, full_name, default_branch)
        ctx = {"repo": full_name}
        for stage in stages:
            if stage == "junit" and not junit:
                continue
            log.info("stage start", extra={**ctx, "stage": stage})
            getattr(self, f"_stage_{stage}")(repo_id, full_name, stats)
            log.info("stage done", extra={**ctx, "stage": stage, "calls": self.gh.calls})
        return stats

    # -- runs -----------------------------------------------------------------

    def _windows(self, start: datetime, end: datetime) -> Iterator[tuple[datetime, datetime]]:
        day = timedelta(days=1)
        t = start
        while t < end:
            yield t, min(t + day, end)
            t += day

    def _fetch_window(self, full_name: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
        """All runs created in [start, end); splits the window if it exceeds the search cap."""
        created = f"{_iso(start)}..{_iso(end - timedelta(seconds=1))}"
        path = f"/repos/{full_name}/actions/runs"
        first = self.gh.get_json(path, {"created": created, "per_page": 100, "page": 1})
        if first["total_count"] > SEARCH_CAP and end - start > self.cfg.min_window:
            mid = start + (end - start) / 2
            return self._fetch_window(full_name, start, mid) + self._fetch_window(
                full_name, mid, end
            )
        runs = list(first["workflow_runs"])
        page = 2
        while len(runs) < min(first["total_count"], SEARCH_CAP):
            batch = self.gh.get_json(path, {"created": created, "per_page": 100, "page": page})
            if not batch["workflow_runs"]:
                break
            runs.extend(batch["workflow_runs"])
            page += 1
        return runs

    def _stage_runs(self, repo_id: int, full_name: str, stats: Stats) -> None:
        """The cursor is the covered range [from, done_until). A later call with an earlier
        `since` extends it backwards (newest window first); a later `until` extends it forwards.
        Either way the covered range stays contiguous after every committed window."""
        with self.engine.begin() as c:
            cur = store.get_cursor(c, repo_id, "runs")
        since, until = self.cfg.since, self.cfg.until
        if not cur:
            cov_from = cov_to = since
        else:
            cov_to = _parse(cur["done_until"])
            cov_from = _parse(cur.get("from", cur["done_until"]))  # pre-range cursors lack "from"
            if since > cov_to:  # gap between old coverage and new range: restart coverage
                cov_from = cov_to = since
        backward = list(self._windows(since, cov_from))[::-1]
        forward = list(self._windows(max(cov_to, since), until))
        for w_start, w_end in backward + forward:
            self._ingest_window(repo_id, full_name, w_start, w_end, stats)
            if w_end <= cov_from:
                cov_from = w_start
            else:
                cov_to = w_end
            with self.engine.begin() as c:
                store.set_cursor(
                    c, repo_id, "runs", {"from": _iso(cov_from), "done_until": _iso(cov_to)}
                )

    def _ingest_window(
        self, repo_id: int, full_name: str, w_start: datetime, w_end: datetime, stats: Stats
    ) -> None:
        runs = self._fetch_window(full_name, w_start, w_end)
        stats.runs_seen += len(runs)
        ci = [r for r in runs if is_ci_run(r) and r.get("conclusion")]
        rows = [run_to_build(r, repo_id) for r in ci]
        for r in ci:  # the list endpoint only returns the latest attempt of a rerun run
            for attempt in range(1, (r.get("run_attempt") or 1)):
                prev = self.gh.get_json(
                    f"/repos/{full_name}/actions/runs/{r['id']}/attempts/{attempt}"
                )
                if prev.get("conclusion"):
                    rows.append(run_to_build(prev, repo_id))
        if ci:
            self.raw.put_json(f"raw/{full_name}/runs/{_iso(w_start)}.json", ci)
        with self.engine.begin() as c:
            store.upsert_builds(c, rows)
            store.link_reruns(c, repo_id)
        stats.builds += len(rows)

    # -- jobs + logs ----------------------------------------------------------

    def _builds_after(self, repo_id: int, last_id: int, failed_only: bool, limit: int = 200):
        q = (
            select(
                builds.c.id,
                builds.c.run_id,
                builds.c.run_attempt,
                builds.c.sha,
                builds.c.started_at,
            )
            .where(builds.c.repo_id == repo_id, builds.c.id > last_id)
            .order_by(builds.c.id)
            .limit(limit)
        )
        if failed_only:
            q = q.where(builds.c.conclusion.in_(FAILED))
        with self.engine.connect() as c:
            return c.execute(q).all()

    def fetch_jobs(self, full_name: str, b: Any, stats: Stats) -> None:
        """Jobs + failed step for one failed build, plus redacted log tails if retained."""
        log_cutoff = self.now() - timedelta(days=self.cfg.log_retention_days)
        job_rows = [
            job_to_row(j, b.id)
            for j in self.gh.paginate(
                f"/repos/{full_name}/actions/runs/{b.run_id}/attempts/{b.run_attempt}/jobs",
                item_key="jobs",
            )
        ]
        log_keys: dict[int, str] = {}
        logs_ok: bool | None = None
        if b.started_at >= log_cutoff:
            for j in job_rows:
                if j["conclusion"] not in FAILED:
                    continue
                blob = self.gh.download(f"/repos/{full_name}/actions/jobs/{j['id']}/logs")
                if blob is None:
                    stats.logs_expired += 1
                    logs_ok = logs_ok or False
                    continue
                key = f"logs/{full_name}/{j['id']}.txt"
                self.raw.put_bytes(key, "\n".join(redact_source(log_tail(blob))).encode())
                log_keys[j["id"]] = key
                logs_ok = True
                stats.logs += 1
        else:
            logs_ok = False
        with self.engine.begin() as c:
            store.upsert_jobs(c, job_rows)
            for job_id, key in log_keys.items():
                store.set_job_log(c, job_id, key)
            if logs_ok is not None:
                store.set_log_available(c, b.id, logs_ok)
        stats.jobs += len(job_rows)

    def _stage_jobs(self, repo_id: int, full_name: str, stats: Stats) -> None:
        with self.engine.begin() as c:
            last = store.get_cursor(c, repo_id, "jobs").get("last_build_id", 0)
        while batch := self._builds_after(repo_id, last, failed_only=True):
            for b in batch:
                self.fetch_jobs(full_name, b, stats)
                with self.engine.begin() as c:
                    store.set_cursor(c, repo_id, "jobs", {"last_build_id": b.id})
                last = b.id

    # -- diffs ----------------------------------------------------------------

    def fetch_diff(self, repo_id: int, full_name: str, sha: str, stats: Stats) -> bool:
        """Commit stats + paths (never patches). False if the commit is unreachable."""
        try:
            payload = self.gh.get_json(f"/repos/{full_name}/commits/{sha}")
        except GitHubError as e:
            if e.status in (404, 422):  # force-pushed away / unreachable
                stats.commits_missing += 1
                return False
            raise
        row, files = commit_to_rows(payload, repo_id, self.salt)
        with self.engine.begin() as c:
            store.insert_commit(c, row, files)
        stats.commits += 1
        return True

    def _stage_diffs(self, repo_id: int, full_name: str, stats: Stats) -> None:
        with self.engine.begin() as c:
            last = store.get_cursor(c, repo_id, "diffs").get("last_build_id", 0)
        while batch := self._builds_after(repo_id, last, failed_only=False, limit=500):
            shas = list(dict.fromkeys(b.sha for b in batch))
            with self.engine.connect() as c:
                have = set(
                    c.execute(
                        select(commits.c.sha).where(
                            commits.c.repo_id == repo_id, commits.c.sha.in_(shas)
                        )
                    ).scalars()
                )
            for sha in shas:
                if sha not in have:
                    self.fetch_diff(repo_id, full_name, sha, stats)
            last = batch[-1].id
            with self.engine.begin() as c:
                store.set_cursor(c, repo_id, "diffs", {"last_build_id": last})

    # -- junit ----------------------------------------------------------------

    def fetch_junit(self, full_name: str, run_id: int) -> list[dict[str, Any]]:
        arts = self.gh.get_json(f"/repos/{full_name}/actions/runs/{run_id}/artifacts")
        rows: list[dict[str, Any]] = []
        for a in arts.get("artifacts", []):
            if a.get("expired") or not JUNIT_HINT.search(a["name"]):
                continue
            blob = self.gh.download(f"/repos/{full_name}/actions/artifacts/{a['id']}/zip")
            if blob:
                try:
                    rows.extend(junit_from_zip(blob))
                except (ValueError, OSError) as e:
                    log.warning("bad artifact", extra={"artifact": a["id"], "err": str(e)})
        return merge_results(rows)

    def _stage_junit(self, repo_id: int, full_name: str, stats: Stats) -> None:
        with self.engine.begin() as c:
            cur = store.get_cursor(c, repo_id, "junit")
        last, done = cur.get("last_build_id", 0), cur.get("runs_done", 0)
        cutoff = self.now() - timedelta(days=self.cfg.artifact_retention_days)
        seen_runs: set[int] = set()  # artifacts belong to the run, not the attempt
        while done < self.cfg.max_junit_runs and (
            batch := self._builds_after(repo_id, last, failed_only=False)
        ):
            for b in batch:
                last = b.id
                if (
                    b.started_at < cutoff
                    or done >= self.cfg.max_junit_runs
                    or b.run_id in seen_runs
                ):
                    continue
                seen_runs.add(b.run_id)
                rows = self.fetch_junit(full_name, b.run_id)
                with self.engine.begin() as c:
                    store.insert_test_results(c, b.id, rows)
                if rows:
                    done += 1
                    stats.junit_runs += 1
            with self.engine.begin() as c:
                store.set_cursor(c, repo_id, "junit", {"last_build_id": last, "runs_done": done})


def _iso(t: datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))
