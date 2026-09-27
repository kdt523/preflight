"""Apply labeler rules to stored builds. Re-runnable: labels are recomputed from stored evidence,
so improving a rule never needs another GitHub fetch."""

import logging
from collections import Counter, defaultdict
from typing import Any

from sqlalchemy import Engine, bindparam, select, text, true, update

from preflight_core.db.schema import builds, jobs, repos
from preflight_core.labeling.rules import CODE, PASS, JobEvidence, LogModel, classify_build

log = logging.getLogger(__name__)

_LATER_SUCCESS = text(
    """
    SELECT b.id, b.conclusion,
           EXISTS (
             SELECT 1 FROM builds s
             WHERE s.repo_id = b.repo_id AND s.sha = b.sha AND s.workflow_id = b.workflow_id
               AND s.event = b.event AND s.conclusion = 'success'
               AND (s.started_at > b.started_at
                    OR (s.run_id = b.run_id AND s.run_attempt > b.run_attempt))
           ) AS later_success
    FROM builds b WHERE b.repo_id = :repo AND (CAST(:sha AS text) IS NULL OR b.sha = :sha)
    """
)


def relabel_repo(
    engine: Engine,
    raw_store: Any,
    repo_id: int,
    log_model: LogModel | None = None,
    sha: str | None = None,
) -> Counter:
    """Label a repo's builds (or only one commit's builds when `sha` is given)."""
    with engine.connect() as c:
        rows = c.execute(_LATER_SUCCESS, {"repo": repo_id, "sha": sha}).all()
        job_rows = c.execute(
            select(jobs.c.build_id, jobs.c.conclusion, jobs.c.failed_step, jobs.c.log_tail_key)
            .join(builds, builds.c.id == jobs.c.build_id)
            .where(builds.c.repo_id == repo_id, (builds.c.sha == sha) if sha else true())
        ).all()

    evidence: dict[int, list[JobEvidence]] = defaultdict(list)
    for j in job_rows:
        lines = None
        if j.log_tail_key:
            try:
                lines = raw_store.get_bytes(j.log_tail_key).decode().splitlines()
            except Exception as e:  # missing object -> fall back to the step name
                log.warning("log tail unreadable", extra={"key": j.log_tail_key, "err": str(e)})
        evidence[j.build_id].append(JobEvidence(j.conclusion, j.failed_step, lines))

    updates, counts = [], Counter()
    for r in rows:
        label = classify_build(r.conclusion, r.later_success, evidence.get(r.id, []), log_model)
        counts[label.failure_type] += 1
        updates.append(
            {"b_id": r.id, "ft": label.failure_type, "src": label.source, "why": label.reason}
        )

    stmt = (
        update(builds)
        .where(builds.c.id == bindparam("b_id"))
        .values(
            failure_type=bindparam("ft"),
            label_source=bindparam("src"),
            label_reason=bindparam("why"),
        )
    )
    labelled = counts[PASS] + counts[CODE]
    with engine.begin() as c:
        if updates:
            c.execute(stmt, updates)
        if sha is not None:
            return counts
        c.execute(
            update(repos)
            .where(repos.c.id == repo_id)
            .values(failure_base_rate=counts[CODE] / labelled if labelled else None)
        )
    return counts


def relabel_all(
    engine: Engine, raw_store: Any, log_model: LogModel | None = None
) -> dict[str, Counter]:
    with engine.connect() as c:
        repo_rows = c.execute(select(repos.c.id, repos.c.full_name)).all()
    return {r.full_name: relabel_repo(engine, raw_store, r.id, log_model) for r in repo_rows}
