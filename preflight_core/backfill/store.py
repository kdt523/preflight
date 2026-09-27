"""Idempotent Postgres writes for the backfill. Every insert is an upsert, so re-running is safe."""

from typing import Any

from sqlalchemy import Connection, select, text, update
from sqlalchemy.dialects.postgresql import insert

from preflight_core.db.schema import (
    backfill_state,
    builds,
    commits,
    file_changes,
    jobs,
    repos,
    test_results,
)


def upsert_repo(conn: Connection, full_name: str, default_branch: str) -> int:
    stmt = insert(repos).values(full_name=full_name, default_branch=default_branch)
    stmt = stmt.on_conflict_do_update(
        index_elements=[repos.c.full_name], set_={"default_branch": default_branch}
    ).returning(repos.c.id)
    return conn.execute(stmt).scalar_one()


def upsert_builds(conn: Connection, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    stmt = insert(builds).values(rows)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_builds_run_attempt",
        set_={
            "conclusion": stmt.excluded.conclusion,
            "completed_at": stmt.excluded.completed_at,
            "pr_number": stmt.excluded.pr_number,
            "base_sha": stmt.excluded.base_sha,
        },
    )
    conn.execute(stmt)


def link_reruns(conn: Connection, repo_id: int) -> None:
    """builds.rerun_of -> the previous attempt of the same run."""
    conn.execute(
        text(
            """
            UPDATE builds b SET rerun_of = p.id
            FROM builds p
            WHERE b.repo_id = :repo AND p.repo_id = b.repo_id AND p.run_id = b.run_id
              AND p.run_attempt = b.run_attempt - 1 AND b.rerun_of IS DISTINCT FROM p.id
            """
        ),
        {"repo": repo_id},
    )


def insert_commit(conn: Connection, row: dict[str, Any], files: list[dict[str, Any]]) -> None:
    conn.execute(insert(commits).values(row).on_conflict_do_nothing())
    if files:
        conn.execute(insert(file_changes).values(files).on_conflict_do_nothing())


def upsert_jobs(conn: Connection, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    stmt = insert(jobs).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=[jobs.c.id],
        set_={"conclusion": stmt.excluded.conclusion, "failed_step": stmt.excluded.failed_step},
    )
    conn.execute(stmt)


def set_job_log(conn: Connection, job_id: int, key: str | None) -> None:
    conn.execute(update(jobs).where(jobs.c.id == job_id).values(log_tail_key=key))


def set_log_available(conn: Connection, build_id: int, available: bool) -> None:
    conn.execute(update(builds).where(builds.c.id == build_id).values(log_available=available))


def insert_test_results(conn: Connection, build_id: int, rows: list[dict[str, Any]]) -> None:
    if rows:
        conn.execute(
            insert(test_results)
            .values([{**r, "build_id": build_id} for r in rows])
            .on_conflict_do_nothing()
        )


def get_cursor(conn: Connection, repo_id: int, stage: str) -> dict[str, Any]:
    row = conn.execute(
        select(backfill_state.c.cursor).where(
            backfill_state.c.repo_id == repo_id, backfill_state.c.stage == stage
        )
    ).first()
    return row.cursor if row else {}


def set_cursor(conn: Connection, repo_id: int, stage: str, cursor: dict[str, Any]) -> None:
    stmt = insert(backfill_state).values(repo_id=repo_id, stage=stage, cursor=cursor)
    stmt = stmt.on_conflict_do_update(
        index_elements=[backfill_state.c.repo_id, backfill_state.c.stage],
        set_={"cursor": stmt.excluded.cursor, "updated_at": text("now()")},
    )
    conn.execute(stmt)
