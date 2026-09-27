"""Nightly batch (SPEC §2): rebuild each repo's online state from Postgres history."""

import time

from sqlalchemy import Engine, text

from preflight_core.features.dataset import load_units
from preflight_core.features.state import build_state
from preflight_core.models.test_ranker import TestHistory, load_test_runs
from preflight_core.online.store import OnlineStore


def materialize_repo(
    engine: Engine,
    store: OnlineStore,
    repo_id: int,
    full_name: str,
    default_branch: str,
    until: float | None = None,
) -> dict[str, int]:
    until = until or time.time()
    units, _ = load_units(engine, repo_id, default_branch)
    state, started, completed = build_state(units, until)
    hist = TestHistory()
    for run in sorted(load_test_runs(engine, repo_id), key=lambda r: r.completed_at):
        if run.completed_at < until:
            hist.apply(run)
    with store.lock(full_name):
        store.replace_snapshot(full_name, state, started, completed)
        store.put_tests(full_name, hist)
        store.set_repo_config(full_name, default_branch=default_branch)
    return {
        "units": len(units),
        "started": len(started),
        "completed": len(completed),
        "test_runs": hist.runs,
    }


def materialize_all(engine: Engine, store: OnlineStore) -> dict[str, dict[str, int]]:
    with engine.connect() as c:
        repos = c.execute(
            text("SELECT id, full_name, default_branch, installation_id FROM repos")
        ).all()
    out = {}
    for r in repos:
        out[r.full_name] = materialize_repo(engine, store, r.id, r.full_name, r.default_branch)
        if r.installation_id:
            store.set_repo_config(r.full_name, installation_id=r.installation_id)
    return out
