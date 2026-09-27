"""Smoke test: compose services are up and the schema is migrated."""

import pytest
from sqlalchemy import inspect

from preflight_core.db.engine import get_engine
from preflight_core.storage.s3 import get_raw_store

pytestmark = pytest.mark.integration


def test_schema_has_week1_tables():
    tables = set(inspect(get_engine()).get_table_names())
    assert {
        "repos",
        "commits",
        "file_changes",
        "builds",
        "jobs",
        "test_results",
        "webhook_deliveries",
        "backfill_state",
        "http_cache",
    } <= tables


def test_raw_store_roundtrip():
    store = get_raw_store()
    store.put_json("smoke/test.json", {"ok": 1})
    assert store.get_json("smoke/test.json") == {"ok": 1}
