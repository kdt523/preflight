import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from preflight_core.config import get_settings

TEST_DB = "preflight_test"


@pytest.fixture(scope="session")
def engine():
    """Fresh, fully migrated test database on the compose Postgres."""
    url = make_url(get_settings().database_url)
    admin = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)"))
        c.execute(text(f"CREATE DATABASE {TEST_DB}"))
    admin.dispose()
    test_url = url.set(database=TEST_DB).render_as_string(hide_password=False)
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", test_url)
    command.upgrade(cfg, "head")
    eng = create_engine(test_url)
    yield eng
    eng.dispose()


@pytest.fixture
def db(engine):
    """Empty tables before each test."""
    with engine.begin() as c:
        c.execute(
            text(
                "TRUNCATE feedback, predictions, model_runs, features, repos, commits, "
                "file_changes, builds, jobs, test_results, "
                "webhook_deliveries, backfill_state, http_cache RESTART IDENTITY CASCADE"
            )
        )
    return engine
