from alembic import context
from sqlalchemy import create_engine

from preflight_core.config import get_settings
from preflight_core.db.schema import metadata


def run_migrations_online() -> None:
    url = context.config.get_main_option("sqlalchemy.url") or get_settings().database_url
    engine = create_engine(url)
    with engine.connect() as conn:
        context.configure(connection=conn, target_metadata=metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
