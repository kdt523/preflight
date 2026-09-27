"""Create an extra database on the same Postgres server if missing (used for MLflow).

python scripts/ensure_db.py mlflow
"""

import sys

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from preflight_core.config import get_settings

if __name__ == "__main__":
    name = sys.argv[1]
    if not name.isidentifier():
        raise SystemExit(f"bad database name {name!r}")
    url = make_url(get_settings().database_url)
    eng = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with eng.connect() as c:
        if not c.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": name}).first():
            c.execute(text(f'CREATE DATABASE "{name}"'))
            print(f"created database {name}")
        else:
            print(f"database {name} exists")
