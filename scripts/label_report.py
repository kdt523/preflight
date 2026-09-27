"""Week-1 exit check: failure-type breakdown straight from Postgres.

uv run python scripts/label_report.py
"""

from preflight_core.db.engine import get_engine
from preflight_core.labeling.report import build_report

if __name__ == "__main__":
    print(build_report(get_engine()))
