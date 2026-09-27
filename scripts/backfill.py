"""Backfill CI history for the repos in repos.yaml. Safe to stop and re-run at any time.

uv run python scripts/backfill.py                   # all repos, last 180 days
uv run python scripts/backfill.py --only pandas-dev/pandas --days 30
uv run python scripts/backfill.py --stages runs     # cheap first pass: build list only
"""

import argparse
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

from preflight_core.backfill.runner import STAGES, BackfillConfig, Backfiller
from preflight_core.config import get_settings
from preflight_core.db.engine import get_engine
from preflight_core.github.client import GitHubClient
from preflight_core.logging import setup_logging
from preflight_core.storage.s3 import get_raw_store

log = logging.getLogger("backfill")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", default="repos.yaml")
    ap.add_argument("--only", help="backfill a single repo (owner/name)")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    args = ap.parse_args()
    setup_logging()

    s = get_settings()
    token = s.github_token.get_secret_value()
    salt = s.preflight_hash_salt.get_secret_value()
    if not token:
        raise SystemExit("GITHUB_TOKEN is empty in .env")
    if not salt or salt.startswith("change-me"):
        raise SystemExit("Set PREFLIGHT_HASH_SALT in .env to a long random string first")

    repos = yaml.safe_load(Path(args.repos).read_text())["repos"]
    if args.only:
        repos = [r for r in repos if r["full_name"] == args.only]

    # Stop a day short of now so in-progress runs are not skipped past by the cursor.
    until = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(days=1)
    cfg = BackfillConfig(since=until - timedelta(days=args.days), until=until)
    # No ETag cache here: history is immutable and commit bodies contain source patches.
    gh = GitHubClient(token, s.github_api_url)
    bf = Backfiller(gh, get_engine(), get_raw_store(), salt, cfg)

    for r in repos:
        stats = bf.run_repo(
            r["full_name"],
            r["default_branch"],
            junit=r.get("junit_artifacts", False),
            stages=args.stages,
        )
        log.info("repo done", extra={"repo": r["full_name"], **vars(stats), "calls": gh.calls})


if __name__ == "__main__":
    main()
