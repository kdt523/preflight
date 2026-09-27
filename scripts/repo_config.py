"""Per-repo serving options.

uv run python scripts/repo_config.py acme/app --blame-free on
uv run python scripts/repo_config.py acme/app            # show
"""

import argparse

import redis

from preflight_core.config import get_settings
from preflight_core.online.store import OnlineStore

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument(
        "--blame-free",
        choices=["on", "off"],
        help="serve the model without author features for this repo",
    )
    args = ap.parse_args()
    store = OnlineStore(redis.Redis.from_url(get_settings().redis_url))
    if args.blame_free:
        store.set_repo_config(args.repo, blame_free=args.blame_free == "on")
    print(store.repo_config(args.repo))
