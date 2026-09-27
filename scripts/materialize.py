"""Nightly batch: rebuild every repo's online state in Redis from Postgres history.

uv run python scripts/materialize.py        (scheduled nightly: EventBridge on AWS)
"""

import json

import redis

from preflight_core.config import get_settings
from preflight_core.db.engine import get_engine
from preflight_core.online.materialize import materialize_all
from preflight_core.online.store import OnlineStore

if __name__ == "__main__":
    store = OnlineStore(redis.Redis.from_url(get_settings().redis_url))
    print(json.dumps(materialize_all(get_engine(), store), indent=2))
