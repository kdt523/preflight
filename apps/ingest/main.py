"""ASGI entrypoint: uvicorn apps.ingest.main:app"""

import redis

from apps.ingest.app import create_app
from preflight_core.config import get_settings
from preflight_core.db.engine import get_engine
from preflight_core.logging import setup_logging
from preflight_core.observability.tracing import instrument

setup_logging()
_s = get_settings()
app = create_app(
    get_engine(),
    redis.Redis.from_url(_s.redis_url),
    _s.github_webhook_secret.get_secret_value(),
    _s.events_stream,
)
instrument(app, "preflight-ingest")
