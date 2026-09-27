"""ASGI entrypoint: uvicorn apps.scorer.main:app --port 8001"""

import redis

from apps.scorer.app import create_app
from preflight_core.config import get_settings
from preflight_core.db.engine import get_engine
from preflight_core.logging import setup_logging
from preflight_core.models.registry import BundleProvider, get_registry
from preflight_core.observability.tracing import instrument
from preflight_core.online.store import OnlineStore
from preflight_core.serving.service import ScoringService

setup_logging()
_s = get_settings()
_provider = BundleProvider(get_registry())
_service = ScoringService(
    OnlineStore(redis.Redis.from_url(_s.redis_url)),
    _provider.champion,
    _s.preflight_hash_salt.get_secret_value(),
    shadow=_provider.shadow,
)
app = create_app(
    _service,
    get_engine(),
    _s.scorer_api_token.get_secret_value() if _s.scorer_api_token else None,
)
instrument(app, "preflight-scorer")
