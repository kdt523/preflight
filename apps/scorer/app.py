"""Risk scorer API. Latency path: must answer well inside the worker's 2 s fail-open budget."""

import hmac
import logging
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field
from sqlalchemy import Engine, select
from sqlalchemy.dialects.postgresql import insert

from preflight_core.db.schema import predictions, repos
from preflight_core.features.state import LeakageError
from preflight_core.observability.metrics import (
    ABSTAIN,
    P_FAIL,
    PREDICTIONS,
    SCORE_ERRORS,
    SCORE_LATENCY,
)
from preflight_core.serving.service import NoModelError, ScoreRequest, ScoringService

log = logging.getLogger("scorer")


class FileIn(BaseModel):
    path: str
    added: int = 0
    deleted: int = 0
    change_type: str = "modified"


class ScoreIn(BaseModel):
    repo: str
    sha: str = Field(min_length=7, max_length=40)
    files: list[FileIn]
    is_pr: bool = True
    is_default_branch: bool = False
    base_sha: str | None = None
    author_hash: str | None = None
    author_login: str | None = None
    pr_number: int | None = None
    truncated: bool = False
    tests: list[str] | None = None
    test_durations: dict[str, float] = {}
    explain: bool = True
    rank_known_tests: bool = False
    log_prediction: bool = True  # the CLI sends False for local what-if checks


class TestsIn(BaseModel):
    repo: str
    changed_files: list[str]
    tests: list[str]
    durations: dict[str, float] = {}


def create_app(
    service: ScoringService, engine: Engine | None = None, api_token: str | None = None
) -> FastAPI:
    app = FastAPI(title="preflight-scorer")

    def require_token(authorization: str | None = Header(None)) -> None:
        """Bearer auth on /v1/* when a token is configured (always on AWS)."""
        if not api_token:
            return
        given = (authorization or "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(given, api_token):
            raise HTTPException(401, "missing or invalid bearer token")

    def repo_id(name: str) -> int | None:
        if engine is None:
            return None
        with engine.connect() as c:
            return c.execute(select(repos.c.id).where(repos.c.full_name == name)).scalar()

    def log_prediction(req: ScoreRequest, out: dict[str, Any], shadow: bool) -> int | None:
        rid = repo_id(req.repo)
        if rid is None:
            return None
        with engine.begin() as c:
            return c.execute(
                insert(predictions)
                .values(
                    repo_id=rid,
                    sha=req.sha,
                    pr_number=req.pr_number,
                    model_version=out["model_version"],
                    model_key=out["model_key"],
                    shadow=shadow,
                    p_fail=out["p_fail"],
                    band=out["band"],
                    conformal_set=out["conformal_set"],
                    shap_top=out.get("reasons"),
                    what_ifs=out.get("what_ifs"),
                    ci_plan=out["ci_plan"],
                    latency_ms=out.get("latency_ms"),
                    features=out.get("_features"),
                )
                .returning(predictions.c.id)
            ).scalar()

    def run_shadow(req: ScoreRequest, feats: dict, details: dict) -> None:
        try:
            s = service.shadow_score(req, feats, details)
            if s is not None and engine is not None:
                log_prediction(req, {**s, "_features": feats}, shadow=True)
        except Exception:  # shadow must never affect the champion path
            log.exception("shadow scoring failed", extra={"repo": req.repo, "sha": req.sha})

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, Any]:
        b = service._champion()
        if b is None:
            raise HTTPException(503, "no champion model")
        return {"status": "ready", "champion": b.version}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/score", dependencies=[Depends(require_token)])
    def score(body: ScoreIn, background: BackgroundTasks) -> dict[str, Any]:
        req = ScoreRequest(
            **body.model_dump(exclude={"files", "log_prediction"}),
            files=[f.model_dump() for f in body.files],
        )
        with SCORE_LATENCY.time():
            try:
                out = service.score(req)
            except NoModelError as e:
                SCORE_ERRORS.labels("no_model").inc()
                raise HTTPException(503, str(e)) from e
            except LeakageError as e:
                SCORE_ERRORS.labels("leakage_guard").inc()
                raise HTTPException(500, "point-in-time guard tripped") from e
        PREDICTIONS.labels(out["band"], out["model_key"]).inc()
        P_FAIL.observe(out["p_fail"])
        if out["abstain"]:
            ABSTAIN.inc()
        feats, details = out.pop("_features"), out.pop("_details")
        if body.log_prediction and engine is not None:
            out["prediction_id"] = log_prediction(req, {**out, "_features": feats}, False)
            background.add_task(run_shadow, req, feats, details)
        return out

    @app.post("/v1/tests/order", dependencies=[Depends(require_token)])
    def tests_order(body: TestsIn) -> dict[str, Any]:
        order = service.order_tests(body.repo, body.changed_files, body.tests, body.durations)
        return {"order": order, "reordered": order != body.tests}

    return app
