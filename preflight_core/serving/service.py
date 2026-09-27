"""Scoring: diff + online state -> features -> champion bundle -> risk, reasons, plan, test order.

Shared by the scorer API, the worker (via the API), the CLI and the pytest plugin, so every
entry point computes features with the same code the model was trained on.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any

from preflight_core.backfill.parse import hash_author, is_test_path
from preflight_core.features.diff import FileChange
from preflight_core.features.state import Unit
from preflight_core.models.bundle import PreflightBundle
from preflight_core.online.store import OnlineStore

# SPEC feature 5: adaptive CI plan. Pushes to the default branch always get the full matrix.
PLANS = {"low": "lint+affected", "watch": "full", "high": "full-matrix"}


def ci_plan(band: str, abstain: bool, is_default_branch: bool) -> str:
    if is_default_branch:
        return "full-matrix"
    if abstain or band not in PLANS:
        return "full"
    return PLANS[band]


@dataclass
class ScoreRequest:
    repo: str
    sha: str
    files: list[dict[str, Any]]  # {path, added, deleted, change_type?}
    is_pr: bool = True
    is_default_branch: bool = False
    base_sha: str | None = None
    author_hash: str | None = None
    author_login: str | None = None  # hashed here, never stored
    pr_number: int | None = None
    truncated: bool = False
    tests: list[str] | None = None
    test_durations: dict[str, float] = field(default_factory=dict)
    explain: bool = True
    rank_known_tests: bool = False  # Check Run: rank tests seen in history


class NoModelError(RuntimeError):
    pass


def to_unit(req: ScoreRequest, as_of: float, salt: str) -> Unit:
    files = [
        FileChange(
            f["path"],
            int(f.get("added", 0)),
            int(f.get("deleted", 0)),
            f.get("change_type", "modified"),
            is_test_path(f["path"]),
        )
        for f in req.files
    ]
    author = req.author_hash or hash_author(req.author_login, salt)
    return Unit(
        key=req.sha,
        as_of=as_of,
        completed_at=None,
        label=None,
        files=files,
        truncated=req.truncated,
        author=author,
        is_pr=req.is_pr,
        is_default_branch=req.is_default_branch,
        base_sha=req.base_sha,
    )


def _clean(v: float) -> float | None:
    return None if isinstance(v, float) and math.isnan(v) else v


class ScoringService:
    def __init__(self, store: OnlineStore, champion, salt: str, shadow=None) -> None:
        self.store = store
        self._champion = champion  # callables returning PreflightBundle | None
        self._shadow = shadow or (lambda: None)
        self.salt = salt

    def features_for(self, req: ScoreRequest) -> tuple[dict[str, float], dict[str, Any]]:
        state = self.store.get_state(req.repo)
        # as_of = now, nudged past the watermark if clocks disagree (never read the future)
        as_of = max(time.time(), state.watermark + 1e-3)
        return state.features(to_unit(req, as_of, self.salt))

    def score(self, req: ScoreRequest) -> dict[str, Any]:
        t0 = time.perf_counter()
        bundle: PreflightBundle | None = self._champion()
        if bundle is None:
            raise NoModelError("no champion model loaded")
        feats, details = self.features_for(req)
        cfg = self.store.repo_config(req.repo)
        s = bundle.score(
            req.repo, feats, details, blame_free=cfg["blame_free"], explain=req.explain
        )
        out: dict[str, Any] = {
            "repo": req.repo,
            "sha": req.sha,
            "model_version": bundle.version,
            **s.as_dict(),
            "ci_plan": ci_plan(s.band, s.abstain, req.is_default_branch),
        }
        tests = req.tests
        if not tests and req.rank_known_tests and bundle.test_ranker is not None:
            tests = sorted(self.store.get_tests(req.repo).stats)[:5000]
        if tests:
            out["test_order"] = self.order_tests(
                req.repo, [f["path"] for f in req.files], tests, req.test_durations, bundle
            )
        out["latency_ms"] = (time.perf_counter() - t0) * 1000
        out["_features"] = {k: _clean(v) for k, v in feats.items()}  # stripped by the API
        out["_details"] = details
        return out

    def shadow_score(
        self, req: ScoreRequest, feats: dict[str, float], details: dict[str, Any]
    ) -> dict[str, Any] | None:
        bundle = self._shadow()
        if bundle is None:
            return None
        cfg = self.store.repo_config(req.repo)
        s = bundle.score(req.repo, feats, details, blame_free=cfg["blame_free"], explain=False)
        return {
            "model_version": bundle.version,
            **s.as_dict(),
            "ci_plan": ci_plan(s.band, s.abstain, req.is_default_branch),
        }

    def order_tests(
        self,
        repo: str,
        changed: list[str],
        tests: list[str],
        durations: dict[str, float] | None = None,
        bundle: PreflightBundle | None = None,
    ) -> list[str]:
        """Fail-fast order. Without a ranker, the input order is returned unchanged."""
        from preflight_core.models.test_ranker import canonical_test_id

        bundle = bundle or self._champion()
        if bundle is None or bundle.test_ranker is None or not tests:
            return list(tests)
        hist = self.store.get_tests(repo)
        as_of = max(time.time(), hist.watermark + 1e-3)
        canon = {t: canonical_test_id(t) for t in tests}
        feats = {t: hist.features(canon[t], changed, as_of) for t in tests}
        return bundle.test_ranker.rank(feats, durations or {})
