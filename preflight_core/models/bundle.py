"""A trained, deployable Preflight model set.

One bundle holds: the global cross-repo model (cold start), per-repo models for repos with
enough labels (used only where they beat the global model in CV), blame-free twins without
author features, the per-test ranker, tuned bands, and the PSI reference distribution.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from preflight_core.features.definitions import (
    FEATURE_SET_VERSION,
    HIDDEN_FAMILIES,
    feature_names,
    monotone_constraints,
)
from preflight_core.models.conformal import ConformalAbstainer
from preflight_core.models.explain import reasons_for_row, visible_contributions
from preflight_core.models.lgbm import CalibratedLGBM
from preflight_core.models.test_ranker import TestRanker
from preflight_core.models.threshold import Bands
from preflight_core.models.whatif import what_ifs


def feature_hash(names: list[str]) -> str:
    spec = f"{FEATURE_SET_VERSION}|" + ",".join(
        f"{n}:{m}" for n, m in zip(names, monotone_constraints(names), strict=True)
    )
    return hashlib.sha256(spec.encode()).hexdigest()[:16]


@dataclass
class ModelVariant:
    model: CalibratedLGBM
    conformal: ConformalAbstainer
    bands: Bands
    features: list[str]
    n_train: int
    cv: dict[str, Any] = field(default_factory=dict)


@dataclass
class Score:
    p_fail: float
    band: str
    conformal_set: list[str] | None
    abstain: bool
    model_key: str
    reasons: list[dict[str, Any]]
    what_ifs: list[dict[str, Any]]
    contributions: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreflightBundle:
    version: str
    created_at: str
    data_window: tuple[str, str]
    global_model: ModelVariant
    global_blame_free: ModelVariant
    per_repo: dict[str, ModelVariant] = field(default_factory=dict)
    per_repo_bands: dict[str, Bands] = field(default_factory=dict)
    test_ranker: TestRanker | None = None
    reference: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    feature_set_version: str = FEATURE_SET_VERSION

    @property
    def feature_hash(self) -> str:
        return feature_hash(feature_names())

    def variant(self, repo: str, blame_free: bool = False) -> tuple[str, ModelVariant]:
        if not blame_free and repo in self.per_repo:
            return f"repo:{repo}", self.per_repo[repo]
        return (
            ("global-blame-free", self.global_blame_free)
            if blame_free
            else ("global", self.global_model)
        )

    def bands_for(self, repo: str, variant: ModelVariant) -> Bands:
        return self.per_repo_bands.get(repo, variant.bands)

    def score(
        self,
        repo: str,
        features: dict[str, float],
        details: dict[str, Any],
        blame_free: bool = False,
        explain: bool = True,
    ) -> Score:
        key, v = self.variant(repo, blame_free)
        X = pd.DataFrame([{n: features.get(n, np.nan) for n in v.features}])[v.features]
        p = float(v.model.predict_proba(X)[0])
        cset = v.conformal.sets(X)[0]
        reasons, wis, contrib = [], [], {}
        if explain:
            c = v.model.contributions(X)[0, :-1]
            contrib = visible_contributions(v.features, c)
            all_c = dict(zip(v.features, map(float, c), strict=True))
            reasons = [asdict(r) for r in reasons_for_row(features, details, all_c)]
            wis = [
                {**asdict(w), "delta": w.delta}
                for w in what_ifs(
                    v.model.predict_proba,
                    {n: features.get(n, np.nan) for n in v.features},
                    details,
                    v.features,
                )
            ]
        return Score(
            p_fail=p,
            band=self.bands_for(repo, v).band(p),
            conformal_set=cset,
            abstain=ConformalAbstainer.abstains(cset),
            model_key=key,
            reasons=reasons,
            what_ifs=wis,
            contributions=contrib,
        )

    # -- persistence -----------------------------------------------------------------

    def metadata(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "feature_set_version": self.feature_set_version,
            "feature_hash": self.feature_hash,
            "data_window": list(self.data_window),
            "per_repo_models": sorted(self.per_repo),
            "has_test_ranker": self.test_ranker is not None,
            "hidden_families": sorted(HIDDEN_FAMILIES),
            "global_bands": asdict(self.global_model.bands),
            "per_repo_bands": {k: asdict(b) for k, b in self.per_repo_bands.items()},
            "metrics": self.metrics,
        }

    def save(self, root: str | Path = "models/bundles") -> Path:
        d = Path(root) / self.version
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, d / "bundle.joblib", compress=3)
        (d / "metadata.json").write_text(json.dumps(self.metadata(), indent=2, default=str))
        return d

    @staticmethod
    def load(path: str | Path) -> "PreflightBundle":
        p = Path(path)
        return joblib.load(p / "bundle.joblib" if p.is_dir() else p)
