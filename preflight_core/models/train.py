"""Train a full PreflightBundle from the feature dataset (+ optional per-test runs)."""

import logging
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from preflight_core.features.definitions import AUTHOR, drift_features, feature_names
from preflight_core.models.bundle import ModelVariant, PreflightBundle
from preflight_core.models.conformal import ConformalAbstainer
from preflight_core.models.evaluate import cross_validate, training_rows
from preflight_core.models.lgbm import CalibratedLGBM
from preflight_core.models.metrics import pr_auc
from preflight_core.models.test_ranker import TestRanker, TestRun, evaluate_ordering, replay_tests
from preflight_core.models.threshold import (
    DEFAULT_COST_FN,
    DEFAULT_COST_FP,
    Bands,
    tune_bands,
)
from preflight_core.monitoring.drift import reference

log = logging.getLogger(__name__)

PER_REPO_MIN_LABELLED = 500  # SPEC: per-repo model after ~500 labelled builds


def lgbm_factory(features: list[str]) -> CalibratedLGBM:
    return CalibratedLGBM(features)


def fit_variant(
    rows: pd.DataFrame, features: list[str], bands: Bands, conf_level: float, cv: dict | None = None
) -> ModelVariant:
    model = CalibratedLGBM(features).fit(rows[features], rows["label"].to_numpy())
    conformal = ConformalAbstainer(conf_level).fit(model, model.X_conf_, model.y_conf_)
    return ModelVariant(model, conformal, bands, features, len(rows), cv or {})


def train_bundle(
    df: pd.DataFrame,
    test_runs: list[TestRun] | None = None,
    version: str | None = None,
    n_folds: int = 5,
    per_repo_min: int = PER_REPO_MIN_LABELLED,
    c_fp: float = DEFAULT_COST_FP,
    c_fn: float = DEFAULT_COST_FN,
    conf_level: float = 0.9,
) -> PreflightBundle:
    rows = training_rows(df)
    if rows["label"].nunique() < 2:
        raise ValueError("need both passing and failing labelled builds to train")
    all_f, no_author = feature_names(), feature_names({AUTHOR})

    # Honest metrics + band tuning from rolling-origin out-of-fold predictions.
    cv_all = cross_validate("global", lgbm_factory, rows, all_f, n_folds)
    cv_bf = cross_validate("global-blame-free", lgbm_factory, rows, no_author, n_folds)
    s_all, s_bf = cv_all.summary(), cv_bf.summary()
    if cv_all.oof is not None:
        global_bands, repo_bands = tune_bands(
            cv_all.oof["y"], cv_all.oof["p"], cv_all.oof["repo"], c_fp, c_fn
        )
    else:
        global_bands, repo_bands = Bands(), {}

    global_v = fit_variant(rows, all_f, global_bands, conf_level, s_all)
    bf_v = fit_variant(rows, no_author, global_bands, conf_level, s_bf)

    # Per-repo models: only where enough labels AND they beat the global model on that repo.
    per_repo: dict[str, ModelVariant] = {}
    repo_cmp: dict[str, Any] = {}
    for repo, g in rows.groupby("repo"):
        if len(g) < per_repo_min:
            continue
        cv_r = cross_validate(
            f"repo:{repo}", lgbm_factory, g.reset_index(drop=True), all_f, n_folds
        )
        s_r = cv_r.summary()
        if cv_r.oof is None or cv_all.oof is None:
            continue
        glob_on_repo = cv_all.oof[cv_all.oof["repo"] == repo]
        g_auc = (
            pr_auc(glob_on_repo["y"], glob_on_repo["p"])
            if glob_on_repo["y"].nunique() == 2
            else float("nan")
        )
        r_auc = s_r.get("pooled_pr_auc", float("nan"))
        use = r_auc > g_auc if g_auc == g_auc else False
        repo_cmp[repo] = {
            "repo_model_pr_auc": r_auc,
            "global_on_repo_pr_auc": g_auc,
            "selected": "repo" if use else "global",
        }
        if use:
            per_repo[repo] = fit_variant(
                g.reset_index(drop=True), all_f, repo_bands.get(repo, global_bands), conf_level, s_r
            )

    ranker, test_metrics = None, {}
    if test_runs:
        ranker, test_metrics = train_test_ranker(test_runs)

    now = datetime.now(UTC)
    return PreflightBundle(
        version=version or now.strftime("v%Y%m%d%H%M%S"),
        created_at=now.isoformat(),
        data_window=(str(rows["as_of"].min()), str(rows["as_of"].max())),
        global_model=global_v,
        global_blame_free=bf_v,
        per_repo=per_repo,
        per_repo_bands=repo_bands,
        test_ranker=ranker,
        # newest 30% of training: closest to what serving sees, past repo warm-up
        reference=reference(rows[drift_features()].iloc[-max(1, int(len(rows) * 0.3)) :]),
        metrics={
            "n_labelled": len(rows),
            "base_rate": float(rows["label"].mean()),
            "global_cv": s_all,
            "blame_free_cv": s_bf,
            "blame_free_pr_auc_cost": (s_all.get("pr_auc_mean") or 0)
            - (s_bf.get("pr_auc_mean") or 0),
            "per_repo": repo_cmp,
            "test_ranker": test_metrics,
            "costs": {"false_alarm": c_fp, "miss": c_fn},
            "conformal_confidence": conf_level,
        },
    )


def train_test_ranker(runs: list[TestRun], holdout: float = 0.2) -> tuple[TestRanker, dict]:
    runs = sorted(runs, key=lambda r: r.as_of)
    cut = int(len(runs) * (1 - holdout))
    rows, evals = replay_tests(runs, neg_per_run=50)
    train_keys = {r.key for r in runs[:cut]}
    metrics: dict[str, Any] = {"runs": len(runs)}
    tr = rows[rows["run"].isin(train_keys)]
    if tr["y"].nunique() == 2:
        held = [e for e in evals if e["key"] not in train_keys]
        metrics |= evaluate_ordering(TestRanker().fit(tr), held)
    final = TestRanker().fit(rows) if rows["y"].nunique() == 2 else None
    return final, metrics
