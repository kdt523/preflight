from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import yaml

from preflight_core.features.dataset import build_frame
from preflight_core.features.definitions import drift_features
from preflight_core.models.bundle import PreflightBundle
from preflight_core.models.promotion import LiveMetrics, decide
from preflight_core.monitoring.drift import PSI_ALERT, psi, psi_table, reference
from preflight_core.monitoring.monitor import retrain_reasons
from tests.synth import make_units

NOW = datetime(2026, 9, 1, tzinfo=UTC)


def live(n=500, pos=60, auc=0.4, ece_=0.03, days_ago=10):
    return LiveMetrics(n, pos, auc, ece_, NOW - timedelta(days=days_ago))


# -- promotion rules --------------------------------------------------------------------


def test_promotion_rules():
    champ = live(auc=0.40, ece_=0.03)
    assert decide(champ, live(auc=0.45), NOW)[0]
    assert not decide(champ, live(auc=0.35), NOW)[0]
    assert "< 7 days" in decide(champ, live(auc=0.5, days_ago=3), NOW)[1]
    assert "failing builds" in decide(champ, live(auc=0.5, pos=5), NOW)[1]
    assert "ECE" in decide(champ, live(auc=0.45, ece_=0.09), NOW)[1]
    first = LiveMetrics(0, 0, float("nan"), float("nan"), None)
    assert decide(first, live(), NOW)[0]  # first champion: minimums only


# -- drift + retrain triggers ------------------------------------------------------------


@pytest.fixture(scope="module")
def train_X():
    u = make_units(3000, seed=9)
    return build_frame(u, {x.key for x in u}, 1, "a/b")[drift_features()]


def test_psi_stable_on_stationary_features_and_flags_a_shift(train_X):
    ref = reference(train_X.iloc[1500:2200])
    diff_derived = [
        "n_files",
        "lines_added",
        "log_churn",
        "change_entropy",
        "touches_tests",
        "frac_test_files",
        "is_pr",
        "missing_partners",
    ]
    same = psi_table(ref, train_X.iloc[2200:]).set_index("feature")
    assert (same.loc[diff_derived, "psi"] < 0.1).all()
    shifted = train_X.iloc[2200:].copy()
    shifted["lines_added"] *= 8  # e.g. a repo starts vendoring generated code
    shifted["n_files"] += 20
    t = psi_table(ref, shifted).set_index("feature")
    assert t.loc["lines_added", "psi"] > PSI_ALERT and t.loc["n_files", "status"] == "shift"


def test_counters_and_repo_scalars_are_not_drift_monitored(synth_bundle):
    for f in ("author_prior_builds", "repo_base_rate", "repo_recent_fail_rate"):
        assert f not in drift_features() and f not in synth_bundle.reference
    assert "max_file_fail_rate" in synth_bundle.reference


def test_psi_counts_missing_values_as_their_own_bin():
    ref = reference(pd.DataFrame({"x": np.r_[np.arange(100.0), [np.nan] * 5]}))
    assert psi(ref["x"], np.r_[np.arange(100.0), [np.nan] * 5]) < 0.01
    assert psi(ref["x"], np.r_[np.arange(50.0), [np.nan] * 55]) > PSI_ALERT


def test_drift_alert_fires_and_triggers_retrain():
    """Week-5 exit: a drift alert fires in a test."""
    reasons = retrain_reasons(
        served=500,
        pmax=0.4,
        live=live(),
        cv_pr_auc=0.4,
        trained_at=NOW - timedelta(days=1),
        now=NOW,
    )
    assert reasons == ["drift: max PSI 0.40 > 0.25"]
    assert retrain_reasons(50, 0.9, live(), 0.4, NOW, NOW) == []  # too few served to judge
    perf = retrain_reasons(500, 0.0, live(auc=0.2), 0.4, NOW, NOW)
    assert perf and perf[0].startswith("performance")
    assert retrain_reasons(500, 0.0, live(), 0.4, NOW - timedelta(days=8), NOW) == [
        "schedule: champion older than 7 days"
    ]


def test_prometheus_alert_rules_match_code_thresholds():
    with open("infra/monitoring/prometheus/alerts.yml") as f:
        rules = yaml.safe_load(f)
    by_name = {r["alert"]: r for g in rules["groups"] for r in g["rules"]}
    assert f"preflight_drift_psi_max > {PSI_ALERT}" in by_name["PreflightFeatureDrift"]["expr"]
    assert "0.3" in by_name["PreflightScorerSlow"]["expr"]  # p95 < 300 ms SLO
    for name in (
        "PreflightFailOpenSpike",
        "PreflightQueueBacklog",
        "PreflightDeadLetters",
        "PreflightLivePrAucDrop",
    ):
        assert name in by_name


# -- MLflow registry lifecycle -------------------------------------------------------------


def test_mlflow_registry_shadow_promote_rollback(tmp_path, monkeypatch, synth_bundle):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    import mlflow

    from preflight_core.models.mlflow_registry import MlflowRegistry
    from preflight_core.models.tracking import log_to_mlflow

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    reg = MlflowRegistry("preflight-test", tmp_path / "cache", client=mlflow.MlflowClient())
    for v in ("v1", "v2"):
        synth_bundle.version = v
        run_id = log_to_mlflow(synth_bundle, synth_bundle.save(tmp_path / "bundles"))
        reg.add_candidate(v, run_id)
    synth_bundle.version = "vsynth"

    reg.promote("v1")
    reg.set_shadow("v2")
    assert (reg.read()["champion"], reg.read()["shadow"]) == ("v1", "v2")
    assert PreflightBundle.load(reg.bundle_path("v2")).version == "v2"  # downloaded artifact
    reg.promote("v2")
    ptr = reg.read()
    assert (ptr["champion"], ptr["shadow"], ptr["previous"]) == ("v2", None, "v1")
    assert reg.rollback() == "v1" and reg.read()["champion"] == "v1"
    with pytest.raises(RuntimeError):
        reg.rollback()
    run = mlflow.get_run(run_id)
    assert run.data.tags["feature_hash"] and "global.pr_auc_mean" in run.data.metrics
