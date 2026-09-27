import math
import warnings

import numpy as np
import pandas as pd
import pytest
import shap

from preflight_core.features.dataset import build_frame
from preflight_core.features.definitions import AUTHOR, feature_names
from preflight_core.models.bundle import PreflightBundle
from preflight_core.models.evaluate import cross_validate, training_rows
from preflight_core.models.explain import reasons_for_row
from preflight_core.models.lgbm import CalibratedLGBM
from preflight_core.models.metrics import ece
from preflight_core.models.test_ranker import (
    TestRanker,
    canonical_test_id,
    evaluate_ordering,
    replay_tests,
)
from preflight_core.models.threshold import Bands, bands_from_threshold, cost_threshold
from preflight_core.models.train import train_bundle
from tests.synth import PAIR, RISKY, make_test_runs, make_units

warnings.filterwarnings("ignore", category=UserWarning)


@pytest.fixture(scope="module")
def frame():
    units = make_units(3000, seed=21)
    return build_frame(units, {u.key for u in units}, repo_id=1, repo="acme/app")


@pytest.fixture(scope="module")
def model(frame):
    rows = training_rows(frame)
    return CalibratedLGBM(feature_names()).fit(rows[feature_names()], rows["label"].to_numpy())


@pytest.fixture(scope="module")
def bundle(frame):
    return train_bundle(frame, n_folds=3, per_repo_min=10**9, version="vtest")


def row_like(frame, **overrides):
    r = training_rows(frame).iloc[-1]
    f = {n: float(r[n]) for n in feature_names()}
    f.update(overrides)
    return f


# -- model -------------------------------------------------------------------------


def test_lgbm_is_competitive_and_calibrated(frame):
    """The synthetic generator IS a logistic model of a few features, so logistic regression
    has the true functional form here; LightGBM must stay close, beat the heuristic baselines,
    and be calibrated. Which model wins on real data is decided by scripts/evaluate.py."""
    from preflight_core.models.baselines import FilesChangedHeuristic, LogisticBaseline

    lgb = cross_validate("lgbm", lambda f: CalibratedLGBM(f), frame, feature_names(), 3)
    lr = cross_validate("lr", lambda f: LogisticBaseline(), frame, feature_names(), 3)
    files = cross_validate("files", lambda f: FilesChangedHeuristic(), frame, feature_names(), 3)
    s = lgb.summary()
    assert s["pr_auc_mean"] >= lr.summary()["pr_auc_mean"] - 0.06
    assert s["pr_auc_mean"] > files.summary()["pr_auc_mean"]
    assert s["pr_auc_mean"] > 1.4 * s["base_rate_mean"]
    assert ece(lgb.oof["y"], lgb.oof["p"]) < 0.08


def test_isotonic_tie_break_preserves_ranking(model, frame):
    X = training_rows(frame)[feature_names()].tail(300)
    p = model.predict_proba(X)
    raw = model._raw(X)
    order_p, order_raw = np.argsort(p, kind="stable"), np.argsort(raw, kind="stable")
    assert (order_p == order_raw).all()
    assert len(np.unique(p)) == len(np.unique(raw))


def test_contributions_are_exact_shap(model, frame):
    X = training_rows(frame)[feature_names()].tail(20).astype(float)
    ours = model.contributions(X)[:, :-1]
    theirs = shap.TreeExplainer(model.booster).shap_values(X)
    theirs = theirs[1] if isinstance(theirs, list) else theirs
    np.testing.assert_allclose(ours, theirs, atol=1e-6)


def test_smote_ablation_runs(frame):
    rows = training_rows(frame).head(1500)
    m = CalibratedLGBM(feature_names(), smote=True).fit(
        rows[feature_names()], rows["label"].to_numpy()
    )
    assert ((m.predict_proba(rows.tail(50)) >= 0) & (m.predict_proba(rows.tail(50)) <= 1)).all()


# -- behavioural tests (SPEC §8) ------------------------------------------------------------


def test_adding_tests_never_raises_risk(model, frame):
    for i in range(0, 200, 7):
        r = training_rows(frame).iloc[-200 + i]
        f = {n: float(r[n]) for n in feature_names()}
        worse = dict(f, touches_tests=0.0, test_code_ratio=0.0, frac_src_with_tests=0.0)
        better = dict(f, touches_tests=1.0, test_code_ratio=1.0, frac_src_with_tests=1.0)
        p = model.predict_proba(pd.DataFrame([worse, better])[feature_names()])
        assert p[1] <= p[0] + 1e-9


def test_more_untested_churn_never_lowers_risk(model, frame):
    f = row_like(frame, touches_tests=0.0, test_code_ratio=0.0)
    grid = [
        dict(f, lines_added=a, log_churn=math.log1p(a + f["lines_deleted"]))
        for a in (10, 100, 1000, 10000)
    ]
    p = model.predict_proba(pd.DataFrame(grid)[feature_names()])
    assert all(np.diff(p) >= -1e-9)


def test_whitespace_only_diff_stays_low(bundle, frame):
    f = row_like(
        frame,
        n_files=1.0,
        lines_added=2.0,
        lines_deleted=2.0,
        log_churn=math.log1p(4),
        n_dirs=1.0,
        change_entropy=0.0,
        max_file_fail_rate=0.05,
        mean_file_fail_rate=0.05,
        hotspot_score=0.0,
        missing_partners=0.0,
        max_missing_confidence=0.0,
        file_rate_lift=0.5,
        touches_tests=0.0,
        main_red=0.0,
    )
    assert bundle.score("acme/app", f, {}).band == "low"


# -- threshold / bands ------------------------------------------------------------------


def test_cost_threshold_rises_when_false_alarms_cost_more():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(int)
    assert cost_threshold(y, p, c_fp=5, c_fn=1) > cost_threshold(y, p, c_fp=1, c_fn=5)
    assert cost_threshold(y, p, c_fp=1, c_fn=1) == pytest.approx(0.5, abs=0.05)


def test_bands():
    b = Bands(0.25, 0.55)
    assert [b.band(x) for x in (0.1, 0.25, 0.54, 0.55, 0.9)] == [
        "low",
        "watch",
        "watch",
        "high",
        "high",
    ]
    assert bands_from_threshold(0.55) == Bands(0.25, 0.55)


# -- explanations ------------------------------------------------------------------------------


def test_reasons_hide_author_and_dedupe_groups():
    feats = {
        "n_files": 30,
        "lines_added": 900,
        "lines_deleted": 40,
        "n_dirs": 7,
        "author_fail_rate": 0.9,
        "max_file_fail_rate": 0.4,
        "touches_tests": 0.0,
    }
    details = {
        "riskiest_file": {"path": "auth/session.py", "rate": 0.38, "builds": 42, "fails": 16},
        "base_rate": 0.1,
    }
    contrib = {
        "author_fail_rate": 5.0,
        "n_files": 0.9,
        "lines_added": 0.8,
        "max_file_fail_rate": 0.7,
        "touches_tests": 0.3,
    }
    reasons = reasons_for_row(feats, details, contrib, top_k=3)
    texts = [r.text for r in reasons]
    assert not any("author" in r.feature for r in reasons)
    assert sum("Large change" in t for t in texts) == 1  # size family collapsed
    assert "`auth/session.py` failed 16 of the 42 builds" in texts[1]
    assert texts[2] == "No test files changed alongside the code."


def test_bundle_score_end_to_end(bundle, frame):
    risky_path = RISKY[0]
    f = row_like(
        frame,
        max_file_fail_rate=0.6,
        file_rate_lift=4.0,
        missing_partners=1.0,
        max_missing_confidence=0.9,
        touches_tests=0.0,
        test_code_ratio=0.0,
        frac_src_with_tests=0.0,
    )
    d = {
        "riskiest_file": {"path": risky_path, "rate": 0.6, "builds": 50, "fails": 30},
        "missing_partners": [[PAIR[1], 0.9]],
        "untested_files": [PAIR[0]],
        "base_rate": 0.15,
    }
    s = bundle.score("acme/app", f, d)
    assert 0 <= s.p_fail <= 1 and s.band in {"low", "watch", "high"}
    assert s.model_key == "global" and s.reasons
    assert not any(k.startswith("author") for k in s.contributions)
    assert all(w["p_after"] < w["p_before"] for w in s.what_ifs)
    bf = bundle.score("acme/app", f, d, blame_free=True)
    assert bf.model_key == "global-blame-free"
    assert AUTHOR not in {c.split("_")[0] for c in bundle.global_blame_free.features}


def test_bundle_roundtrip_and_metadata(bundle, tmp_path, frame):
    d = bundle.save(tmp_path)
    loaded = PreflightBundle.load(d)
    f = row_like(frame)
    assert loaded.score("x/y", f, {}).p_fail == pytest.approx(bundle.score("x/y", f, {}).p_fail)
    meta = bundle.metadata()
    assert meta["feature_hash"] and meta["metrics"]["global_cv"]["folds"] == 3
    assert "blame_free_pr_auc_cost" in meta["metrics"]


def test_per_repo_model_used_only_when_it_wins(frame):
    two = pd.concat([frame, frame.assign(repo="other/repo", repo_id=2)], ignore_index=True)
    b = train_bundle(two, n_folds=3, per_repo_min=100, version="v2")
    for repo, cmp_ in b.metrics["per_repo"].items():
        assert (repo in b.per_repo) == (cmp_["selected"] == "repo")
    assert b.variant("brand/new-repo")[0] == "global"  # cold start


# -- per-test model -------------------------------------------------------------------------------


def test_canonical_test_ids():
    assert canonical_test_id("tests.test_api::test_login") == "tests/test_api.py::test_login"
    assert (
        canonical_test_id("tests.unit.test_api.TestAuth::test_x")
        == "tests/unit/test_api.py::TestAuth::test_x"
    )
    assert canonical_test_id("tests/test_api.py::test_login") == "tests/test_api.py::test_login"


def test_ranker_beats_default_order():
    runs = make_test_runs()
    rows, evals = replay_tests(runs[:220], neg_per_run=20)
    ranker = TestRanker().fit(rows)
    _, held = replay_tests(runs)
    m = evaluate_ordering(ranker, held[220:])
    assert m["runs"] > 20
    assert m["apfd_model"] > m["apfd_default"] + 0.15
    assert m["apfd_model"] > m["apfd_random"] + 0.15
    assert m["ttff_reduction"] > 0


def test_test_history_is_point_in_time():
    runs = make_test_runs(40)
    _, evals = replay_tests(runs)
    first = evals[0]
    assert all(f["t_is_new"] == 1.0 for f in first["features"].values())
