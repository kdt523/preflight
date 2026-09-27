import numpy as np
import pandas as pd
import pytest

from preflight_core.features.dataset import build_frame
from preflight_core.features.definitions import feature_names
from preflight_core.models.baselines import AlwaysPass, FilesChangedHeuristic, LogisticBaseline
from preflight_core.models.cv import rolling_origin_splits
from preflight_core.models.evaluate import ablate_families, compare, training_rows
from preflight_core.models.metrics import apfd, ece, precision_recall_at, summarize
from tests.synth import make_units


@pytest.fixture(scope="module")
def frame():
    units = make_units(2500, seed=11)
    return build_frame(units, {u.key for u in units}, repo_id=1, repo="acme/app")


# -- metrics -------------------------------------------------------------------------


def test_apfd():
    assert apfd(["a", "b", "c", "d"], {"a"}) == pytest.approx(1 - 1 / 4 + 1 / 8)
    assert apfd(["a", "b", "c", "d"], {"d"}) == pytest.approx(1 - 4 / 4 + 1 / 8)
    assert apfd(["x", "a", "b"], {"a", "b"}) < apfd(["a", "b", "x"], {"a", "b"})


def test_ece_perfect_and_bad():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)
    assert ece(y, p) < 0.02
    assert ece(y, np.clip(p + 0.3, 0, 1)) > 0.2


def test_precision_recall_at_threshold():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1])
    assert precision_recall_at(y, p, 0.5) == (0.5, 0.5)


def test_summary_never_reports_accuracy():
    s = summarize(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.7]), threshold=0.5)
    assert "accuracy" not in s and s["pr_auc"] == 1.0


# -- temporal CV -----------------------------------------------------------------------


def test_folds_train_only_on_labels_known_before_test(frame):
    rows = training_rows(frame)
    folds = list(rolling_origin_splits(rows, n_folds=4))
    assert len(folds) == 4
    for f in folds:
        tr, te = rows.iloc[f.train_idx], rows.iloc[f.test_idx]
        assert (tr["completed_at"] < te["as_of"].min()).all()  # embargo
        assert tr["as_of"].max() < te["as_of"].min()
    tests = np.concatenate([f.test_idx for f in folds])
    assert len(tests) == len(set(tests))  # test slices never overlap


def test_rows_finishing_after_test_start_are_embargoed():
    t = pd.Timestamp("2026-01-01", tz="UTC")
    df = pd.DataFrame(
        {
            "as_of": [t + pd.Timedelta(hours=i) for i in range(200)],
            "completed_at": [
                t + pd.Timedelta(hours=i + (500 if i == 99 else 0.5)) for i in range(200)
            ],
            "label": [i % 2 for i in range(200)],
        }
    )
    f = next(rolling_origin_splits(df, n_folds=2))
    assert 99 not in set(f.train_idx)  # started early, labelled late -> excluded


# -- baselines and model comparison ---------------------------------------------------------


def test_baselines_rank_as_expected(frame):
    table, _ = compare(
        {
            "always_pass": lambda f: AlwaysPass(),
            "files": lambda f: FilesChangedHeuristic(),
            "logreg": lambda f: LogisticBaseline(),
        },
        frame,
        n_folds=4,
    )
    s = table.set_index("model")["pr_auc_mean"]
    base = table.set_index("model")["base_rate_mean"]["always_pass"]
    assert s["always_pass"] == pytest.approx(base, rel=0.05)  # constant score = base rate
    assert s["logreg"] > s["files"] > s["always_pass"]


def test_ablation_shows_file_history_matters(frame):
    t = ablate_families(
        lambda f: LogisticBaseline(), frame, n_folds=3, families=("file_history", "author")
    ).set_index("dropped")
    # the planted risky-file signal lives in file history; author is pure noise here
    assert t.loc["file_history", "delta"] < t.loc["author", "delta"]
    assert len(feature_names({"author"})) == len(feature_names()) - 3
