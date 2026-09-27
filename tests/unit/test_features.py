import dataclasses
import json
import math

import pandas as pd
import pytest

from preflight_core.features.dataset import build_frame, unit_label, validate
from preflight_core.features.definitions import FEATURES, feature_names
from preflight_core.features.diff import FileChange, change_entropy, diff_features
from preflight_core.features.state import (
    HistoryState,
    LeakageError,
    Unit,
    replay,
    smoothed,
)
from tests.synth import PAIR, RISKY, make_units

H = 3600.0


def unit(key, as_of, done=None, label=None, paths=("src/a.py",), **kw):
    return Unit(
        key=key,
        as_of=as_of,
        completed_at=done,
        label=label,
        files=[FileChange(p, 10, 2) for p in paths],
        **kw,
    )


# -- diff features -----------------------------------------------------------------


def test_change_entropy():
    assert change_entropy([10]) == 0.0
    assert change_entropy([5, 5]) == pytest.approx(1.0)
    assert change_entropy([5, 5, 5, 5]) == pytest.approx(1.0)
    assert 0 < change_entropy([100, 1]) < 0.2


def test_change_type_detection():
    f = diff_features(
        [
            FileChange("poetry.lock", 100, 90),
            FileChange(".github/workflows/ci.yml", 2, 1),
            FileChange("app/migrations/0003_x.py", 20, 0, "added"),
            FileChange("pyproject.toml", 1, 1),
        ]
    )
    assert (
        f["touches_lockfile"],
        f["touches_ci_config"],
        f["touches_migrations"],
        f["touches_dep_manifest"],
        f["n_new_files"],
    ) == (1, 1, 1, 1, 1)
    assert (
        diff_features([FileChange("docs/a.md", 1, 0), FileChange("README.md", 1, 0)])["docs_only"]
        == 1
    )
    assert (
        diff_features([FileChange("docs/a.md", 1, 0), FileChange("x.py", 1, 0)])["docs_only"] == 0
    )


# -- smoothing ---------------------------------------------------------------------


def test_bayesian_smoothing_shrinks_sparse_rates_to_prior():
    assert smoothed(1, 1, prior=0.1, strength=10) == pytest.approx(2 / 11)
    assert smoothed(0, 0, prior=0.1, strength=10) == pytest.approx(0.1)
    assert smoothed(500, 1000, prior=0.1, strength=10) == pytest.approx(0.496, abs=1e-3)


# -- point-in-time correctness -------------------------------------------------------


def test_guard_refuses_to_read_state_at_or_after_as_of():
    s = HistoryState()
    s.apply_start(unit("a", 100.0))
    with pytest.raises(LeakageError):
        s.features(unit("b", 100.0))
    s.features(unit("c", 100.5))  # strictly later is fine


def test_outcome_visible_only_after_completion():
    a = unit("a", 0.0, done=2 * H, label=1)
    before = unit("b", 1 * H)  # a still running
    tie = unit("c", 2 * H)  # a completes exactly at c's as_of -> still invisible
    after = unit("d", 3 * H)
    feats = {u.key: f for u, f, _ in replay([a, before, tie, after])}
    prior = feats["b"]["max_file_fail_rate"]
    assert feats["c"]["max_file_fail_rate"] == pytest.approx(prior)
    assert feats["d"]["max_file_fail_rate"] > prior


def test_simultaneous_starts_do_not_see_each_other():
    a = unit("a", 10.0, paths=("x.py", "y.py"))
    b = unit("b", 10.0, paths=("x.py",))
    feats = {u.key: f for u, f, _ in replay([a, b])}
    assert feats["b"]["concurrent_same_files"] == 0
    c = unit("c", 11.0, paths=("x.py",))
    assert {u.key: f for u, f, _ in replay([a, b, c])}["c"]["concurrent_same_files"] == 2


@pytest.mark.parametrize("cut_index", [50, 300, 777, 1199])
def test_features_are_invariant_to_the_future(cut_index):
    """The leakage test: deleting or changing everything at/after as_of changes nothing."""
    units = make_units(1200, seed=3)
    full = {u.key: f for u, f, _ in replay(units)}
    cut = units[cut_index]
    past = []
    for u in units:
        if u.as_of > cut.as_of or (u.as_of == cut.as_of and u.key != cut.key):
            continue  # future units are gone
        if u.completed_at is not None and u.completed_at >= cut.as_of:
            # outcome not known yet at cut.as_of: flip it to prove it is never read
            u = dataclasses.replace(
                u, label=None if u.label is None else 1 - u.label, completed_at=u.completed_at + 1e6
            )
        past.append(u)
    truncated = {u.key: f for u, f, _ in replay(past)}
    assert truncated[cut.key] == full[cut.key]


def test_state_serialisation_roundtrip_gives_identical_features():
    units = make_units(400, seed=5)
    s = HistoryState()
    for u in units[:-1]:
        s.apply_start(u)
        s.apply_completion(u) if u.completed_at and u.completed_at < units[-1].as_of else None
    probe = units[-1]
    restored = HistoryState.from_dict(json.loads(json.dumps(s.to_dict())))
    a, b = s.features(probe)[0], restored.features(probe)[0]
    assert a.keys() == b.keys()
    for k in a:
        assert (math.isnan(a[k]) and math.isnan(b[k])) or a[k] == pytest.approx(b[k]), k


# -- planted signals are recovered ---------------------------------------------------


def test_file_history_and_cochange_find_planted_signals():
    out = replay(make_units(1500, seed=1))
    late = [(u, f, d) for u, f, d in out[-500:]]
    risky = [f["max_file_fail_rate"] for u, f, _ in late if {x.path for x in u.files} & set(RISKY)]
    safe = [
        f["max_file_fail_rate"]
        for u, f, _ in late
        if not {x.path for x in u.files} & (set(RISKY) | set(PAIR))
    ]
    assert sum(risky) / len(risky) > 2 * sum(safe) / len(safe)

    lonely = [
        (f, d)
        for u, f, d in late
        if PAIR[0] in {x.path for x in u.files} and PAIR[1] not in {x.path for x in u.files}
    ]
    assert lonely and all(f["missing_partners"] >= 1 for f, _ in lonely)
    assert lonely[0][1]["missing_partners"][0][0] == PAIR[1]


def test_untested_source_detected():
    s = HistoryState()
    s.apply_start(Unit("t", 1.0, None, None, [FileChange("tests/test_a.py", 5, 0, is_test=True)]))
    f, d = s.features(
        Unit("u", 2.0, None, None, [FileChange("src/a.py", 5, 0), FileChange("src/b.py", 5, 0)])
    )
    assert f["frac_src_with_tests"] == 0.5 and d["untested_files"] == ["src/b.py"]


def test_pushes_since_base_counts_default_branch_pushes():
    base = unit("base", 1.0, is_default_branch=True)
    pushes = [unit(f"p{i}", 2.0 + i, is_default_branch=True) for i in range(3)]
    pr = unit("pr", 10.0, is_pr=True, base_sha="base")
    feats = {u.key: f for u, f, _ in replay([base, *pushes, pr])}
    assert feats["pr"]["pushes_since_base"] == 3
    assert math.isnan(feats["base"]["pushes_since_base"])


# -- dataset ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "types, want",
    [
        (["pass", "pass"], 0),
        (["pass", "code_defect"], 1),
        (["pass", "flaky"], None),
        (["infra"], None),
        (["cancelled", "pass"], 0),
        (["cancelled"], None),
        (["code_defect", "infra"], 1),
    ],
)
def test_unit_label(types, want):
    assert unit_label(types) == want


def test_frame_validates_and_has_every_feature():
    units = make_units(300, seed=2)
    df = build_frame(units, {u.key for u in units}, repo_id=1, repo="acme/app")
    validate(df)
    assert list(df.columns[6:-1]) == feature_names() and len(feature_names()) == len(FEATURES)


def test_schema_rejects_impossible_values():
    units = make_units(50, seed=2)
    df = build_frame(units, {u.key for u in units}, repo_id=1, repo="acme/app")
    df.loc[0, "max_file_fail_rate"] = 1.5
    df.loc[1, "completed_at"] = df.loc[1, "as_of"] - pd.Timedelta(minutes=1)
    with pytest.raises(Exception) as e:
        validate(df)
    assert "max_file_fail_rate" in str(e.value) and "label known before" in str(e.value)
