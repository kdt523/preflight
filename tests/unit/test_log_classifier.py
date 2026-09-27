import random

import pandas as pd

from preflight_core.labeling.log_classifier import LogClassifier, label_quality, train
from preflight_core.labeling.rules import CODE, INFRA, JobEvidence, classify_build, classify_job

CODE_LOGS = [
    "FAILED tests/test_{n}.py::test_x - assert {n} == 2",
    "E   KeyError: 'user_{n}'\nsrc/app.py:{n}: KeyError",
    "tests failed: {n} of 300 in suite api",
]
INFRA_LOGS = [
    "The runner has received a shutdown signal. job {n}",
    "curl: (6) Could not resolve host: mirror{n}.example.org",
    "No space left on device while writing layer {n}",
]
# Unlabelled-by-rules styles the classifier must learn:
CODE_ODD = "Segfault in extension module _speedups_{n} during test collection"
INFRA_ODD = "Worker lost connection to docker daemon, retry {n} exhausted"


def corpus(n=40, seed=0):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        rows.append({"log_tail": rng.choice(CODE_LOGS).format(n=i), "human_label": CODE})
        rows.append({"log_tail": rng.choice(INFRA_LOGS).format(n=i), "human_label": INFRA})
        rows.append({"log_tail": CODE_ODD.format(n=i), "human_label": CODE})
        rows.append({"log_tail": INFRA_ODD.format(n=i), "human_label": INFRA})
    return pd.DataFrame(rows)


def test_label_quality_reports_rule_precision_and_classifier_cv():
    q = label_quality(corpus())
    assert q["n"] == 160
    assert 0.3 < q["rule_coverage"] < 0.6  # the odd styles escape the rules
    assert q["per_class"][INFRA]["rule_precision"] == 1.0
    assert q["classifier_cv_report"]["macro avg"]["f1-score"] > 0.9


def test_classifier_fills_in_where_rules_are_silent(tmp_path):
    clf = LogClassifier(train(corpus()))
    clf.save(tmp_path / "m.joblib")
    clf = LogClassifier.load(tmp_path / "m.joblib")
    lines = [INFRA_ODD.format(n=999)]
    assert classify_job(JobEvidence("failure", None, lines))[0] is None
    cat, source, reason = classify_job(JobEvidence("failure", None, lines), clf.predict)
    assert (cat, source) == (INFRA, "classifier") and reason.startswith("clf:")
    # rules still take precedence when they match
    rule_hit = ["FAILED tests/test_a.py::test_b - assert 1 == 2"]
    assert classify_job(JobEvidence("failure", None, rule_hit), clf.predict)[1] == "log"
    assert (
        classify_build(
            "failure", False, [JobEvidence("failure", None, lines)], clf.predict
        ).failure_type
        == INFRA
    )


def test_missing_model_file_means_rules_only(tmp_path):
    assert LogClassifier.load(tmp_path / "nope.joblib") is None
