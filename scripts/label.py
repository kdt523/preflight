"""Recompute taxonomy labels for every stored build (no GitHub calls).

uv run python scripts/label.py
"""

from preflight_core.db.engine import get_engine
from preflight_core.labeling.labeler import relabel_all
from preflight_core.labeling.log_classifier import LogClassifier
from preflight_core.logging import setup_logging
from preflight_core.storage.s3 import get_raw_store

if __name__ == "__main__":
    setup_logging()
    # Trained by scripts/label_quality.py from hand labels; rules-only until it exists.
    clf = LogClassifier.load("models/log_classifier.joblib")
    print("log classifier:", "on" if clf else "off (rules only)")
    for repo, counts in relabel_all(get_engine(), get_raw_store(), clf and clf.predict).items():
        print(f"{repo:42} {dict(counts)}")
