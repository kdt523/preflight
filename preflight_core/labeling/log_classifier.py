"""TF-IDF + logistic regression over redacted log tails (SPEC §4).

Rules label what they can; this model handles the rest ("unexplained" failures). It is
trained only on hand labels, and rule precision is measured against the same hand labels.
"""

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from preflight_core.labeling.rules import CLASSES_FOR_LOGS, JobEvidence, classify_job

MIN_CONFIDENCE = 0.7


def _normalise(text: str) -> str:
    # Numbers, hashes and paths vary per run; collapse them so tokens generalise.
    text = re.sub(r"[0-9a-f]{7,40}", " HASH ", text)
    text = re.sub(r"\d+(\.\d+)?", " NUM ", text)
    text = re.sub(r"(/[\w.\-]+){2,}", " PATH ", text)
    return text


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=_normalise,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50_000,
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b[A-Za-z_][A-Za-z_]+\b",
                ),
            ),
            ("lr", LogisticRegression(class_weight="balanced", max_iter=2000, C=4.0)),
        ]
    )


def rule_predictions(df: pd.DataFrame) -> list[str | None]:
    return [
        classify_job(JobEvidence("failure", step or None, str(log).splitlines()))[0]
        for log, step in zip(df["log_tail"], df.get("failed_step", [None] * len(df)), strict=True)
    ]


def label_quality(df: pd.DataFrame, n_splits: int = 5) -> dict:
    """df: hand-labelled rows with `log_tail`, `human_label` (and optional `failed_step`)."""
    df = df[df["human_label"].isin(CLASSES_FOR_LOGS)].reset_index(drop=True)
    y = df["human_label"].to_numpy()
    rules = rule_predictions(df)
    covered = [r is not None for r in rules]
    rules_arr = np.array([r or "unknown" for r in rules])

    per_class = {}
    for cls in CLASSES_FOR_LOGS:
        predicted = rules_arr == cls
        per_class[cls] = {
            "rule_predicted": int(predicted.sum()),
            "rule_precision": float((y[predicted] == cls).mean()) if predicted.any() else None,
            "rule_recall": float((rules_arr[y == cls] == cls).mean()) if (y == cls).any() else None,
        }
    out = {
        "n": len(df),
        "rule_coverage": float(np.mean(covered)),
        "rule_accuracy_when_covered": float((rules_arr[covered] == y[covered]).mean())
        if any(covered)
        else None,
        "per_class": per_class,
    }
    counts = pd.Series(y).value_counts()
    folds = min(n_splits, int(counts.min())) if len(counts) > 1 else 0
    if folds >= 2:
        pred = cross_val_predict(
            make_pipeline(),
            df["log_tail"].astype(str),
            y,
            cv=StratifiedKFold(folds, shuffle=True, random_state=0),
        )
        out["classifier_cv_report"] = classification_report(
            y, pred, output_dict=True, zero_division=0
        )
        out["classifier_confusion"] = {
            "labels": list(CLASSES_FOR_LOGS),
            "matrix": confusion_matrix(y, pred, labels=list(CLASSES_FOR_LOGS)).tolist(),
        }
    return out


def train(df: pd.DataFrame) -> Pipeline:
    df = df[df["human_label"].isin(CLASSES_FOR_LOGS)]
    return make_pipeline().fit(df["log_tail"].astype(str), df["human_label"])


class LogClassifier:
    def __init__(self, pipe: Pipeline) -> None:
        self.pipe = pipe

    @classmethod
    def load(cls, path: str | Path) -> "LogClassifier | None":
        return cls(joblib.load(path)) if Path(path).exists() else None

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipe, path)

    def predict(self, log_lines: list[str]) -> tuple[str, float] | None:
        """-> (class, probability) if confident enough, else None."""
        proba = self.pipe.predict_proba(["\n".join(log_lines)])[0]
        i = int(np.argmax(proba))
        return (self.pipe.classes_[i], float(proba[i])) if proba[i] >= MIN_CONFIDENCE else None
