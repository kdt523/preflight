"""Conformal abstention via MAPIE: when the prediction set is {pass, fail} (or empty), the
check says "not enough signal" instead of spending trust on a coin flip."""

import numpy as np
import pandas as pd
from mapie.classification import SplitConformalClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

PASS, FAIL = "pass", "fail"


class _Prefit(ClassifierMixin, BaseEstimator):
    """Presents a fitted Preflight model to MAPIE as a prefit sklearn classifier."""

    def __init__(self, model=None) -> None:
        self.model = model

    def fit(self, X, y):  # never called with prefit=True
        return self

    def predict_proba(self, X) -> np.ndarray:
        p = self.model.predict_proba(X)
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.model.predict_proba(X) >= 0.5).astype(int)

    def __sklearn_is_fitted__(self) -> bool:
        return True

    @property
    def classes_(self) -> np.ndarray:
        return np.array([0, 1])


class ConformalAbstainer:
    def __init__(self, confidence: float = 0.9) -> None:
        self.confidence = confidence
        self.mapie_: SplitConformalClassifier | None = None

    def fit(self, model, X_conf: pd.DataFrame, y_conf: np.ndarray) -> "ConformalAbstainer":
        y_conf = np.asarray(y_conf, int)
        if len(X_conf) < 20 or len(set(y_conf)) < 2:
            self.mapie_ = None  # too little calibration data: never abstain rather than guess
            return self
        self.mapie_ = SplitConformalClassifier(
            estimator=_Prefit(model),
            confidence_level=self.confidence,
            conformity_score="lac",
            prefit=True,
        ).conformalize(X_conf, y_conf)
        return self

    def sets(self, X: pd.DataFrame) -> list[list[str] | None]:
        """Prediction set per row; None when no conformal calibration was possible."""
        if self.mapie_ is None:
            return [None] * len(X)
        _, s = self.mapie_.predict_set(X)
        s = s[:, :, 0]
        return [[lbl for lbl, on in ((PASS, row[0]), (FAIL, row[1])) if on] for row in s]

    @staticmethod
    def abstains(pred_set: list[str] | None) -> bool:
        return pred_set is not None and len(pred_set) != 1
