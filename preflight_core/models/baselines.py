"""Baselines every model must beat: always-pass, files-changed > N, logistic regression.

All models share one tiny interface: fit(X, y) -> self, predict_proba(X) -> P(fail) 1-D.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


class AlwaysPass:
    """Constant score = training base rate. Its PR-AUC equals the test base rate."""

    name = "always_pass"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "AlwaysPass":
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.rate_)


class FilesChangedHeuristic:
    """Flag when n_files > N. N is chosen on training data to maximise F1;
    each side of the cut predicts its training failure rate."""

    name = "files_changed_gt_n"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "FilesChangedHeuristic":
        n = X["n_files"].to_numpy()
        y = np.asarray(y)
        best = (-1.0, 0)
        for cut in np.unique(np.quantile(n, np.linspace(0.05, 0.95, 19)).round()):
            flag = n > cut
            tp = (flag & (y == 1)).sum()
            if not flag.any() or tp == 0:
                continue
            prec, rec = tp / flag.sum(), tp / max(1, y.sum())
            f1 = 2 * prec * rec / (prec + rec)
            best = max(best, (f1, cut))
        self.n_ = best[1]
        flag = n > self.n_
        overall = y.mean()
        self.hi_ = y[flag].mean() if flag.any() else overall
        self.lo_ = y[~flag].mean() if (~flag).any() else overall
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.where(X["n_files"].to_numpy() > self.n_, self.hi_, self.lo_)


def _signed_log1p(X: np.ndarray) -> np.ndarray:
    return np.sign(X) * np.log1p(np.abs(X))


class LogisticBaseline:
    """log1p -> median impute (+ missing indicators) -> standardise -> L2 logistic regression.
    No class weights, so probabilities stay close to calibrated."""

    name = "logistic_regression"

    def __init__(self, C: float = 1.0) -> None:
        self.C = C

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LogisticBaseline":
        self.columns_ = list(X.columns)
        self.pipe_ = make_pipeline(
            FunctionTransformer(_signed_log1p),
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(C=self.C, max_iter=2000),
        )
        self.pipe_.fit(X[self.columns_].to_numpy(float), np.asarray(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipe_.predict_proba(X[self.columns_].to_numpy(float))[:, 1]
