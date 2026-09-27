"""LightGBM with monotone constraints, class weights, isotonic calibration (SPEC §non-negotiables).

Rows must arrive in time order (training_rows sorts by as_of). The newest slice is held out:
  [ ---- trees (80%, early-stops on its newest 10%) ---- | isotonic 10% | conformal 10% ]
so calibration and conformal sets are fitted on data the trees never saw, and on the most
recent behaviour, which is what the model faces next.

Isotonic regression is a step function, so it ties many scores and throws away ranking
(measured: about -0.06 PR-AUC on the synthetic benchmark). A 1e-6 raw-score tie-break keeps
the calibrated value and the model's ordering.

Class weights use scale_pos_weight = (neg/pos) ** class_weight_power. Full weighting (power 1)
cost ranking and calibration in experiments; the square root (0.5) is the default.
"""

import numpy as np
import pandas as pd
from lightgbm import Booster, LGBMClassifier, early_stopping
from sklearn.isotonic import IsotonicRegression

from preflight_core.features.definitions import monotone_constraints

DEFAULT_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "n_estimators": 1000,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "monotone_constraints_method": "advanced",
    "verbose": -1,
    "deterministic": True,
    "force_row_wise": True,
}


class CalibratedLGBM:
    name = "lgbm_calibrated"

    def __init__(
        self,
        features: list[str],
        monotone: bool = True,
        class_weight: bool = True,
        class_weight_power: float = 0.5,
        smote: bool = False,
        holdout_frac: float = 0.2,
        seed: int = 0,
        params: dict | None = None,
    ) -> None:
        self.features = list(features)
        self.monotone = monotone
        self.class_weight = class_weight and not smote
        self.class_weight_power = class_weight_power
        self.smote = smote
        self.holdout_frac = holdout_frac
        self.seed = seed
        self.params = {**DEFAULT_PARAMS, **(params or {})}

    # -- fitting -------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "CalibratedLGBM":
        X = X[self.features].astype(float)
        y = np.asarray(y).astype(int)
        n = len(X)
        n_hold = max(int(n * self.holdout_frac), min(40, n // 3))
        n_trees = n - n_hold
        X_tr, y_tr = X.iloc[:n_trees], y[:n_trees]
        X_cal, y_cal = X.iloc[n_trees:], y[n_trees:]
        half = len(X_cal) // 2
        self.X_conf_, self.y_conf_ = X_cal.iloc[half:], y_cal[half:]
        X_iso, y_iso = X_cal.iloc[:half], y_cal[:half]

        params = dict(self.params, random_state=self.seed)
        if self.monotone:
            params["monotone_constraints"] = monotone_constraints(self.features)
        if self.class_weight:
            pos = max(1, y_tr.sum())
            params["scale_pos_weight"] = ((len(y_tr) - pos) / pos) ** self.class_weight_power
        X_fit, y_fit = X_tr, y_tr
        if self.smote:
            X_fit, y_fit = _smote(X_tr, y_tr, self.seed)

        self.clf_ = LGBMClassifier(**params)
        n_es = int(len(X_fit) * 0.1) if not self.smote else 0
        if n_es >= 50 and len(set(y_fit[-n_es:])) == 2:
            self.clf_.fit(
                X_fit.iloc[:-n_es],
                y_fit[:-n_es],
                eval_set=[(X_fit.iloc[-n_es:], y_fit[-n_es:])],
                callbacks=[early_stopping(50, verbose=False)],
            )
        else:
            self.clf_.set_params(n_estimators=200)
            self.clf_.fit(X_fit, y_fit)

        raw = self._raw(X_iso)
        self.iso_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        if len(set(y_iso)) == 2:
            self.iso_.fit(raw, y_iso)
        else:  # degenerate slice: fall back to the uncalibrated sigmoid
            self.iso_ = None
        self.train_window_ = (len(X_tr), len(X_iso), len(self.X_conf_))
        return self

    # -- scoring -------------------------------------------------------------------

    @property
    def booster(self) -> Booster:
        return self.clf_.booster_

    def _raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(X[self.features].astype(float), raw_score=True)

    def predict_uncalibrated(self, X: pd.DataFrame) -> np.ndarray:
        return 1 / (1 + np.exp(-self._raw(X)))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self._raw(X)
        sig = 1 / (1 + np.exp(-raw))
        if self.iso_ is None:
            return sig
        return (1 - 1e-6) * self.iso_.predict(raw) + 1e-6 * sig  # tie-break, see top

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """Exact TreeSHAP values in log-odds space: (n, n_features + 1), last column = bias."""
        return self.booster.predict(X[self.features].astype(float), pred_contrib=True)


def _smote(X: pd.DataFrame, y: np.ndarray, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """SMOTE ablation only. Needs complete rows, so NaNs are median-filled first."""
    from imblearn.over_sampling import SMOTE

    filled = X.fillna(X.median(numeric_only=True)).fillna(0)
    k = max(1, min(5, int(y.sum()) - 1))
    Xs, ys = SMOTE(random_state=seed, k_neighbors=k).fit_resample(filled, y)
    return pd.DataFrame(Xs, columns=X.columns), np.asarray(ys)
