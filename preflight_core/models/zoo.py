"""Named model factories shared by evaluation, training and scripts."""

from collections.abc import Callable
from typing import Any

from preflight_core.models.baselines import AlwaysPass, FilesChangedHeuristic, LogisticBaseline

MODELS: dict[str, Callable[[list[str]], Any]] = {
    "always_pass": lambda f: AlwaysPass(),
    "files_changed_gt_n": lambda f: FilesChangedHeuristic(),
    "logistic_regression": lambda f: LogisticBaseline(),
}
BASELINES = ["always_pass", "files_changed_gt_n", "logistic_regression"]

# Imported late to keep baselines importable without LightGBM.
from preflight_core.models.lgbm import CalibratedLGBM  # noqa: E402

MODELS |= {
    "lgbm_calibrated": lambda f: CalibratedLGBM(f),
    # ablations of the modelling decisions (SPEC: SMOTE only as ablation)
    "lgbm_smote": lambda f: CalibratedLGBM(f, smote=True),
    "lgbm_no_monotone": lambda f: CalibratedLGBM(f, monotone=False),
    "lgbm_full_class_weight": lambda f: CalibratedLGBM(f, class_weight_power=1.0),
    "lgbm_no_class_weight": lambda f: CalibratedLGBM(f, class_weight=False),
}
