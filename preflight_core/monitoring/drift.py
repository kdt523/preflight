"""Population Stability Index for feature drift (reference = training distribution).

PSI = sum((cur - ref) * ln(cur / ref)) over quantile bins of the reference.
Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant shift.
"""

import math
from typing import Any

import numpy as np
import pandas as pd

N_BINS = 10
EPS = 1e-4
PSI_ALERT = 0.25


def reference(X: pd.DataFrame, n_bins: int = N_BINS) -> dict[str, dict[str, Any]]:
    ref = {}
    for col in X.columns:
        v = X[col].to_numpy(float)
        finite = v[np.isfinite(v)]
        if len(finite) == 0:
            continue
        edges = np.unique(np.quantile(finite, np.linspace(0, 1, n_bins + 1)[1:-1]))
        ref[col] = {"edges": edges.tolist(), "props": _props(v, edges).tolist()}
    return ref


def _props(v: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Share per bin, with an extra last bin for missing values."""
    missing = ~np.isfinite(v)
    idx = np.searchsorted(edges, v[~missing], side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    counts = np.append(counts, missing.sum())
    return counts / max(1.0, counts.sum())


def psi(ref: dict[str, Any], values: np.ndarray) -> float:
    edges = np.asarray(ref["edges"])
    r = np.clip(np.asarray(ref["props"]), EPS, None)
    c = np.clip(_props(np.asarray(values, float), edges), EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_table(reference_: dict[str, dict[str, Any]], X: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"feature": col, "psi": psi(ref, X[col].to_numpy(float))}
        for col, ref in reference_.items()
        if col in X.columns and len(X)
    ]
    df = pd.DataFrame(rows, columns=["feature", "psi"]).sort_values("psi", ascending=False)
    df["status"] = df["psi"].map(
        lambda p: "shift" if p > PSI_ALERT else "moderate" if p > 0.1 else "stable"
    )
    return df.reset_index(drop=True)


def psi_max(table: pd.DataFrame) -> float:
    return float(table["psi"].max()) if len(table) else 0.0


def is_finite(x: float) -> bool:
    return x is not None and not math.isnan(x)
