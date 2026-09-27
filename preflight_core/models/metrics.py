"""Evaluation metrics. PR-AUC is primary; accuracy is deliberately absent."""

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p))


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan")


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(brier_score_loss(y, p))


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    """Equal-width bins: mean predicted vs observed rate (the reliability curve)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    out = []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            out.append(
                {
                    "bin": b,
                    "n": int(m.sum()),
                    "predicted": float(p[m].mean()),
                    "observed": float(y[m].mean()),
                }
            )
    return out


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error: bin-size-weighted |observed - predicted|."""
    n = len(y)
    return float(
        sum(r["n"] / n * abs(r["observed"] - r["predicted"]) for r in reliability(y, p, n_bins))
    )


def precision_recall_at(y: np.ndarray, p: np.ndarray, threshold: float) -> tuple[float, float]:
    y, flagged = np.asarray(y).astype(bool), np.asarray(p) >= threshold
    tp = int((flagged & y).sum())
    precision = tp / flagged.sum() if flagged.sum() else float("nan")
    recall = tp / y.sum() if y.sum() else float("nan")
    return float(precision), float(recall)


def apfd(order: Sequence[str], failing: set[str]) -> float:
    """Average Percentage of Faults Detected (each failing test detects itself).

    APFD = 1 - sum(position of each failing test) / (n * m) + 1 / (2n), positions 1-based.
    1.0 ~ failures first; ~0.5 random; lower = failures last.
    """
    n, m = len(order), len(failing)
    if n == 0 or m == 0:
        return float("nan")
    positions = [i + 1 for i, t in enumerate(order) if t in failing]
    return 1 - sum(positions) / (n * m) + 1 / (2 * n)


def summarize(y: np.ndarray, p: np.ndarray, threshold: float | None = None) -> dict[str, float]:
    y, p = np.asarray(y, int), np.asarray(p, float)
    out = {
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "pr_auc": pr_auc(y, p),
        "roc_auc": roc_auc(y, p),
        "brier": brier(y, p),
        "ece": ece(y, p),
    }
    if threshold is not None:
        out["threshold"] = threshold
        out["precision_at_t"], out["recall_at_t"] = precision_recall_at(y, p, threshold)
    return out
