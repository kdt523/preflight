"""Rolling-origin temporal cross-validation. Random K-fold is never used (SPEC).

Fold k tests on the time slice [t_k, t_k+1) and trains on rows whose *label was known*
before t_k (completed_at < t_k). Filtering on completion, not start, is the embargo: a
build that started before t_k but finished after it was not a usable label at t_k.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    k: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def rolling_origin_splits(
    df: pd.DataFrame, n_folds: int = 5, min_train_frac: float = 0.5, min_train_rows: int = 50
) -> Iterator[Fold]:
    """df: labelled rows with `as_of` and `completed_at`. Yields positional indices."""
    order = np.argsort(df["as_of"].to_numpy(), kind="stable")
    as_of = df["as_of"].to_numpy()[order]
    done = df["completed_at"].to_numpy()
    y = df["label"].to_numpy()
    n = len(df)
    start = int(n * min_train_frac)
    edges = np.linspace(start, n, n_folds + 1).astype(int)
    for k in range(n_folds):
        lo, hi = edges[k], edges[k + 1]
        if hi <= lo:
            continue
        t0 = as_of[lo]
        t1 = as_of[hi - 1]
        test_idx = order[lo:hi]
        train_idx = np.flatnonzero(done < t0)
        if len(train_idx) < min_train_rows or len(set(y[train_idx])) < 2:
            continue
        if y[test_idx].sum() == 0:  # PR-AUC undefined without positives
            continue
        yield Fold(k, train_idx, test_idx, pd.Timestamp(t0), pd.Timestamp(t1))
