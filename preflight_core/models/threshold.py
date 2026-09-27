"""Cost-based operating point and risk bands.

A false alarm costs trust (developers stop reading the check), so it is priced higher than a
miss by default. The "high" band starts at the cost-optimal threshold for the repo.
"""

from dataclasses import dataclass

import numpy as np

DEFAULT_COST_FP = 3.0
DEFAULT_COST_FN = 1.0
DEFAULT_LOW, DEFAULT_HIGH = 0.25, 0.55
MIN_POSITIVES_PER_REPO = 30


def expected_cost(y: np.ndarray, p: np.ndarray, t: float, c_fp: float, c_fn: float) -> float:
    flag = p >= t
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    return c_fp * fp + c_fn * fn


def cost_threshold(
    y: np.ndarray, p: np.ndarray, c_fp: float = DEFAULT_COST_FP, c_fn: float = DEFAULT_COST_FN
) -> float:
    """Threshold in [0.05, 0.95] minimising c_fp*FP + c_fn*FN (ties -> higher threshold)."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    costs = [expected_cost(y, p, t, c_fp, c_fn) for t in grid]
    best = min(costs)
    return float(max(t for t, c in zip(grid, costs, strict=True) if c == best))


@dataclass(frozen=True)
class Bands:
    low: float = DEFAULT_LOW  # p < low -> "low"
    high: float = DEFAULT_HIGH  # p >= high -> "high"; between -> "watch"

    def band(self, p: float) -> str:
        return "high" if p >= self.high else "low" if p < self.low else "watch"


def bands_from_threshold(t: float) -> Bands:
    """High band at the cost threshold; low band keeps the default 0.25/0.55 proportion."""
    high = float(np.clip(t, 0.2, 0.95))
    return Bands(low=round(high * DEFAULT_LOW / DEFAULT_HIGH, 3), high=round(high, 3))


def tune_bands(
    y: np.ndarray,
    p: np.ndarray,
    repos: np.ndarray,
    c_fp: float = DEFAULT_COST_FP,
    c_fn: float = DEFAULT_COST_FN,
) -> tuple[Bands, dict[str, Bands]]:
    """Global bands + per-repo bands where the repo has enough positives to tune on."""
    y, p, repos = np.asarray(y, int), np.asarray(p, float), np.asarray(repos)
    global_bands = bands_from_threshold(cost_threshold(y, p, c_fp, c_fn))
    per_repo = {}
    for r in np.unique(repos):
        m = repos == r
        if y[m].sum() >= MIN_POSITIVES_PER_REPO:
            per_repo[str(r)] = bands_from_threshold(cost_threshold(y[m], p[m], c_fp, c_fn))
    return global_bands, per_repo
