import logging
import warnings

import pytest

from preflight_core.features.dataset import build_frame
from preflight_core.models.train import train_bundle
from tests.synth import make_test_runs, make_units

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("lightgbm").setLevel(logging.ERROR)


@pytest.fixture(scope="session")
def synth_units():
    return make_units(2000, seed=42)


@pytest.fixture(scope="session")
def synth_bundle(synth_units):
    """A small trained bundle on SYNTHETIC data, for serving tests only."""
    df = build_frame(synth_units, {u.key for u in synth_units}, repo_id=1, repo="acme/app")
    return train_bundle(df, make_test_runs(200), n_folds=2, per_repo_min=10**9, version="vsynth")
