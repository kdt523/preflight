"""Build the point-in-time feature table for every repo -> Postgres `features` + Parquet.

uv run python scripts/build_features.py
"""

import pandas as pd
from sqlalchemy import text

from preflight_core.db.engine import get_engine
from preflight_core.features.dataset import (
    build_frame,
    dataset_path,
    load_units,
    save_features,
    validate,
    write_parquet,
)
from preflight_core.storage.s3 import get_raw_store


def main() -> None:
    engine = get_engine()
    with engine.connect() as c:
        repos = c.execute(text("SELECT id, full_name, default_branch FROM repos ORDER BY id")).all()
    frames = []
    for r in repos:
        units, with_diff = load_units(engine, r.id, r.default_branch)
        df = build_frame(units, with_diff, r.id, r.full_name)
        validate(df)
        save_features(engine, df)
        frames.append(df)
        labelled = df["label"].notna().sum()
        print(
            f"{r.full_name:40} units={len(units):6d} rows={len(df):6d} labelled={labelled:6d} "
            f"fail_rate={df['label'].mean():.3f}"
            if labelled
            else f"{r.full_name:40} no labels"
        )
    if not frames:
        raise SystemExit("no repos in the database; run the backfill first")
    all_df = pd.concat(frames, ignore_index=True)
    write_parquet(all_df, dataset_path(), get_raw_store())
    print(f"{len(all_df)} rows -> {dataset_path()} (+ s3 datasets/)")


if __name__ == "__main__":
    main()
