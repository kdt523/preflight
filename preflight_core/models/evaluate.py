"""Cross-validated comparison of models, baselines and feature-family ablations."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preflight_core.features.definitions import FAMILIES, feature_names
from preflight_core.models.cv import rolling_origin_splits
from preflight_core.models.metrics import pr_auc, reliability, summarize

ModelFactory = Callable[[list[str]], Any]  # feature list -> unfitted model


def training_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["label"].notna()].copy()
    out["label"] = out["label"].astype(int)
    return out.sort_values("as_of").reset_index(drop=True)


@dataclass
class CVResult:
    name: str
    folds: list[dict[str, Any]] = field(default_factory=list)
    oof: pd.DataFrame | None = None

    def summary(self) -> dict[str, Any]:
        if not self.folds or self.oof is None:
            return {"model": self.name, "folds": 0}
        aucs = [f["pr_auc"] for f in self.folds]
        pooled = summarize(self.oof["y"].to_numpy(), self.oof["p"].to_numpy())
        return {
            "model": self.name,
            "folds": len(self.folds),
            "pr_auc_mean": float(np.mean(aucs)),
            "pr_auc_std": float(np.std(aucs)),
            "base_rate_mean": float(np.mean([f["base_rate"] for f in self.folds])),
            "lift": float(np.mean(aucs) / max(1e-9, np.mean([f["base_rate"] for f in self.folds]))),
            **{f"pooled_{k}": v for k, v in pooled.items()},
        }


def cross_validate(
    name: str, factory: ModelFactory, df: pd.DataFrame, features: list[str], n_folds: int = 5
) -> CVResult:
    rows = training_rows(df)
    res = CVResult(name)
    oof = []
    for fold in rolling_origin_splits(rows, n_folds=n_folds):
        tr, te = rows.iloc[fold.train_idx], rows.iloc[fold.test_idx]
        model = factory(features).fit(tr[features], tr["label"].to_numpy())
        p = model.predict_proba(te[features])
        m = summarize(te["label"].to_numpy(), p)
        res.folds.append(
            {"fold": fold.k, "train_n": len(tr), "test_start": str(fold.test_start), **m}
        )
        oof.append(
            pd.DataFrame(
                {
                    "sha": te["sha"].to_numpy(),
                    "repo": te["repo"].to_numpy(),
                    "y": te["label"].to_numpy(),
                    "p": p,
                    "fold": fold.k,
                }
            )
        )
    res.oof = pd.concat(oof, ignore_index=True) if oof else None
    return res


def compare(
    models: dict[str, ModelFactory],
    df: pd.DataFrame,
    features: list[str] | None = None,
    n_folds: int = 5,
) -> tuple[pd.DataFrame, dict[str, CVResult]]:
    features = features or feature_names()
    results = {n: cross_validate(n, f, df, features, n_folds) for n, f in models.items()}
    return pd.DataFrame([r.summary() for r in results.values()]), results


def ablate_families(
    factory: ModelFactory, df: pd.DataFrame, n_folds: int = 5, families=FAMILIES
) -> pd.DataFrame:
    """PR-AUC with each feature family removed, vs the full model."""
    full = cross_validate("all", factory, df, feature_names(), n_folds).summary()
    rows = [{"dropped": "(none)", "pr_auc_mean": full.get("pr_auc_mean"), "delta": 0.0}]
    for fam in families:
        s = cross_validate(f"-{fam}", factory, df, feature_names({fam}), n_folds).summary()
        rows.append(
            {
                "dropped": fam,
                "pr_auc_mean": s.get("pr_auc_mean"),
                "delta": (s.get("pr_auc_mean") or np.nan) - (full.get("pr_auc_mean") or np.nan),
            }
        )
    return pd.DataFrame(rows)


def per_repo(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for repo, g in oof.groupby("repo"):
        if g["y"].sum() and g["y"].sum() < len(g):
            rows.append(
                {
                    "repo": repo,
                    "n": len(g),
                    "base_rate": g["y"].mean(),
                    "pr_auc": pr_auc(g["y"].to_numpy(), g["p"].to_numpy()),
                }
            )
    return pd.DataFrame(rows)


def write_report(sections: dict[str, Any], out_dir: str = "reports") -> Path:
    """JSON (machine) + Markdown (human). Every number comes from the run that wrote it."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    js, md = Path(out_dir) / f"eval_{stamp}.json", Path(out_dir) / f"eval_{stamp}.md"

    def enc(v: Any) -> Any:
        return v.to_dict(orient="records") if isinstance(v, pd.DataFrame) else v

    js.write_text(json.dumps({k: enc(v) for k, v in sections.items()}, indent=2, default=str))
    lines = [f"# Preflight evaluation ({stamp})", ""]
    for title, v in sections.items():
        lines.append(f"## {title}")
        if isinstance(v, pd.DataFrame):
            lines.append(v.to_markdown(index=False, floatfmt=".4f") if len(v) else "_empty_")
        else:
            lines.append(f"```\n{json.dumps(v, indent=2, default=str)}\n```")
        lines.append("")
    md.write_text("\n".join(lines))
    return md


def reliability_table(oof: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(reliability(oof["y"].to_numpy(), oof["p"].to_numpy()))
