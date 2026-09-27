"""MLflow model registry backend (same interface as LocalRegistry).

Each Preflight bundle version is an MLflow model version whose source is the run's `bundle`
artifact. Pointers are registered-model aliases:
  champion  served            shadow  scored, hidden            previous  rollback target
"""

from pathlib import Path
from typing import Any

from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException

ALIASES = ("champion", "shadow", "previous")


class MlflowRegistry:
    def __init__(
        self,
        model_name: str = "preflight",
        cache_root: str | Path = "models",
        client: MlflowClient | None = None,
    ) -> None:
        self.name = model_name
        self.cache = Path(cache_root) / "mlflow_cache"
        self.c = client or MlflowClient()

    # -- helpers -------------------------------------------------------------------
    def _ensure_model(self) -> None:
        try:
            self.c.get_registered_model(self.name)
        except MlflowException:
            self.c.create_registered_model(self.name, description="Preflight CI-failure bundles")

    def _alias(self, alias: str):
        try:
            return self.c.get_model_version_by_alias(self.name, alias)
        except MlflowException:
            return None

    def _mv_for(self, version: str):
        for mv in self.c.search_model_versions(f"name='{self.name}'"):
            if mv.tags.get("preflight_version") == version:
                return mv
        raise KeyError(f"{version} is not registered")

    # -- Registry interface ------------------------------------------------------------
    def read(self) -> dict[str, Any]:
        out: dict[str, Any] = {"history": []}
        for a in ALIASES:
            mv = self._alias(a)
            out[a] = mv.tags.get("preflight_version") if mv else None
        return out

    def add_candidate(self, version: str, run_id: str | None = None) -> None:
        self._ensure_model()
        if run_id is None:
            runs = self.c.search_runs(
                experiment_ids=[e.experiment_id for e in self.c.search_experiments()],
                filter_string=f"tags.version = '{version}'",
                max_results=1,
            )
            if not runs:
                raise KeyError(f"no MLflow run tagged version={version}")
            run_id = runs[0].info.run_id
        self.c.create_model_version(
            self.name,
            f"runs:/{run_id}/bundle",
            run_id=run_id,
            tags={"preflight_version": version, "status": "candidate"},
        )

    def set_shadow(self, version: str | None) -> None:
        if version is None:
            if self._alias("shadow"):
                self.c.delete_registered_model_alias(self.name, "shadow")
            return
        self.c.set_registered_model_alias(self.name, "shadow", self._mv_for(version).version)

    def promote(self, version: str) -> None:
        mv = self._mv_for(version)
        cur = self._alias("champion")
        if cur and cur.version != mv.version:
            self.c.set_registered_model_alias(self.name, "previous", cur.version)
        self.c.set_registered_model_alias(self.name, "champion", mv.version)
        shadow = self._alias("shadow")
        if shadow and shadow.version == mv.version:
            self.c.delete_registered_model_alias(self.name, "shadow")

    def rollback(self) -> str:
        prev, cur = self._alias("previous"), self._alias("champion")
        if prev is None:
            raise RuntimeError("no previous champion to roll back to")
        self.c.set_registered_model_alias(self.name, "champion", prev.version)
        if cur:
            self.c.set_model_version_tag(self.name, cur.version, "status", "rolled_back")
        self.c.delete_registered_model_alias(self.name, "previous")
        return prev.tags["preflight_version"]

    def bundle_path(self, version: str) -> Path:
        dst = self.cache / version
        if not (dst / "bundle" / "bundle.joblib").exists():
            dst.mkdir(parents=True, exist_ok=True)
            download_artifacts(artifact_uri=self._mv_for(version).source, dst_path=str(dst))
        return dst / "bundle"
