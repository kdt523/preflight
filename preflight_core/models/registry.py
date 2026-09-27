"""Model registry: which bundle is champion (served) and which is shadow (scored, hidden).

LocalRegistry keeps pointers in models/registry.json next to models/bundles/<version>/.
MlflowRegistry (week 5) implements the same interface on MLflow aliases. Rollback is one
call: the previous champion is restored.
"""

import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from preflight_core.models.bundle import PreflightBundle


class Registry(Protocol):
    def read(self) -> dict[str, Any]: ...
    def add_candidate(self, version: str) -> None: ...
    def set_shadow(self, version: str | None) -> None: ...
    def promote(self, version: str) -> None: ...
    def rollback(self) -> str: ...
    def bundle_path(self, version: str) -> Path: ...


class LocalRegistry:
    def __init__(self, root: str | Path = "models") -> None:
        self.root = Path(root)
        self.path = self.root / "registry.json"

    def read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"champion": None, "shadow": None, "history": []}
        return json.loads(self.path.read_text())

    def _write(self, d: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=2))
        tmp.replace(self.path)  # atomic swap: readers never see a half-written file

    def _event(self, d: dict[str, Any], version: str | None, action: str) -> None:
        d["history"].append(
            {"version": version, "action": action, "at": datetime.now(UTC).isoformat()}
        )

    def add_candidate(self, version: str) -> None:
        d = self.read()
        self._event(d, version, "candidate")
        self._write(d)

    def set_shadow(self, version: str | None) -> None:
        d = self.read()
        d["shadow"] = version
        self._event(d, version, "shadow")
        self._write(d)

    def promote(self, version: str) -> None:
        d = self.read()
        if not self.bundle_path(version).exists():
            raise FileNotFoundError(f"no bundle for {version}")
        if d["champion"] and d["champion"] != version:
            d.setdefault("previous", []).append(d["champion"])
        d["champion"] = version
        if d.get("shadow") == version:
            d["shadow"] = None
        self._event(d, version, "champion")
        self._write(d)

    def rollback(self) -> str:
        d = self.read()
        prev = d.get("previous") or []
        if not prev:
            raise RuntimeError("no previous champion to roll back to")
        bad = d["champion"]
        d["champion"] = prev.pop()
        self._event(d, bad, "rolled_back")
        self._event(d, d["champion"], "champion")
        self._write(d)
        return d["champion"]

    def bundle_path(self, version: str) -> Path:
        return self.root / "bundles" / version


class BundleProvider:
    """Serves champion/shadow bundles, re-reading registry pointers every `reload_s`."""

    def __init__(self, registry: Registry, reload_s: float = 30.0) -> None:
        self.registry = registry
        self.reload_s = reload_s
        self._cache: dict[str, PreflightBundle] = {}
        self._pointers: dict[str, Any] = {}
        self._checked = 0.0
        self._lock = threading.Lock()

    def _refresh(self) -> None:
        if time.monotonic() - self._checked < self.reload_s and self._pointers:
            return
        with self._lock:
            self._pointers = self.registry.read()
            self._checked = time.monotonic()
            wanted = {
                v for v in (self._pointers.get("champion"), self._pointers.get("shadow")) if v
            }
            for v in wanted - set(self._cache):
                self._cache[v] = PreflightBundle.load(self.registry.bundle_path(v))
            for v in set(self._cache) - wanted:
                del self._cache[v]

    def champion(self) -> PreflightBundle | None:
        self._refresh()
        v = self._pointers.get("champion")
        return self._cache.get(v) if v else None

    def shadow(self) -> PreflightBundle | None:
        self._refresh()
        v = self._pointers.get("shadow")
        return self._cache.get(v) if v else None

    def force_reload(self) -> None:
        self._checked = 0.0


def get_registry() -> Registry:
    from preflight_core.config import get_settings

    s = get_settings()
    if s.model_registry == "mlflow":
        from preflight_core.models.mlflow_registry import MlflowRegistry

        return MlflowRegistry(s.mlflow_model_name, s.model_root)
    return LocalRegistry(s.model_root)
