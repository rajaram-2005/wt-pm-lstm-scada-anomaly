"""BaseWTModel + ModelRegistry — the integration boundary for all 25 repos.

Each of the 25 wt-pm repositories is wrapped by exactly one adapter class
implementing :class:`BaseWTModel`. The adapter:

* imports the repository's *actual* code (from ``external/<repo>/model.py`` or
  the installed ``wt_pm_lstm`` package) — it never re-implements the model;
* converts a platform :class:`SensorBatch` into the repo's expected input;
* converts the repo's raw output into a :class:`ModelOutput`;
* declares its capabilities so the ModelRouter can select it honestly.

If the repository's dependency stack is missing (e.g. TensorFlow on an edge
box), the adapter reports itself unavailable instead of crashing the platform.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from wtpm_platform.contracts import (
    Deployment, ModelOutput, OperatingContext, SensorBatch, Stopwatch, TaskType,
)

#: where the 24 sibling repos are cloned (kept untouched, read-only)
EXTERNAL_DIR = os.environ.get(
    "WTPM_EXTERNAL_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external"),
)


def load_repo_module(repo: str, module_name: Optional[str] = None):
    """Import ``external/<repo>/model.py`` under an isolated module name.

    This is how adapters call the original research code without modifying it
    and without the 24 identical ``model.py`` filenames colliding.
    """
    # Resolve at call time so installed wheels and configured deployments agree.
    root = os.environ.get("WTPM_EXTERNAL_DIR", EXTERNAL_DIR)
    path = os.path.join(root, repo, "model.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{repo}: model.py not found at {path} (clone the repo into external/)")
    name = module_name or ("wtpm_ext_" + repo.replace("-", "_"))
    if name in sys.modules and getattr(sys.modules[name], "__file__", None) == path:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod


@dataclass
class ModelSpec:
    """Registry row: model_id -> repository -> task -> requirements -> ..."""

    model_id: str
    repository: str
    task: TaskType
    input_requirements: Sequence[str]      # which SensorBatch views it needs
    output_schema: Sequence[str]           # which ModelOutput fields it fills
    resource_requirements: Sequence[str]   # python deps
    typical_latency_ms: float
    deployment_targets: Sequence[Deployment]
    fallback: Optional[str] = None         # model_id to fall back to
    subsystem_focus: Sequence[str] = ()
    notes: str = ""


class BaseWTModel(ABC):
    """The uniform interface every adapter implements."""

    spec: ModelSpec

    def __init__(self) -> None:
        self._fitted = False
        self._unavailable_reason = ""
        self._fit_error = ""
        self._fit_status = "not fitted"

    # -- capability ---------------------------------------------------------
    def available(self) -> bool:
        """Whether the underlying repo's dependencies import in this env."""
        if self._unavailable_reason:
            return False
        try:
            self._check_deps()
            return True
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            self._unavailable_reason = f"{type(exc).__name__}: {exc}"
            return False

    def _check_deps(self) -> None:
        """Override: raise ImportError if the repo cannot run here."""

    @property
    def fitted(self) -> bool:
        return self._fitted

    # -- lifecycle ------------------------------------------------------------
    @abstractmethod
    def fit(self, batch: SensorBatch, train_mask: np.ndarray) -> None:
        """Train/calibrate on the healthy band of ``batch`` (rows where
        ``train_mask`` is True). Adapters that wrap non-trainable code
        (e.g. the particle filter) may no-op."""

    @abstractmethod
    def _predict(self, batch: SensorBatch) -> ModelOutput:
        """Raw prediction. Called only when available() and fitted."""

    # -- uniform, guarded entry point -----------------------------------------
    def predict(self, batch: SensorBatch) -> ModelOutput:
        mid, task = self.spec.model_id, self.spec.task
        if not self.available():
            return ModelOutput.failed(mid, task, batch.turbine_id,
                                      f"unavailable: {self._unavailable_reason}")
        if not self._fitted:
            return ModelOutput.failed(mid, task, batch.turbine_id, "not fitted")
        try:
            with Stopwatch() as sw:
                out = self._predict(batch)
            out.inference_time_ms = sw.ms
            return out
        except Exception:  # noqa: BLE001
            return ModelOutput.failed(mid, task, batch.turbine_id,
                                      traceback.format_exc(limit=2))

    def health(self) -> Dict[str, Any]:
        return {
            "model_id": self.spec.model_id,
            "repository": self.spec.repository,
            "available": self.available(),
            "fitted": self._fitted,
            "reason": self._unavailable_reason or self._fit_error,
            "fit_status": self._fit_status,
            "fit_error": self._fit_error,
            "notes": self.spec.notes,
        }


class ModelRegistry:
    """model_id -> adapter instance + spec; the orchestrator's phone book."""

    def __init__(self) -> None:
        self._models: Dict[str, BaseWTModel] = {}

    def register(self, model: BaseWTModel) -> None:
        self._models[model.spec.model_id] = model

    def get(self, model_id: str) -> BaseWTModel:
        return self._models[model_id]

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._models

    def ids(self) -> List[str]:
        return sorted(self._models)

    def by_task(self, task: TaskType) -> List[BaseWTModel]:
        return [m for m in self._models.values() if m.spec.task == task]

    def specs(self) -> List[ModelSpec]:
        return [self._models[k].spec for k in self.ids()]

    def health_report(self) -> List[Dict[str, Any]]:
        return [self._models[k].health() for k in self.ids()]

    def resolve_fallback(self, model_id: str) -> Optional[BaseWTModel]:
        fb = self._models[model_id].spec.fallback
        return self._models.get(fb) if fb else None
