"""wtpm_platform — the unified WT-PM predictive-maintenance intelligence system.

25 specialized models (adapters over the original wt-pm-* repositories)
+ 1 common data layer   (contracts, data)
+ 1 orchestrator        (orchestrator.Orchestrator / ModelRouter / InferenceEngine)
+ 1 fusion engine       (fusion.FusionEngine)
+ 1 diagnosis engine    (engines.DiagnosisEngine)
+ 1 RUL engine          (engines.RULManager)
+ 1 XAI layer           (engines.XAIEngine, adapter m21)
+ 1 digital-twin layer  (engines.DigitalTwinInterface, adapter m19)
+ 1 edge/safety layer   (engines.EdgeInferenceManager/SafetyManager, m24/m25)
+ 1 monitoring/evaluation layer (evaluation.Evaluator/ExperimentTracker/drift)
"""

import os as _os

_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
_os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

# torch-geometric MUST be imported before tensorflow in this environment or
# the process segfaults (observed: tf->pyg crashes, pyg->tf is fine). Import
# it defensively up front; if absent, m20 falls back to the repo's linear net.
try:  # pragma: no cover - environment guard
    import torch_geometric as _pyg  # noqa: F401
except Exception:  # noqa: BLE001
    pass

from wtpm_platform.contracts import (  # noqa: F401
    ModelOutput, OperatingContext, SensorBatch, TaskType, WTDataSchema,
)
from wtpm_platform.base import BaseWTModel, ModelRegistry, ModelSpec  # noqa: F401
from wtpm_platform.data import FeaturePipeline, ingest_timeline  # noqa: F401
from wtpm_platform.fusion import FusionConfig, FusionEngine  # noqa: F401
from wtpm_platform.orchestrator import (  # noqa: F401
    InferenceEngine, ModelRouter, Orchestrator, build_default_registry,
)
from wtpm_platform.evaluation import Evaluator, ExperimentTracker  # noqa: F401

__version__ = "0.1.0"
