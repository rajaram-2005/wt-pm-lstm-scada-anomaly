"""WTDataSchema — the common data contract every model communicates through.

No model in the fleet talks to another model directly; everything crosses this
boundary as a ``ModelOutput`` (standardised prediction record) or a
``SensorBatch`` (standardised input). Incompatible inputs are never forced into
a model — each adapter converts from these types to whatever its upstream
repository expects.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

SCHEMA_VERSION = "wt-pm.platform.v1"


class TaskType(str, Enum):
    ANOMALY_DETECTION = "anomaly_detection"
    FAULT_CLASSIFICATION = "fault_classification"
    DEGRADATION_STATE = "degradation_state"
    RUL_ESTIMATION = "rul_estimation"
    FORECASTING = "forecasting"
    FEATURE_EXTRACTION = "feature_extraction"
    COMPRESSION = "compression"
    SURROGATE = "surrogate"
    GRAPH_ANALYSIS = "graph_analysis"
    EXPLAINABILITY = "explainability"
    EDGE_INFERENCE = "edge_inference"
    SAFETY = "safety"


class Deployment(str, Enum):
    CLOUD = "cloud"
    EDGE_CPU = "edge_cpu"
    EDGE_GPU = "edge_gpu"
    MCU = "mcu"  # ESP32 / TinyML


class Subsystem(str, Enum):
    DRIVETRAIN = "drivetrain"
    GEARBOX = "gearbox"
    GENERATOR = "generator"
    ROTOR_BLADES = "rotor_blades"
    YAW_SYSTEM = "yaw_system"
    PITCH_SYSTEM = "pitch_system"
    CONVERTER = "converter"
    SENSORS = "sensors"
    STRUCTURE = "structure"
    FARM = "farm"
    UNKNOWN = "unknown"


#: Map from the simulator's fault kinds to the affected subsystem — used by the
#: DiagnosisEngine to answer "WHERE is it happening?".
FAULT_SUBSYSTEM: Dict[str, Subsystem] = {
    "gearbox_thermal": Subsystem.GEARBOX,
    "bearing_wear": Subsystem.DRIVETRAIN,
    "pitch_misalignment": Subsystem.PITCH_SYSTEM,
    "yaw_error": Subsystem.YAW_SYSTEM,
    "converter_fault": Subsystem.CONVERTER,
    "sensor_freeze": Subsystem.SENSORS,
    "sensor_drift": Subsystem.SENSORS,
    "healthy": Subsystem.UNKNOWN,
}

FAULT_CLASSES: List[str] = [
    "healthy",
    "gearbox_thermal",
    "bearing_wear",
    "pitch_misalignment",
    "yaw_error",
    "converter_fault",
    "sensor_freeze",
    "sensor_drift",
]

#: Vibration-surrogate classes used by the waveform models (m01/m07/m08).
VIB_CLASSES: List[str] = ["healthy", "bearing_wear", "imbalance", "other"]


# ---------------------------------------------------------------------------
# WTDataSchema — canonical record fields
# ---------------------------------------------------------------------------
class WTDataSchema:
    """Field dictionary + validation for the cross-model record contract."""

    REQUIRED = ("timestamp", "turbine_id", "model_id")
    OPTIONAL = (
        "subsystem", "sensor_id", "operating_state",
        "wind_speed", "power", "temperature", "vibration", "rpm", "torque",
        "electrical_parameters", "scada_features", "extracted_features",
        "prediction", "probability", "anomaly_score", "degradation_state",
        "rul_hours", "uncertainty", "inference_time_ms", "explanation",
        "schema_version",
    )

    @classmethod
    def fields(cls) -> Sequence[str]:
        return tuple(cls.REQUIRED) + tuple(cls.OPTIONAL)

    @classmethod
    def validate(cls, record: Mapping[str, Any]) -> List[str]:
        """Return a list of problems (empty list == valid)."""
        problems = []
        for k in cls.REQUIRED:
            if k not in record or record[k] in (None, ""):
                problems.append(f"missing required field: {k}")
        unknown = set(record) - set(cls.fields())
        if unknown:
            problems.append(f"unknown fields: {sorted(unknown)}")
        if "probability" in record and record["probability"] is not None:
            p = record["probability"]
            if isinstance(p, Mapping):
                total = float(sum(p.values()))
                if total > 0 and not (0.95 <= total <= 1.05):
                    problems.append(f"probability mass {total:.3f} not ~1")
        return problems


# ---------------------------------------------------------------------------
# SensorBatch — standardised model input
# ---------------------------------------------------------------------------
@dataclass
class SensorBatch:
    """A slice of one turbine's synchronized SCADA data plus derived views.

    ``values`` is (T, C) over the canonical 12 channels of model 05's schema.
    Derived views (tabular features, windows, vibration waveforms) are attached
    by the FeaturePipeline so each adapter picks the representation it needs —
    nothing is forced through an incompatible input path.
    """

    turbine_id: str
    timestamps: np.ndarray          # (T,) unix seconds
    channel_names: Sequence[str]
    values: np.ndarray              # (T, C)
    operating_state: Optional[np.ndarray] = None   # (T,) int codes
    features: Optional[np.ndarray] = None          # (T, F) tabular features
    feature_names: Sequence[str] = ()
    windows: Optional[np.ndarray] = None           # (N, W, C) sequence windows
    window_index: Optional[np.ndarray] = None      # (N,) end-step of window i
    vib_waveforms: Optional[np.ndarray] = None     # (M, L) surrogate waveforms
    vib_index: Optional[np.ndarray] = None         # (M,) step of waveform j
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return int(self.values.shape[0])

    def channel(self, name: str) -> np.ndarray:
        return self.values[:, list(self.channel_names).index(name)]


# ---------------------------------------------------------------------------
# ModelOutput — standardised model output
# ---------------------------------------------------------------------------
@dataclass
class ModelOutput:
    """The single result type every adapter must return."""

    model_id: str
    task: TaskType
    turbine_id: str
    timestamps: np.ndarray                      # (T,) aligned to input steps
    subsystem: Subsystem = Subsystem.UNKNOWN
    prediction: Optional[np.ndarray] = None     # class labels / values per step
    probability: Optional[Dict[str, np.ndarray]] = None  # class -> (T,) probs
    anomaly_score: Optional[np.ndarray] = None  # (T,) higher = more anomalous
    degradation_state: Optional[np.ndarray] = None  # (T,) int 0..K
    rul_hours: Optional[np.ndarray] = None      # (T,)
    uncertainty: Optional[np.ndarray] = None    # (T,) std-dev-like
    inference_time_ms: float = 0.0
    explanation: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    error: str = ""

    def to_records(self, stride: int = 1) -> List[Dict[str, Any]]:
        """Emit contract-conformant dict records (thinned by ``stride``)."""
        out = []
        for i in range(0, len(self.timestamps), stride):
            rec: Dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "timestamp": int(self.timestamps[i]),
                "turbine_id": self.turbine_id,
                "model_id": self.model_id,
                "subsystem": self.subsystem.value,
                "inference_time_ms": round(self.inference_time_ms, 3),
                "explanation": self.explanation,
            }
            if self.prediction is not None:
                rec["prediction"] = _item(self.prediction[i])
            if self.probability:
                rec["probability"] = {k: float(v[i]) for k, v in self.probability.items()}
            if self.anomaly_score is not None:
                rec["anomaly_score"] = float(self.anomaly_score[i])
            if self.degradation_state is not None:
                rec["degradation_state"] = int(self.degradation_state[i])
            if self.rul_hours is not None:
                rec["rul_hours"] = float(self.rul_hours[i])
            if self.uncertainty is not None:
                rec["uncertainty"] = float(self.uncertainty[i])
            out.append(rec)
        return out

    @staticmethod
    def failed(model_id: str, task: TaskType, turbine_id: str, error: str) -> "ModelOutput":
        return ModelOutput(
            model_id=model_id, task=task, turbine_id=turbine_id,
            timestamps=np.zeros(0), ok=False, error=error,
        )


def _item(x: Any) -> Any:
    try:
        return x.item()
    except AttributeError:
        return x


# ---------------------------------------------------------------------------
# OperatingContext — what the router uses to pick models
# ---------------------------------------------------------------------------
@dataclass
class OperatingContext:
    """Everything the ModelRouter is allowed to look at when routing."""

    mode: str = "production"                 # "research" | "production"
    deployment: Deployment = Deployment.CLOUD
    has_scada: bool = True
    has_vibration_waveform: bool = False     # true high-frequency DAQ present?
    has_fleet_graph: bool = False
    has_labels: bool = False                 # research mode with labelled data
    questions: Sequence[str] = ("what", "where", "why", "severity", "rul", "action", "safety")
    latency_budget_ms: float = 60_000.0
    turbine_count: int = 1
    # When True (default for research/production), every available adapter is
    # eligible on CLOUD even if its primary target is edge/MCU. MCU deployments
    # still refuse large models.
    connect_all: bool = True


class Question(str, Enum):
    WHAT = "what"          # anomaly detection + classification
    WHERE = "where"        # subsystem identification
    WHY = "why"            # XAI + physics reasoning
    SEVERITY = "severity"  # degradation state
    RUL = "rul"            # remaining useful life
    ACTION = "action"      # maintenance recommendation
    SAFETY = "safety"      # continue operating?


class Stopwatch:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.ms = (time.perf_counter() - self.t0) * 1000.0
