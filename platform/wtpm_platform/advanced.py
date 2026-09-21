"""Advanced production features sitting on the 25-model pipeline.

AlertManager          — hysteresis + cooldown so fused scores don't chatter
SensorQualityMonitor  — freeze / spike / stuck-at / range checks (data trust)
HealthIndex           — composite 0–100 turbine health from fused streams
MaintenancePlanner    — work orders, parts, window, cost from diagnosis+RUL
CostRiskOptimizer     — continue / derate / inspect / trip expected-cost
DecisionAuditLog      — immutable (in-process) trail of analyses + operator acts
StreamingCursor       — rolling-window inference over a live record
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from wtpm_platform.contracts import SensorBatch


# ---------------------------------------------------------------------------
# Sensor quality — trust the data before you trust the models
# ---------------------------------------------------------------------------
class SensorQualityMonitor:
    """Unsupervised quality flags that do not depend on any of the 25 models.

    Freeze: zero variance over ``freeze_steps``. Spike: |z| > spike_z vs rolling
    median. Range: outside physical limits. These feed the SafetyManager and
    the dashboard's data-trust badge.
    """

    LIMITS = {
        "wind_speed_ms": (0.0, 40.0),
        "rotor_speed_rpm": (0.0, 30.0),
        "power_kw": (-50.0, 4000.0),
        "gearbox_oil_temp_c": (0.0, 95.0),
        "bearing_vib_rms_mm_s": (0.0, 25.0),
        "grid_frequency_hz": (47.0, 53.0),
        "generator_current_a": (0.0, 8000.0),
    }

    def __init__(self, freeze_steps: int = 18, spike_z: float = 8.0) -> None:
        self.freeze_steps = freeze_steps
        self.spike_z = spike_z

    def run(self, batch: SensorBatch) -> Dict[str, Any]:
        flags: List[Dict[str, Any]] = []
        per_ch: Dict[str, Dict[str, float]] = {}
        T = batch.n_steps
        w = min(self.freeze_steps, max(T // 8, 4))
        for ch in batch.channel_names:
            x = np.asarray(batch.channel(ch), float)
            finite = np.isfinite(x)
            freeze_frac = 0.0
            if T >= w:
                # rolling range ~0 => frozen
                cmax = np.maximum.accumulate(x)
                # simpler: last-w std
                std_w = float(np.nanstd(x[-w:]))
                freeze_frac = 1.0 if std_w < 1e-6 else 0.0
                if freeze_frac and ch not in ("grid_frequency_hz",):
                    flags.append({"channel": ch, "kind": "freeze",
                                  "detail": f"std({w} steps)={std_w:.2e}"})
            med = np.nanmedian(x)
            mad = np.nanmedian(np.abs(x - med)) + 1e-9
            z = np.abs(x - med) / (1.4826 * mad)
            spike_frac = float(np.mean(z > self.spike_z))
            if spike_frac > 0.02:
                flags.append({"channel": ch, "kind": "spike",
                              "detail": f"{spike_frac:.1%} of steps |z|>{self.spike_z}"})
            lo, hi = self.LIMITS.get(ch, (None, None))
            range_frac = 0.0
            if lo is not None:
                range_frac = float(np.mean((x < lo) | (x > hi)))
                if range_frac > 0:
                    flags.append({"channel": ch, "kind": "range",
                                  "detail": f"{range_frac:.1%} outside [{lo},{hi}]"})
            per_ch[ch] = {
                "nan_frac": float(1.0 - finite.mean()),
                "freeze": freeze_frac, "spike_frac": spike_frac,
                "range_frac": range_frac,
            }
        trust = 1.0
        for f in flags:
            trust *= 0.85 if f["kind"] != "range" else 0.9
        return {
            "trust": round(max(trust, 0.2), 3),
            "n_flags": len(flags),
            "flags": flags[:40],
            "per_channel": per_ch,
        }


# ---------------------------------------------------------------------------
# Composite health index
# ---------------------------------------------------------------------------
def health_index(fused_score: Optional[np.ndarray],
                 degradation_state: Optional[np.ndarray],
                 rul_hours: Optional[np.ndarray],
                 vib: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """0–100 health (100 = as-new). Downsampled series for the dashboard."""
    n = 0
    for a in (fused_score, degradation_state, rul_hours, vib):
        if a is not None:
            n = max(n, len(a))
    if n == 0:
        return {"now": 100.0, "series": [], "stride": 1}
    hi = np.full(n, 100.0)
    if fused_score is not None:
        s = _pad(fused_score, n)
        hi -= 18.0 * np.clip(s / 6.0, 0, 1)
    if degradation_state is not None:
        d = _pad(np.asarray(degradation_state, float), n)
        hi -= 22.0 * np.clip(d / 3.0, 0, 1)
    if rul_hours is not None:
        r = _pad(rul_hours, n)
        hi -= 25.0 * np.clip(1.0 - r / 336.0, 0, 1)
    if vib is not None:
        v = _pad(vib, n)
        base = np.nanpercentile(v[: max(n // 3, 1)], 50)
        hi -= 15.0 * np.clip((v - base) / max(base, 1e-6) / 4.0, 0, 1)
    hi = np.clip(hi, 0, 100)
    stride = max(n // 180, 1)
    series = [round(float(x), 2) for x in hi[::stride]]
    return {"now": round(float(hi[-1]), 2), "series": series, "stride": int(stride),
            "n_steps": n}


def _pad(a: np.ndarray, n: int) -> np.ndarray:
    a = np.asarray(a, float)
    if len(a) >= n:
        return a[:n]
    return np.concatenate([np.full(n - len(a), a[0] if len(a) else 0.0), a])


def downsample_series(x: np.ndarray, points: int = 180) -> List[float]:
    x = np.asarray(x, float)
    if len(x) <= points:
        return [round(float(v), 4) for v in x]
    idx = np.linspace(0, len(x) - 1, points).astype(int)
    return [round(float(x[i]), 4) for i in idx]


# ---------------------------------------------------------------------------
# Alerts with hysteresis
# ---------------------------------------------------------------------------
@dataclass
class Alert:
    alert_id: str
    turbine_id: str
    severity: str          # info | warning | high | critical
    kind: str
    message: str
    ts: int
    acked: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


class AlertManager:
    """Raise on crossing, clear only after ``clear_below`` for ``hold`` steps."""

    def __init__(self, raise_at: float = 3.0, clear_at: float = 2.0,
                 hold: int = 6) -> None:
        self.raise_at = raise_at
        self.clear_at = clear_at
        self.hold = hold
        self._active: Dict[str, Alert] = {}
        self._history: List[Alert] = []
        self._below = 0

    def update(self, turbine_id: str, fused_score: float, risk: float,
               safety: str, fault: str, ts: int) -> List[Alert]:
        key = f"{turbine_id}::anomaly"
        new: List[Alert] = []
        if fused_score >= self.raise_at or risk >= 70 or safety == "TRIP":
            self._below = 0
            if key not in self._active:
                sev = "critical" if safety == "TRIP" or risk >= 85 else (
                    "high" if risk >= 70 or fused_score >= 4.5 else "warning")
                a = Alert(alert_id=uuid.uuid4().hex[:10], turbine_id=turbine_id,
                          severity=sev, kind="anomaly_or_risk",
                          message=f"{fault}  fused={fused_score:.2f}  risk={risk}  safety={safety}",
                          ts=ts, extra={"fused_score": fused_score, "risk": risk,
                                        "safety": safety, "fault": fault})
                self._active[key] = a
                self._history.append(a)
                new.append(a)
        else:
            if fused_score < self.clear_at:
                self._below += 1
            if key in self._active and self._below >= self.hold:
                self._active.pop(key)
        return new

    def ack(self, alert_id: str) -> bool:
        for a in self._history:
            if a.alert_id == alert_id:
                a.acked = True
                return True
        return False

    def snapshot(self) -> Dict[str, Any]:
        return {
            "active": [a.__dict__ for a in self._active.values()],
            "recent": [a.__dict__ for a in self._history[-20:]],
        }


# ---------------------------------------------------------------------------
# Maintenance planner + cost-risk
# ---------------------------------------------------------------------------
# Heuristic EUR costs — labelled as such, not a commercial quote.
PARTS = {
    "gearbox": {"oil_filter_kit": 420, "bearing_set": 18000, "labour_h": 16},
    "drivetrain": {"main_bearing": 22000, "labour_h": 24},
    "generator": {"brush_kit": 900, "stator_rewind": 35000, "labour_h": 20},
    "converter": {"igbt_module": 4500, "labour_h": 8},
    "pitch_system": {"actuator": 3200, "labour_h": 6},
    "yaw_system": {"yaw_motor": 2800, "labour_h": 8},
    "sensors": {"sensor_pack": 350, "labour_h": 2},
    "unknown": {"inspection": 800, "labour_h": 4},
}
LABOUR_EUR_H = 95.0
LOST_PROD_EUR_H = 180.0   # ~2 MW * 90 €/MWh * capacity factor-ish


class MaintenancePlanner:
    def plan(self, diagnosis, rul_hours: Optional[float], risk: float,
             safety: str) -> Dict[str, Any]:
        sub = diagnosis.subsystem if hasattr(diagnosis, "subsystem") else "unknown"
        kit = PARTS.get(sub, PARTS["unknown"])
        labour_h = float(kit.get("labour_h", 4))
        parts_eur = sum(v for k, v in kit.items() if k != "labour_h")
        labour_eur = labour_h * LABOUR_EUR_H
        # window: inspect before RUL*0.6, never later than RUL-24h
        rul = 400.0 if rul_hours is None else float(rul_hours)
        window_h = max(min(rul * 0.6, rul - 24.0), 8.0)
        priority = "P1" if safety == "TRIP" or risk >= 70 else (
            "P2" if risk >= 40 else "P3")
        wo = {
            "work_order_id": "WO-" + uuid.uuid4().hex[:8],
            "subsystem": sub,
            "fault": getattr(diagnosis, "fault", "unknown"),
            "priority": priority,
            "window_hours": round(window_h, 1),
            "parts": {k: v for k, v in kit.items() if k != "labour_h"},
            "labour_hours": labour_h,
            "estimated_cost_eur": round(parts_eur + labour_eur, 0),
            "lost_production_if_trip_eur": round(24 * LOST_PROD_EUR_H, 0),
            "note": "costs are heuristic planning figures, not a commercial quote",
        }
        return wo


class CostRiskOptimizer:
    """Pick the action with lowest expected 48h cost (heuristic)."""

    def choose(self, risk: float, rul_hours: Optional[float], safety: str,
               work_order: Dict[str, Any]) -> Dict[str, Any]:
        inspect_cost = work_order["estimated_cost_eur"] * 0.25 + 800
        derate_cost = 48 * LOST_PROD_EUR_H * 0.20
        trip_cost = work_order["estimated_cost_eur"] + 24 * LOST_PROD_EUR_H
        p_fail = min(0.95, (risk / 100.0) ** 1.4)
        if rul_hours is not None and rul_hours < 48:
            p_fail = max(p_fail, 0.55)
        continue_cost = p_fail * trip_cost
        options = {
            "continue_monitoring": continue_cost,
            "inspect_at_next_service": inspect_cost + 0.3 * continue_cost,
            "schedule_inspection": inspect_cost,
            "derate": derate_cost + 0.15 * continue_cost,
            "stop_turbine": trip_cost,
        }
        if safety == "TRIP":
            best = "stop_turbine"
        else:
            best = min(options, key=options.get)
        return {
            "recommended": best,
            "expected_cost_eur": {k: round(v, 0) for k, v in options.items()},
            "p_fail_48h": round(p_fail, 3),
            "note": "expected-cost heuristic over a 48h horizon",
        }


# ---------------------------------------------------------------------------
# Audit log + operator feedback
# ---------------------------------------------------------------------------
class DecisionAuditLog:
    def __init__(self, cap: int = 200) -> None:
        self._rows: List[Dict[str, Any]] = []
        self.cap = cap

    def record(self, kind: str, payload: Dict[str, Any]) -> str:
        rid = uuid.uuid4().hex[:12]
        self._rows.append({"id": rid, "kind": kind, "ts": int(time.time()),
                           **payload})
        if len(self._rows) > self.cap:
            self._rows = self._rows[-self.cap:]
        return rid

    def all(self) -> List[Dict[str, Any]]:
        return list(self._rows)


# ---------------------------------------------------------------------------
# Streaming cursor — walk a record in hops, emit compact ticks
# ---------------------------------------------------------------------------
class StreamingCursor:
    def __init__(self, batch: SensorBatch, hop: int = 12, window: int = 72) -> None:
        self.batch = batch
        self.hop = hop
        self.window = window
        self.i = window

    def slice(self) -> Optional[SensorBatch]:
        if self.i >= self.batch.n_steps:
            return None
        lo = max(0, self.i - self.window)
        hi = self.i
        b = self.batch
        sl = SensorBatch(
            turbine_id=b.turbine_id,
            timestamps=b.timestamps[lo:hi],
            channel_names=b.channel_names,
            values=b.values[lo:hi],
            operating_state=None if b.operating_state is None else b.operating_state[lo:hi],
            features=None if b.features is None else b.features[lo:hi],
            feature_names=b.feature_names,
            windows=None, window_index=None,
            vib_waveforms=None, vib_index=None,
            meta={k: (v[lo:hi] if hasattr(v, "__getitem__") and not isinstance(v, (str, dict))
                      else v) for k, v in b.meta.items()
                  if k not in ("mask",)},
        )
        sl.meta["mask"] = np.isfinite(sl.values)
        self.i += self.hop
        return sl
