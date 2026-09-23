"""Evaluation framework + monitoring + experiment tracking.

Covers: accuracy/precision/recall/F1, ROC-AUC, MAE/RMSE, RUL error, anomaly
performance (incl. false-alarm and missed-fault rates), calibration (ECE),
uncertainty quality, latency, memory and edge-artifact size. Cross-model
comparison only groups models that solve comparable tasks.

Reuses model 05's protocol primitives (roc_auc, average_precision, event
metrics) instead of re-implementing them — the reference repo is the
evaluation authority.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from wtpm_platform.contracts import FAULT_CLASSES, ModelOutput, TaskType

try:
    from wt_pm_lstm.evaluate import roc_auc as _roc_auc, average_precision as _ap
    from wt_pm_lstm.drift import population_stability_index as _psi
    HAVE_REF_EVAL = True
except ImportError:
    HAVE_REF_EVAL = False


# ---------------------------------------------------------------------------
# metric primitives
# ---------------------------------------------------------------------------
def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "accuracy": (tp + tn) / max(len(y_true), 1),
        "precision": prec, "recall": rec, "f1": f1,
        "false_alarm_rate": fp / (fp + tn) if fp + tn else 0.0,
        "missed_fault_rate": fn / (fn + tp) if fn + tp else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if HAVE_REF_EVAL:
        return float(_roc_auc(np.asarray(y_true), np.asarray(score)))
    # rank-based fallback
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, float)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg])))
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) - 1) / 2)
                 / (len(pos) * len(neg)))


def pr_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if HAVE_REF_EVAL:
        return float(_ap(np.asarray(y_true), np.asarray(score)))
    return float("nan")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    e = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "bias": float(e.mean())}


def rul_metrics(rul_true: np.ndarray, rul_pred: np.ndarray) -> Dict[str, float]:
    m = regression_metrics(rul_true, rul_pred)
    # prognostics-standard asymmetric score (late predictions penalised more)
    e = np.asarray(rul_pred) - np.asarray(rul_true)
    a1, a2 = 13.0, 10.0
    exponent = np.where(e < 0, -e / a1, e / a2)
    # Clip before exp: clipping its result is too late to prevent overflow.
    s = np.expm1(np.clip(exponent, 0, np.log1p(1e6)))
    m["phm_score_mean"] = float(np.mean(s))
    m["late_fraction"] = float((e > 0).mean())
    return m


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray,
                               bins: int = 10) -> float:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob, float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1.0)
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def uncertainty_quality(y_err: np.ndarray, y_unc: np.ndarray) -> Dict[str, float]:
    """Does reported uncertainty correlate with actual error? (Spearman-ish)."""
    e = np.abs(np.asarray(y_err, float))
    u = np.asarray(y_unc, float)
    if len(e) < 3 or u.std() < 1e-12 or e.std() < 1e-12:
        return {"err_unc_corr": float("nan")}
    re = np.argsort(np.argsort(e)).astype(float)
    ru = np.argsort(np.argsort(u)).astype(float)
    return {"err_unc_corr": float(np.corrcoef(re, ru)[0, 1])}


def resource_usage() -> Dict[str, float]:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return {"max_rss_mb": ru.ru_maxrss / 1024.0, "user_cpu_s": ru.ru_utime}


# ---------------------------------------------------------------------------
# cross-model evaluation (comparable tasks only)
# ---------------------------------------------------------------------------
class Evaluator:
    def evaluate_anomaly_models(
        self, outputs: Sequence[ModelOutput], y_true: np.ndarray,
        threshold: float = 3.0,
    ) -> Dict[str, Dict[str, float]]:
        res: Dict[str, Dict[str, float]] = {}
        for o in outputs:
            if not o.ok or o.anomaly_score is None:
                continue
            s = np.asarray(o.anomaly_score, float)
            yt = y_true[-len(s):]
            m = binary_metrics(yt, (s > threshold).astype(int))
            m["roc_auc"] = roc_auc(yt, s)
            m["pr_auc"] = pr_auc(yt, s)
            m["latency_ms"] = o.inference_time_ms
            if o.uncertainty is not None and len(o.uncertainty) == len(s):
                m.update(uncertainty_quality(s - s.mean(), o.uncertainty))
            res[o.model_id] = m
        return res

    def evaluate_classifiers(
        self, outputs: Sequence[ModelOutput], fault_kind_per_step: List[str],
    ) -> Dict[str, Dict[str, float]]:
        res = {}
        y = np.array([k if k else "healthy" for k in fault_kind_per_step], dtype=object)
        for o in outputs:
            if (not o.ok or o.prediction is None or
                    o.task not in (TaskType.FAULT_CLASSIFICATION, TaskType.EDGE_INFERENCE)):
                continue
            pred = np.asarray(o.prediction, dtype=object)
            yt = y[-len(pred):]
            if o.probability and "not_electrical" in o.probability:
                yt = np.where(yt == "converter_fault", "converter_fault", "not_electrical")
            classes = [c for c in np.unique(yt)]
            acc = float((pred == yt).mean())
            # macro P/R/F1 over classes present in truth
            precs, recs, f1s = [], [], []
            for c in classes:
                tp = int(((yt == c) & (pred == c)).sum())
                fp = int(((yt != c) & (pred == c)).sum())
                fn = int(((yt == c) & (pred != c)).sum())
                p = tp / (tp + fp) if tp + fp else 0.0
                r = tp / (tp + fn) if tp + fn else 0.0
                precs.append(p); recs.append(r)
                f1s.append(2 * p * r / (p + r) if p + r else 0.0)
            m = {"accuracy": acc, "macro_precision": float(np.mean(precs)),
                 "macro_recall": float(np.mean(recs)), "macro_f1": float(np.mean(f1s)),
                 "latency_ms": o.inference_time_ms}
            # calibration on the healthy class if probs available
            if o.probability and "healthy" in o.probability:
                p_h = np.asarray(o.probability["healthy"], float)
                m["ece_healthy"] = expected_calibration_error(
                    (yt == "healthy").astype(int), p_h[-len(yt):])
            res[o.model_id] = m
        return res

    def evaluate_rul_models(
        self, outputs: Sequence[ModelOutput], rul_true: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        res = {}
        for o in outputs:
            if not o.ok or o.rul_hours is None:
                continue
            r = np.asarray(o.rul_hours, float)
            yt = rul_true[-len(r):]
            fin = np.isfinite(r) & np.isfinite(yt)
            m = rul_metrics(yt[fin], r[fin])
            m["latency_ms"] = o.inference_time_ms
            if o.uncertainty is not None and len(o.uncertainty) == len(r):
                m.update(uncertainty_quality(r[fin] - yt[fin],
                                             np.asarray(o.uncertainty)[fin]))
            res[o.model_id] = m
        return res


def select_weights(anomaly_eval: Dict[str, Dict[str, float]],
                   metric: str = "roc_auc") -> Dict[str, float]:
    """Validation-driven fusion weights: w ∝ max(metric - 0.5, 0.05)."""
    w = {}
    for mid, m in anomaly_eval.items():
        v = m.get(metric)
        if v is None or not np.isfinite(v):
            v = 0.5
        w[mid] = max(v - 0.5, 0.05)
    return w


# ---------------------------------------------------------------------------
# drift monitoring (production mode)
# ---------------------------------------------------------------------------
def drift_report(train_features: np.ndarray, live_features: np.ndarray,
                 feature_names: Sequence[str]) -> Dict[str, Any]:
    """PSI per feature; uses model 05's population_stability_index when present."""
    psis: Dict[str, float] = {}
    for j, name in enumerate(feature_names):
        a, b = train_features[:, j], live_features[:, j]
        if HAVE_REF_EVAL:
            psis[str(name)] = float(_psi(a, b))
        else:
            qs = np.quantile(a, np.linspace(0, 1, 11))
            qs[0], qs[-1] = -np.inf, np.inf
            pa, _ = np.histogram(a, qs); pb, _ = np.histogram(b, qs)
            pa = pa / pa.sum() + 1e-6; pb = pb / pb.sum() + 1e-6
            psis[str(name)] = float(((pa - pb) * np.log(pa / pb)).sum())
    mx = max(psis.values()) if psis else 0.0
    return {"drifted": mx > 0.25, "psi_max": mx,
            "channels": dict(sorted(psis.items(), key=lambda kv: -kv[1])[:8])}


# ---------------------------------------------------------------------------
# ExperimentTracker (research mode)
# ---------------------------------------------------------------------------
class ExperimentTracker:
    """Filesystem-backed run records, same philosophy as model 05's registry:
    plain JSON, reproducible config, no pickle."""

    def __init__(self, root: str = "runs_platform") -> None:
        self.root = root

    def save_run(self, name: str, config: Dict[str, Any],
                 metrics: Dict[str, Any], artifacts: Optional[Dict[str, str]] = None) -> str:
        import hashlib
        fp = hashlib.sha256(json.dumps(config, sort_keys=True, default=str)
                            .encode()).hexdigest()[:10]
        run_dir = os.path.join(self.root, f"{name}-{fp}")
        os.makedirs(run_dir, exist_ok=True)
        env = {"python": platform.python_version(),
               "platform": platform.platform(),
               "timestamp": time.time(), **resource_usage()}
        for fname, obj in (("config.json", config), ("metrics.json", metrics),
                           ("environment.json", env),
                           ("artifacts.json", artifacts or {})):
            with open(os.path.join(run_dir, fname), "w") as f:
                json.dump(obj, f, indent=2, default=_json_safe)
        return run_dir


def _json_safe(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
