"""Model registry and run records (the platform's MLOps boundary).

Two artefacts, both plain files so a reviewer can diff them:

* **Detector bundle** — ``.npz`` weights plus a ``.json`` sidecar holding the
  scaler, residual normaliser, threshold and every configuration value that
  affected the run. Reloading a bundle reproduces the exact scores; there is no
  hidden state and no pickle (pickle would make the artefact neither portable
  nor safely inspectable).
* **Run record** — ``runs/<name>/<fingerprint>/`` with ``config.json``,
  ``metrics.json``, ``history.json`` and the alert list.

The registry key is ``config_fingerprint``: identical configurations map to the
same directory, so a re-run overwrites rather than silently forking history.
"""

from __future__ import annotations

import json
import os
import platform
from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from wt_pm_lstm.baseline import NormalBehaviourModel
from wt_pm_lstm.config import RunConfig, config_fingerprint
from wt_pm_lstm.detect import Detector, ResidualNormaliser, Threshold
from wt_pm_lstm.nn import SequenceAutoencoder
from wt_pm_lstm.windows import ChannelScaler

MODEL_ID_TEMPLATE = "wtpm-lstm-scada-{cell}-v1@{fingerprint}"


def model_id(cfg: RunConfig) -> str:
    return MODEL_ID_TEMPLATE.format(cell=cfg.model.cell, fingerprint=config_fingerprint(cfg))


def environment_info() -> Dict[str, str]:
    """Provenance for the run record: numbers without provenance are anecdotes."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


# --------------------------------------------------------------------------
# Detector bundles
# --------------------------------------------------------------------------
def save_detector(detector: Detector, path: str) -> Tuple[str, str]:
    """Write ``<path>.npz`` (weights) and ``<path>.json`` (everything else)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    arrays: Dict[str, Any] = {}
    for mi, model in enumerate(detector.models):
        for k, v in model.params.items():
            arrays[f"model{mi}.{k}"] = v
    np.savez_compressed(path + ".npz", **arrays)

    sidecar = detector.to_metadata()
    sidecar["config"] = detector.cfg.to_dict()
    sidecar["fingerprint"] = config_fingerprint(detector.cfg)
    sidecar["model_id"] = detector.metadata.get("model_id", model_id(detector.cfg))
    sidecar["environment"] = environment_info()
    with open(path + ".json", "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2, sort_keys=True)
    return path + ".npz", path + ".json"


def load_detector(path: str) -> Detector:
    """Rebuild a :class:`Detector` from a bundle written by :func:`save_detector`."""
    with open(path + ".json", "r", encoding="utf-8") as fh:
        sidecar = json.load(fh)
    cfg = RunConfig.from_dict(sidecar["config"])
    channel_names = tuple(sidecar["channel_names"])
    with np.load(path + ".npz") as bundle:
        n_members = int(sidecar["model"]["n_members"])
        models: List[SequenceAutoencoder] = []
        members = sidecar.get("model", {}).get("members") or []
        for mi in range(n_members):
            # Which channels the decoder was given is part of the architecture,
            # not a training detail: restoring a model with no context channels
            # loads without error and quietly scores differently. Read the
            # per-member record first, then the pool.
            exogenous = ()
            if mi < len(members) and "exogenous_index" in members[mi]:
                exogenous = tuple(int(i) for i in members[mi]["exogenous_index"])
            elif "exogenous_index_pool" in sidecar:
                exogenous = tuple(int(i) for i in sidecar["exogenous_index_pool"])
            model = SequenceAutoencoder(
                input_size=len(channel_names),
                hidden_size=cfg.model.hidden_size,
                latent_size=int(sidecar["model"]["latent_size"]),
                cell=cfg.model.cell,
                seed=cfg.model.offline_seed + 1000 * mi,
                forecast_weight=cfg.model.forecast_weight,
                exogenous_index=exogenous,
            )
            prefix = f"model{mi}."
            model.load_params({key: bundle[prefix + key] for key in model.params})
            models.append(model)
    detector = Detector(
        cfg=cfg,
        models=models,
        score_mask=(
            np.asarray(sidecar["score_mask"], dtype=bool) if "score_mask" in sidecar else None
        ),
        drift_scale=(
            np.asarray(sidecar["drift_scale"], dtype=np.float64) if "drift_scale" in sidecar else None
        ),
        nbm=(
            NormalBehaviourModel.from_dict(sidecar["nbm"]) if sidecar.get("nbm") else None
        ),
        level_centre=(
            np.asarray(sidecar["level_centre"], dtype=np.float64) if "level_centre" in sidecar else None
        ),
        level_scale=(
            np.asarray(sidecar["level_scale"], dtype=np.float64) if "level_scale" in sidecar else None
        ),
        flatline_ref=(
            np.asarray(sidecar["flatline_ref"], dtype=np.float64) if "flatline_ref" in sidecar else None
        ),
        scaler=ChannelScaler.from_dict(sidecar["scaler"]),
        residual_normaliser=ResidualNormaliser.from_dict(sidecar["residual_normaliser"]),
        threshold=Threshold.from_dict(sidecar["threshold"]),
        channel_names=channel_names,
        metadata={
            "model_id": sidecar["model_id"],
            "fingerprint": sidecar["fingerprint"],
            "environment": sidecar["environment"],
        },
    )
    return detector


# --------------------------------------------------------------------------
# Run records
# --------------------------------------------------------------------------
def run_dir(cfg: RunConfig, root: str = "runs") -> str:
    fp = config_fingerprint(cfg)
    return os.path.join(root, f"{cfg.name}-{fp}")


def save_run(
    cfg: RunConfig,
    metrics: Mapping[str, Any],
    history: Optional[Sequence[Mapping[str, Any]]] = None,
    records: Optional[Sequence[Mapping[str, Any]]] = None,
    extras: Optional[Mapping[str, Any]] = None,
    root: str = "runs",
) -> str:
    """Persist a run record; returns the directory written."""
    directory = run_dir(cfg, root)
    os.makedirs(directory, exist_ok=True)
    fingerprint = config_fingerprint(cfg)

    def dump(name: str, payload: Any) -> None:
        with open(os.path.join(directory, name), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=float)

    dump("config.json", cfg.to_dict())
    dump(
        "metrics.json",
        {"fingerprint": fingerprint, "model_id": model_id(cfg), "environment": environment_info(), **metrics},
    )
    if history is not None:
        dump("history.json", list(history))
    if records is not None:
        dump("alerts.json", list(records))
    if extras:
        dump("extras.json", dict(extras))
    return directory


def load_run(path: str) -> Dict[str, Any]:
    """Load a run record directory back into a dict."""
    out: Dict[str, Any] = {"path": path}
    for name in ("config", "metrics", "history", "alerts", "extras"):
        fp = os.path.join(path, f"{name}.json")
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as fh:
                out[name] = json.load(fh)
    return out


def list_runs(root: str = "runs") -> List[Dict[str, Any]]:
    """Summarise every run record under ``root`` (newest first by name)."""
    if not os.path.isdir(root):
        return []
    out: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        metrics_path = os.path.join(path, "metrics.json")
        if not os.path.exists(metrics_path):
            continue
        with open(metrics_path, "r", encoding="utf-8") as fh:
            metrics = json.load(fh)
        out.append(
            {
                "run": name,
                "model_id": metrics.get("model_id"),
                "f1": metrics.get("point", {}).get("f1"),
                "pr_auc": metrics.get("pr_auc"),
                "path": path,
            }
        )
    return out


__all__ = [
    "save_detector",
    "load_detector",
    "save_run",
    "load_run",
    "list_runs",
    "run_dir",
    "model_id",
    "environment_info",
]
