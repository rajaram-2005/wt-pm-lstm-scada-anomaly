"""Adapters for the anomaly-detection family.

m05 lstm-scada-anomaly  — reference detector, used via its public package API
m04 gru-scada-telemetry — GRU one-step forecaster (torch), forecast error
m03 tcn-power-curve     — TCN power-curve model (torch), power residual
m13 deep-svdd-boundary  — one-class hypersphere (torch)
m14 isolation-forest-telemetry — sklearn IsolationForest
m22 vae-reconstruction-loss    — VAE reconstruction error (torch)

Each adapter calls the original repository's classes/functions; the platform
adds only data conversion, a training loop where the repo ships none, and
score normalisation (all scores are re-expressed as robust z-scores against
the healthy calibration band so the FusionEngine can combine them).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from wtpm_platform.base import BaseWTModel, ModelSpec, load_repo_module
from wtpm_platform.contracts import Deployment, ModelOutput, SensorBatch, Subsystem, TaskType
from wtpm_platform.data import RESPONSE_CHANNELS


def robust_calibrate(scores_train: np.ndarray):
    med = float(np.median(scores_train))
    mad = float(np.median(np.abs(scores_train - med))) * 1.4826 + 1e-9
    return lambda s: (s - med) / mad


def _standardiser(x_train: np.ndarray):
    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0) + 1e-9
    return lambda x: (x - mu) / sd


# ---------------------------------------------------------------------------
# m05 — reference LSTM SCADA anomaly detector (full package)
# ---------------------------------------------------------------------------
class LSTMScadaAnomaly(BaseWTModel):
    spec = ModelSpec(
        model_id="m05-lstm-scada-anomaly",
        repository="wt-pm-lstm-scada-anomaly",
        task=TaskType.ANOMALY_DETECTION,
        input_requirements=["values(12ch canonical)", "timestamps"],
        output_schema=["anomaly_score", "uncertainty", "explanation"],
        resource_requirements=["numpy", "wt_pm_lstm"],
        typical_latency_ms=4000,
        deployment_targets=[Deployment.CLOUD, Deployment.EDGE_CPU],
        fallback="m14-isolation-forest",
        notes="reference implementation; supplies threshold + attribution",
    )

    def _check_deps(self) -> None:
        import wt_pm_lstm  # noqa: F401

    def fit(self, batch: SensorBatch, train_mask: np.ndarray) -> None:
        from wt_pm_lstm.config import RunConfig
        from wt_pm_lstm.pipeline import fit_detector
        from wt_pm_lstm.schema import TurbineTimeline

        cfg = RunConfig()
        cfg.model.epochs = int(batch.meta.get("m05_epochs", 8))
        cfg.model.n_ensemble = 2
        n_steps = batch.n_steps
        cfg.data.n_days = n_steps * cfg.data.sample_minutes / (24 * 60)
        # platform records inject faults from 55% onward: keep train+calibration
        # strictly inside the healthy prefix (fit_detector refuses otherwise)
        cfg.data.healthy_fraction = 0.40
        cfg.data.calibration_fraction = 0.12
        timeline = TurbineTimeline(
            turbine_id=batch.turbine_id,
            channel_names=tuple(batch.channel_names),
            values=batch.values,
            timestamps=batch.timestamps,
            fault_label=batch.meta.get("fault_label"),
        )
        self._detector, _, _ = fit_detector(cfg, timeline)
        self._fitted = True

    def _predict(self, batch: SensorBatch) -> ModelOutput:
        from wt_pm_lstm.schema import TurbineTimeline

        timeline = TurbineTimeline(
            turbine_id=batch.turbine_id,
            channel_names=tuple(batch.channel_names),
            values=batch.values,
            timestamps=batch.timestamps,
        )
        res = self._detector.score_timeline(timeline)
        score = np.asarray(res.score, float)
        thr = float(self._detector.threshold.value)
        unc = np.asarray(getattr(res, "uncertainty", np.zeros_like(score)), float)
        if unc.shape != score.shape:
            unc = np.zeros_like(score)
        return ModelOutput(
            model_id=self.spec.model_id, task=self.spec.task,
            turbine_id=batch.turbine_id, timestamps=batch.timestamps[-len(score):],
            # rescale so "at threshold" == 3.0, matching the robust-z scale
            # the other anomaly adapters emit (fusion compares like with like)
            anomaly_score=3.0 * score / max(thr, 1e-9),
            uncertainty=unc,
            explanation=f"fused LSTM/NBM/trend score (3.0 == calibrated threshold {thr:.2f})",
            extra={"threshold": thr, "raw": res},
        )


# ---------------------------------------------------------------------------
# m04 — GRU one-step forecaster
# ---------------------------------------------------------------------------
class GRUScadaTelemetry(BaseWTModel):
    spec = ModelSpec(
        model_id="m04-gru-scada-telemetry",
        repository="wt-pm-gru-scada-telemetry",
        task=TaskType.ANOMALY_DETECTION,
        input_requirements=["windows"],
        output_schema=["anomaly_score"],
        resource_requirements=["torch"],
        typical_latency_ms=2000,
        deployment_targets=[Deployment.CLOUD, Deployment.EDGE_GPU, Deployment.EDGE_CPU],
        fallback="m14-isolation-forest",
    )

    def _check_deps(self) -> None:
        import torch  # noqa: F401
        load_repo_module("wt-pm-gru-scada-telemetry")

    def fit(self, batch: SensorBatch, train_mask: np.ndarray) -> None:
        import torch
        mod = load_repo_module("wt-pm-gru-scada-telemetry")
        W = batch.windows
        ends = batch.window_index
        keep = train_mask[ends]
        X = W[keep]
        self._std = _standardiser(X.reshape(-1, X.shape[-1]))
        Xs = self._std(X)
        xb = torch.tensor(Xs[:, :-1, :], dtype=torch.float32)
        yb = torch.tensor(Xs[:, -1, :], dtype=torch.float32)
        model = mod.GRUAnomalyDetector(input_size=X.shape[-1], hidden_size=48, num_layers=1)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        lossf = torch.nn.MSELoss()
        model.train()
        for _ in range(30):
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        self._model = model
        with torch.no_grad():
            err = ((model(xb) - yb) ** 2).mean(dim=1).numpy()
        self._cal = robust_calibrate(err)
        self._fitted = True

    def _predict(self, batch: SensorBatch) -> ModelOutput:
        import torch
        W, ends = batch.windows, batch.window_index
        Xs = self._std(W)
        self._model.eval()
        with torch.no_grad():
            pred = self._model(torch.tensor(Xs[:, :-1, :], dtype=torch.float32))
            err = ((pred - torch.tensor(Xs[:, -1, :], dtype=torch.float32)) ** 2).mean(dim=1).numpy()
        z = self._cal(err)
        score = _to_steps(z, ends, batch.n_steps)
        return ModelOutput(
            model_id=self.spec.model_id, task=self.spec.task,
            turbine_id=batch.turbine_id, timestamps=batch.timestamps,
            anomaly_score=score,
            explanation="GRU one-step forecast error (robust z vs healthy band)",
        )


# ---------------------------------------------------------------------------
# m03 — TCN power-curve residual
# ---------------------------------------------------------------------------
class TCNPowerCurve(BaseWTModel):
    spec = ModelSpec(
        model_id="m03-tcn-power-curve",
        repository="wt-pm-tcn-power-curve",
        task=TaskType.ANOMALY_DETECTION,
        input_requirements=["windows", "power_kw channel"],
        output_schema=["anomaly_score", "prediction(power forecast)"],
        resource_requirements=["torch"],
        typical_latency_ms=3000,
        deployment_targets=[Deployment.CLOUD, Deployment.EDGE_GPU],
        fallback="m04-gru-scada-telemetry",
        subsystem_focus=["rotor_blades", "converter"],
        notes="power-curve degradation tracking via causal TCN residual",
    )

    def _check_deps(self) -> None:
        import torch  # noqa: F401
        load_repo_module("wt-pm-tcn-power-curve")

    def fit(self, batch: SensorBatch, train_mask: np.ndarray) -> None:
        import torch
        mod = load_repo_module("wt-pm-tcn-power-curve")
        W, ends = batch.windows, batch.window_index
        p_idx = list(batch.channel_names).index("power_kw")
        keep = train_mask[ends]
        X = W[keep]
        self._std = _standardiser(X.reshape(-1, X.shape[-1]))
        Xs = self._std(X)
        xb = torch.tensor(Xs[:, :-1, :], dtype=torch.float32).transpose(1, 2)  # (N,C,T)
        yb = torch.tensor(Xs[:, -1, p_idx], dtype=torch.float32)
        model = mod.TCNPowerCurve(input_size=X.shape[-1], num_channels=[16, 16])
        # repo's forward returns (N, C_out, T); take last step -> linear head
        head = torch.nn.Linear(16, 1)
        params = list(model.parameters()) + list(head.parameters())
        opt = torch.optim.Adam(params, lr=3e-3)
        model.train()
        for _ in range(25):
            opt.zero_grad()
            feat = model.network(xb)[:, :, -1] if hasattr(model, "network") else model(xb)[:, :, -1]
            loss = torch.nn.functional.mse_loss(head(feat).squeeze(-1), yb)
            loss.backward()
            opt.step()
        self._model, self._head, self._p_idx = model, head, p_idx
        with torch.no_grad():
            feat = model.network(xb)[:, :, -1] if hasattr(model, "network") else model(xb)[:, :, -1]
            err = np.abs(head(feat).squeeze(-1).numpy() - yb.numpy())
        self._cal = robust_calibrate(err)
        self._fitted = True

    def _predict(self, batch: SensorBatch) -> ModelOutput:
        import torch
        W, ends = batch.windows, batch.window_index
        Xs = self._std(W)
        xb = torch.tensor(Xs[:, :-1, :], dtype=torch.float32).transpose(1, 2)
        with torch.no_grad():
            m = self._model
            feat = m.network(xb)[:, :, -1] if hasattr(m, "network") else m(xb)[:, :, -1]
            pred = self._head(feat).squeeze(-1).numpy()
        err = np.abs(pred - Xs[:, -1, self._p_idx])
        z = self._cal(err)
        score = _to_steps(z, ends, batch.n_steps)
        return ModelOutput(
            model_id=self.spec.model_id, task=self.spec.task,
            turbine_id=batch.turbine_id, timestamps=batch.timestamps,
            anomaly_score=score, subsystem=Subsystem.ROTOR_BLADES,
            explanation="TCN power-curve residual (robust z)",
        )


# ---------------------------------------------------------------------------
# m13 — Deep SVDD one-class boundary
# ---------------------------------------------------------------------------
class DeepSVDDBoundary(BaseWTModel):
    spec = ModelSpec(
        model_id="m13-deep-svdd",
        repository="wt-pm-deep-svdd-boundary",
        task=TaskType.ANOMALY_DETECTION,
        input_requirements=["features"],
        output_schema=["anomaly_score"],
        resource_requirements=["torch"],
        typical_latency_ms=1500,
        deployment_targets=[Deployment.CLOUD, Deployment.EDGE_GPU, Deployment.EDGE_CPU],
        fallback="m14-isolation-forest",
    )

    def _check_deps(self) -> None:
        import torch  # noqa: F401
        load_repo_module("wt-pm-deep-svdd-boundary")

    def fit(self, batch: SensorBatch, train_mask: np.ndarray) -> None:
        import torch
        mod = load_repo_module("wt-pm-deep-svdd-boundary")
        X = batch.features[train_mask]
        self._std = _standardiser(X)
        Xs = torch.tensor(self._std(X), dtype=torch.float32)
        model = mod.DeepSVDDNetwork(input_dim=X.shape[1], rep_dim=16)
        loader = [(Xs,)]
        mod.init_center(model, loader)          # repo's own center init
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        model.train()
        for _ in range(60):
            opt.zero_grad()
            loss = mod.svdd_loss(model(Xs), model.c)   # repo's own loss
            loss.backward()
            opt.step()
        self._model = model
        with torch.no_grad():
            d = ((model(Xs) - model.c) ** 2).sum(dim=1).numpy()
        self._cal = robust_calibrate(d)
        self._fitted = True

    def _predict(self, batch: SensorBatch) -> ModelOutput:
        import torch
        Xs = torch.tensor(self._std(batch.features), dtype=torch.float32)
        with torch.no_grad():
            d = ((self._model(Xs) - self._model.c) ** 2).sum(dim=1).numpy()
        return ModelOutput(
            model_id=self.spec.model_id, task=self.spec.task,
            turbine_id=batch.turbine_id, timestamps=batch.timestamps,
            anomaly_score=self._cal(d),
            explanation="distance to Deep SVDD hypersphere center (robust z)",
        )


# ---------------------------------------------------------------------------
# m14 — Isolation Forest
# ---------------------------------------------------------------------------
class IsolationForestTelemetry(BaseWTModel):
    spec = ModelSpec(
        model_id="m14-isolation-forest",
        repository="wt-pm-isolation-forest-telemetry",
        task=TaskType.ANOMALY_DETECTION,
        input_requirements=["features"],
        output_schema=["anomaly_score"],
        resource_requirements=["scikit-learn"],
        typical_latency_ms=400,
        deployment_targets=[Deployment.CLOUD, Deployment.EDGE_CPU],
        notes="cheapest detector; is the fallback for the heavy ones",
    )

    def _check_deps(self) -> None:
        from sklearn.ensemble import IsolationForest  # noqa: F401

    def fit(self, batch: SensorBatch, train_mask: np.ndarray) -> None:
        # The repo's train_isolation_forest() reads a CSV and fillna(0)s it;
        # we keep its estimator settings but feed clean masked features
        # (a 0-filled gap is a fake healthy reading — see INSPECTION.md).
        from sklearn.ensemble import IsolationForest
        X = batch.features[train_mask]
        self._clf = IsolationForest(n_estimators=200, contamination=0.02,
                                    random_state=42, n_jobs=-1).fit(X)
        d = -self._clf.decision_function(X)
        self._cal = robust_calibrate(d)
        self._fitted = True

    def _predict(self, batch: SensorBatch) -> ModelOutput:
        d = -self._clf.decision_function(batch.features)
        return ModelOutput(
            model_id=self.spec.model_id, task=self.spec.task,
            turbine_id=batch.turbine_id, timestamps=batch.timestamps,
            anomaly_score=self._cal(d),
            explanation="isolation-forest outlier score (robust z)",
        )


# ---------------------------------------------------------------------------
# m22 — VAE reconstruction loss
# ---------------------------------------------------------------------------
class VAEReconstruction(BaseWTModel):
    spec = ModelSpec(
        model_id="m22-vae-reconstruction",
        repository="wt-pm-vae-reconstruction-loss",
        task=TaskType.ANOMALY_DETECTION,
        input_requirements=["features"],
        output_schema=["anomaly_score", "uncertainty"],
        resource_requirements=["torch"],
        typical_latency_ms=1500,
        deployment_targets=[Deployment.CLOUD, Deployment.EDGE_GPU],
        fallback="m14-isolation-forest",
    )

    def _check_deps(self) -> None:
        import torch  # noqa: F401
        load_repo_module("wt-pm-vae-reconstruction-loss")

    def fit(self, batch: SensorBatch, train_mask: np.ndarray) -> None:
        import torch
        mod = load_repo_module("wt-pm-vae-reconstruction-loss")
        X = batch.features[train_mask]
        self._std = _standardiser(X)
        Xs = torch.tensor(self._std(X), dtype=torch.float32)
        model = mod.VAEAnomalyDetector(input_dim=X.shape[1], hidden_dim=32, latent_dim=8)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(60):
            opt.zero_grad()
            recon, mu, logvar = model(Xs)
            loss = mod.vae_loss(recon, Xs, mu, logvar) / len(Xs)  # repo's loss
            loss.backward()
            opt.step()
        self._model = model
        with torch.no_grad():
            recon, mu, logvar = model(Xs)
            err = ((recon - Xs) ** 2).mean(dim=1).numpy()
        self._cal = robust_calibrate(err)
        self._fitted = True

    def _predict(self, batch: SensorBatch) -> ModelOutput:
        import torch
        Xs = torch.tensor(self._std(batch.features), dtype=torch.float32)
        with torch.no_grad():
            recon, mu, logvar = self._model(Xs)
            err = ((recon - Xs) ** 2).mean(dim=1).numpy()
            unc = logvar.exp().mean(dim=1).sqrt().numpy()
        return ModelOutput(
            model_id=self.spec.model_id, task=self.spec.task,
            turbine_id=batch.turbine_id, timestamps=batch.timestamps,
            anomaly_score=self._cal(err), uncertainty=unc,
            explanation="VAE reconstruction error (robust z)",
        )


def _to_steps(window_scores: np.ndarray, ends: np.ndarray, n_steps: int) -> np.ndarray:
    """Spread window-level scores onto per-step scores (hold last value)."""
    out = np.zeros(n_steps)
    if len(ends) == 0:
        return out
    j = 0
    cur = window_scores[0]
    for t in range(n_steps):
        while j < len(ends) - 1 and ends[j + 1] <= t:
            j += 1
            cur = window_scores[j]
        out[t] = cur if t >= ends[0] else window_scores[0]
    return out
