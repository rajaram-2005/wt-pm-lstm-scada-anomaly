"""Adapters for degradation-state and RUL models.

m15 hmm-degradation-states — GaussianHMM over health indicators
m17 mlp-rul-regression     — sklearn MLP baseline RUL
m02 convlstm-wear-prognostics — Keras ConvLSTM RUL (spatiotemporal, partial)
m16 particle-filter-rul    — probabilistic fusion of RUL observations
m06 informer-long-sequence — long-horizon power forecaster (partial attention)

RUL ground truth does not exist in any repo, so research mode supervises these
on a clearly-labelled proxy: time-to-next-fault-event from the simulator's
fault schedule (or a monotone transform of the health indicator when no event
follows). Every output states this in its explanation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from wtpm_platform.base import BaseWTModel, ModelSpec, load_repo_module
from wtpm_platform.contracts import Deployment, ModelOutput, SensorBatch, Subsystem, TaskType
from wtpm_platform.data import degradation_target, fault_kinds_from_batch
from wtpm_platform.adapters.anomaly import _standardiser

SAMPLE_HOURS = 1.0 / 6.0  # 10-minute cadence


def rul_proxy_hours(batch: SensorBatch, horizon_hours: float = 400.0) -> np.ndarray:
    """Time to next fault onset (hours), capped; proxy target, not real life data."""
    kinds = fault_kinds_from_batch(batch)
    n = batch.n_steps
    rul = np.full(n, horizon_hours)
    if kinds is None:
        return rul
    onsets = [i for i in range(1, n) if kinds[i] and not kinds[i - 1]]
    for t in range(n):
        nxt = [o for o in onsets if o >= t]
        if nxt:
            rul[t] = min((nxt[0] - t) * SAMPLE_HOURS, horizon_hours)
        if kinds[t]:
            rul[t] = 0.0
    return rul


# ---------------------------------------------------------------------------
# m15 — HMM degradation states
# ---------------------------------------------------------------------------
class HMMDegradationStates(BaseWTModel):
    spec = ModelSpec(
        model_id="m15-hmm-degradation",
        repository="wt-pm-hmm-degradation-states",
        task=TaskType.DEGRADATION_STATE,
        input_requirements=["health-indicator series"],
        output_schema=["degradation_state", "probability(state)"],
        resource_requirements=["hmmlearn"],
        typical_latency_ms=1500,
        deployment_targets=[Deployment.CLOUD],
        notes="states re-ordered by mean health indicator so index == severity",
    )

    N_STATES = 4

    def _check_deps(self) -> None:
        from hmmlearn.hmm import GaussianHMM  # noqa: F401

    def _obs(self, batch: SensorBatch) -> np.ndarray:
        hi = degradation_target(batch)
        cols = [hi]
        for ch in ("bearing_vib_rms_mm_s", "gearbox_oil_temp_c"):
            if ch in batch.channel_names:
                cols.append(batch.channel(ch))
        return np.column_stack(cols)

    def fit(self, batch: SensorBatch, train_mask: np.ndarray) -> None:
        from hmmlearn.hmm import GaussianHMM
        obs = self._obs(batch)
        # same estimator settings as the repo's train_hmm()
        self._hmm = GaussianHMM(n_components=self.N_STATES, covariance_type="diag",
                                n_iter=100, random_state=42)
        self._hmm.fit(obs)
        # order states by mean health indicator => monotone severity
        order = np.argsort(self._hmm.means_[:, 0])
        self._rank = {s: r for r, s in enumerate(order)}
        self._fitted = True

    def _predict(self, batch: SensorBatch) -> ModelOutput:
        obs = self._obs(batch)
        raw = self._hmm.predict(obs)
        post = self._hmm.predict_proba(obs)
        states = np.array([self._rank[s] for s in raw])
        conf = post.max(axis=1)
        return ModelOutput(
            model_id=self.spec.model_id, task=self.spec.task,
            turbine_id=batch.turbine_id, timestamps=batch.timestamps,
            degradation_state=states,
            uncertainty=1.0 - conf,
            explanation=f"HMM {self.N_STATES}-state degradation (0=healthy .. {self.N_STATES-1}=severe), "
                        "severity-ordered by health indicator",
            extra={"posterior": post, "transmat": self._hmm.transmat_},
        )


# ---------------------------------------------------------------------------
# m17 — MLP RUL baseline
# ---------------------------------------------------------------------------
class MLPRULRegression(BaseWTModel):
    spec = ModelSpec(
        model_id="m17-mlp-rul",
        repository="wt-pm-mlp-rul-regression",
        task=TaskType.RUL_ESTIMATION,
        input_requirements=["features", "RUL proxy target (research mode)"],
        output_schema=["rul_hours"],
        resource_requirements=["scikit-learn"],
        typical_latency_ms=1000,
        deployment_targets=[Deployment.CLOUD, Deployment.EDGE_CPU],
        notes="the baseline every other RUL model must beat",
    )

    def _check_deps(self) -> None:
        from sklearn.neural_network import MLPRegressor  # noqa: F401

    def fit(self, batch: SensorBatch, train_mask: np.ndarray) -> None:
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        y = rul_proxy_hours(batch)
        # repo pipeline: StandardScaler + MLP(128, 64)
        self._pipe = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                         max_iter=300, random_state=42),
        )
        self._pipe.fit(batch.features, y)
        self._fitted = True

    def _predict(self, batch: SensorBatch) -> ModelOutput:
        rul = np.maximum(self._pipe.predict(batch.features), 0.0)
        return ModelOutput(
            model_id=self.spec.model_id, task=self.spec.task,
            turbine_id=batch.turbine_id, timestamps=batch.timestamps,
            rul_hours=rul,
            explanation="MLP RUL regression on tabular features (proxy-target trained)",
        )


# ---------------------------------------------------------------------------
# m02 — ConvLSTM wear prognostics (partial: needs spatial wear maps)
# ---------------------------------------------------------------------------
class ConvLSTMWearPrognostics(BaseWTModel):
    spec = ModelSpec(
        model_id="m02-convlstm-wear",
        repository="wt-pm-convlstm-wear-prognostics",
        task=TaskType.RUL_ESTIMATION,
        input_requirements=["windows -> recurrence images"],
        output_schema=["rul_hours"],
        resource_requirements=["tensorflow"],
        typical_latency_ms=15000,
        deployment_targets=[Deployment.CLOUD],
        fallback="m17-mlp-rul",
        notes="PARTIAL: repo expects (10,64,64,1) wear maps that no data source "
              "provides; platform builds 16x16 channel-recurrence images instead",
    )

    GRID = 16
    SEQ = 6

    def _check_deps(self) -> None:
        import tensorflow  # noqa: F401
        load_repo_module("wt-pm-convlstm-wear-prognostics")

    def _images(self, batch: SensorBatch) -> np.ndarray:
        """Down-sampled recurrence-style images from each window (T,C)->(G,G)."""
        W = batch.windows
        n, T, C = W.shape
        G = self.GRID
        Xn = (W - W.mean(axis=1, keepdims=True)) / (W.std(axis=1, keepdims=True) + 1e-9)
        idx = np.linspace(0, T - 1, G).astype(int)
        sub = Xn[:, idx, :]                       # (n, G, C)
        img = np.einsum("ngc,nhc->ngh", sub, sub) / C   # recurrence matrix
        return img.astype(np.float32)

    def fit(self, batch: SensorBatch, train_mask: np.ndarray) -> None:
        mod = load_repo_module("wt-pm-convlstm-wear-prognostics")
        imgs = self._images(batch)
        ends = batch.window_index
        y = rul_proxy_hours(batch)[ends]
        S = self.SEQ
        seqs, ys = [], []
        for i in range(S, len(imgs)):
            seqs.append(imgs[i - S:i])
            ys.append(y[i])
        Xs = np.array(seqs)[..., None]
        model = mod.build_convlstm(input_shape=(S, self.GRID, self.GRID, 1))
        model.fit(Xs, np.array(ys), epochs=3, batch_size=32, verbose=0)
        self._model = model
        self._fitted = True

    def _predict(self, batch: SensorBatch) -> ModelOutput:
        imgs = self._images(batch)
        ends = batch.window_index
        S = self.SEQ
        seqs = [imgs[i - S:i] for i in range(S, len(imgs))]
        if not seqs:
            raise RuntimeError("record too short for ConvLSTM sequence")
        pred = self._model.predict(np.array(seqs)[..., None], verbose=0).ravel()
        pred = np.maximum(pred, 0.0)
        n = batch.n_steps
        rul = np.full(n, np.nan)
        for j, i in enumerate(range(S, len(imgs))):
            rul[ends[i]] = pred[j]
        # forward-fill
        last = pred[0]
        for t in range(n):
            if np.isfinite(rul[t]):
                last = rul[t]
            rul[t] = last
        return ModelOutput(
            model_id=self.spec.model_id, task=self.spec.task,
            turbine_id=batch.turbine_id, timestamps=batch.timestamps,
            rul_hours=rul,
            explanation="ConvLSTM RUL on channel-recurrence images (partial integration; proxy target)",
        )


# ---------------------------------------------------------------------------
# m16 — Particle filter RUL (fusion of RUL observations)
# ---------------------------------------------------------------------------
class ParticleFilterRUL(BaseWTModel):
    spec = ModelSpec(
        model_id="m16-particle-filter-rul",
        repository="wt-pm-particle-filter-rul",
        task=TaskType.RUL_ESTIMATION,
        input_requirements=["RUL observations from m17/m02", "degradation rate from m15"],
        output_schema=["rul_hours", "uncertainty"],
        resource_requirements=["numpy"],
        typical_latency_ms=800,
        deployment_targets=[Deployment.CLOUD, Deployment.EDGE_CPU],
        notes="consumes sibling RUL estimates as observations; pure numpy",
    )

    def _check_deps(self) -> None:
        load_repo_module("wt-pm-particle-filter-rul")

    def fit(self, batch: SensorBatch, train_mask: np.ndarray) -> None:
        self._fitted = True  # stateless: filter is (re)initialised per timeline

    def _predict(self, batch: SensorBatch) -> ModelOutput:
        mod = load_repo_module("wt-pm-particle-filter-rul")
        obs = batch.meta.get("rul_observations")  # (T,) from upstream models
        deg_state = batch.meta.get("degradation_state")  # (T,) from m15
        n = batch.n_steps
        if obs is None:
            raise RuntimeError("m16 requires meta['rul_observations'] from an upstream RUL model")
        init = float(np.nanmax(obs[:24])) if np.isfinite(obs[:24]).any() else 400.0
        pf = mod.ParticleFilterRUL(num_particles=800, init_rul=max(init, 1.0))
        est = np.zeros(n)
        std = np.zeros(n)
        for t in range(n):
            rate = SAMPLE_HOURS
            if deg_state is not None:      # degrade faster in worse states
                rate *= (1.0 + 0.8 * float(deg_state[t]))
            pf.predict(degradation_rate=rate, noise=0.5)
            if np.isfinite(obs[t]) and t % 3 == 0:
                # Upstream numerical guard (repo bug, worked around here, not
                # patched in the repo): when the observation is far from every
                # particle the Gaussian likelihood underflows; update() then
                # normalises by (sum + 1e-12), leaving weights that do not sum
                # to 1, and resample()'s np.random.choice rejects them. The
                # guard must therefore trigger whenever the likelihood mass is
                # small relative to that 1e-12 epsilon, not merely at exact 0.
                lik = np.exp(-0.5 * ((pf.particles - float(obs[t])) / 25.0) ** 2)
                if float(lik.sum()) < 1e-6:
                    pf.particles = np.random.default_rng(t).normal(
                        float(obs[t]), 25.0, size=pf.num_particles)
                    pf.particles = np.maximum(pf.particles, 0)
                    pf.weights = np.ones(pf.num_particles) / pf.num_particles
                else:
                    pf.update(observed_rul=float(obs[t]), obs_noise=25.0)
            est[t] = pf.estimate()
            std[t] = float(np.std(pf.particles))
        return ModelOutput(
            model_id=self.spec.model_id, task=self.spec.task,
            turbine_id=batch.turbine_id, timestamps=batch.timestamps,
            rul_hours=est, uncertainty=std,
            explanation="particle-filter RUL posterior fusing upstream RUL observations "
                        "with degradation-state-dependent decay",
        )


# ---------------------------------------------------------------------------
# m06 — Informer long-sequence power forecasting (partial)
# ---------------------------------------------------------------------------
class InformerLongSequence(BaseWTModel):
    spec = ModelSpec(
        model_id="m06-informer-forecast",
        repository="wt-pm-informer-long-sequence",
        task=TaskType.FORECASTING,
        input_requirements=["windows"],
        output_schema=["prediction(power forecast)", "anomaly_score(forecast residual)"],
        resource_requirements=["torch"],
        typical_latency_ms=8000,
        deployment_targets=[Deployment.CLOUD],
        fallback="m03-tcn-power-curve",
        notes="PARTIAL: repo's ProbSparse attention is a self-declared placeholder",
    )

    def _check_deps(self) -> None:
        import torch  # noqa: F401
        load_repo_module("wt-pm-informer-long-sequence")

    def fit(self, batch: SensorBatch, train_mask: np.ndarray) -> None:
        import torch
        mod = load_repo_module("wt-pm-informer-long-sequence")
        W, ends = batch.windows, batch.window_index
        p_idx = list(batch.channel_names).index("power_kw")
        keep = train_mask[ends]
        X = W[keep]
        self._std = _standardiser(X.reshape(-1, X.shape[-1]))
        Xs = self._std(X)
        # d_model=64 (not the repo's 512) to keep CPU latency honest
        model = mod.InformerPowerForecaster(enc_in=X.shape[-1], dec_in=X.shape[-1],
                                            c_out=1, d_model=64)
        # Upstream quirk (unmodified, worked around here): the repo's forward
        # embeds only x_enc, so the decoder input must arrive pre-embedded.
        def fwd(xe_raw, xd_raw):
            xd_emb = model.enc_embedding(xd_raw)
            return model(xe_raw, xd_emb)

        self._fwd = fwd
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        xe = torch.tensor(Xs[:, :-1, :], dtype=torch.float32)
        xd = torch.tensor(Xs[:, -1:, :], dtype=torch.float32)
        yb = torch.tensor(Xs[:, -1, p_idx], dtype=torch.float32)
        for _ in range(20):
            opt.zero_grad()
            out = fwd(xe, xd).squeeze(-1).squeeze(-1)
            loss = torch.nn.functional.mse_loss(out, yb)
            loss.backward()
            opt.step()
        self._model, self._p_idx = model, p_idx
        with torch.no_grad():
            err = np.abs(fwd(xe, xd).squeeze(-1).squeeze(-1).numpy() - yb.numpy())
        from wtpm_platform.adapters.anomaly import robust_calibrate
        self._cal = robust_calibrate(err)
        self._fitted = True

    def _predict(self, batch: SensorBatch) -> ModelOutput:
        import torch
        from wtpm_platform.adapters.anomaly import _to_steps
        W, ends = batch.windows, batch.window_index
        Xs = self._std(W)
        xe = torch.tensor(Xs[:, :-1, :], dtype=torch.float32)
        xd = torch.tensor(Xs[:, -1:, :], dtype=torch.float32)
        with torch.no_grad():
            pred = self._fwd(xe, xd).squeeze(-1).squeeze(-1).numpy()
        err = np.abs(pred - Xs[:, -1, self._p_idx])
        score = _to_steps(self._cal(err), ends, batch.n_steps)
        return ModelOutput(
            model_id=self.spec.model_id, task=self.spec.task,
            turbine_id=batch.turbine_id, timestamps=batch.timestamps,
            anomaly_score=score,
            explanation="Informer power-forecast residual (placeholder attention; partial)",
        )
