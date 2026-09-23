"""Adapters for fault classification and representation learning.

m10 random-forest-telemetry — sklearn RF (feature importances kept for XAI)
m11 xgboost-tabular-faults  — XGBoost multi-class fault classifier
m12 svm-rbf-generator-stator — SVM-RBF on electrical features (stator focus)
m01 1d-cnn-bearing-vibration — Keras 1D-CNN on vibration waveforms (surrogate)
m07 snn-event-vibration      — spiking net on event-encoded vibration
m08 contrastive-ssl-vibration — SSL encoder; embeddings feed tabular models
m09 dbn-feature-extraction   — DBN/RBM features (adds the CD-k loop the repo omits)
m23 aerozip-autoencoder-compressor — telemetry compression; latent features

Classifiers are supervised: in research mode they train on simulator labels;
in production mode (no labels) the router simply does not select them, or they
run with the last research-mode weights.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from wtpm_platform.base import BaseWTModel, ModelSpec, load_repo_module
from wtpm_platform.contracts import (
    Deployment, FAULT_CLASSES, ModelOutput, SensorBatch, Subsystem, TaskType, VIB_CLASSES,
)
from wtpm_platform.data import ELECTRICAL_CHANNELS, labels_from_batch, fault_kinds_from_batch
from wtpm_platform.adapters.anomaly import _standardiser, robust_calibrate
from wtpm_platform.protocol import training_mask, hold_probabilities


def _step_labels(batch: SensorBatch) -> Optional[np.ndarray]:
    """Per-step integer fault labels (index into FAULT_CLASSES), or None."""
    kinds = fault_kinds_from_batch(batch)
    if kinds is None:
        return None
    lut = {k: i for i, k in enumerate(FAULT_CLASSES)}
    unknown = set(kinds) - set(lut) - {""}
    if unknown:
        raise ValueError(f"unknown fault labels: {unknown}")
    return np.array([lut[k or "healthy"] for k in kinds])


def _proba_dict(proba: np.ndarray, classes: List[str], class_ids: np.ndarray) -> Dict[str, np.ndarray]:
    out = {c: np.zeros(proba.shape[0]) for c in classes}
    for j, ci in enumerate(class_ids):
        out[classes[int(ci)]] = proba[:, j]
    return out


# ---------------------------------------------------------------------------
# m11 — XGBoost tabular faults (primary tabular classifier)
# ---------------------------------------------------------------------------
class XGBoostTabularFaults(BaseWTModel):
    spec = ModelSpec(
        model_id="m11-xgboost-tabular",
        repository="wt-pm-xgboost-tabular-faults",
        task=TaskType.FAULT_CLASSIFICATION,
        input_requirements=["features", "labels (research mode)"],
        output_schema=["prediction", "probability"],
        resource_requirements=["xgboost", "scikit-learn"],
        typical_latency_ms=800,
        deployment_targets=[Deployment.CLOUD, Deployment.EDGE_CPU],
        fallback="m10-random-forest",
        notes="SHAP TreeExplainer-compatible; primary input to the XAI engine",
    )

    def _check_deps(self) -> None:
        import xgboost  # noqa: F401

    def fit(self, batch: SensorBatch, train_mask: np.ndarray) -> None:
        import xgboost as xgb
        y = _step_labels(batch)
        if y is None:
            raise RuntimeError("m11 needs labels; fit in research mode first")
        # same estimator family/params as the repo's train_xgboost()
        mask = training_mask(batch, train_mask)
        X, y = batch.features[mask], y[mask]
        if len(np.unique(y)) < 2:
            raise ValueError("XGBoost needs at least two labelled training classes")
        present = np.unique(y)
        self._class_ids = present
        remap = {c: i for i, c in enumerate(present)}
        self._clf = xgb.XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.08, subsample=0.8,
            eval_metric="mlogloss", n_jobs=2,
        )
        self._clf.fit(X, np.array([remap[v] for v in y]))
        self._fitted = True

    def _predict(self, batch: SensorBatch) -> ModelOutput:
        proba = self._clf.predict_proba(batch.features)
        pred_ids = self._class_ids[np.argmax(proba, axis=1)]
        pred = np.array([FAULT_CLASSES[i] for i in pred_ids], dtype=object)
        return ModelOutput(
            model_id=self.spec.model_id, task=self.spec.task,
            turbine_id=batch.turbine_id, timestamps=batch.timestamps,
            prediction=pred,
            probability=_proba_dict(proba, FAULT_CLASSES, self._class_ids),
            explanation="XGBoost multi-class fault probabilities",
            extra={"estimator": self._clf, "feature_names": list(batch.feature_names)},
        )


# ---------------------------------------------------------------------------
# m10 — Random forest telemetry
# ---------------------------------------------------------------------------
class RandomForestTelemetry(BaseWTModel):
    spec = ModelSpec(
        model_id="m10-random-forest",
        repository="wt-pm-random-forest-telemetry",
        task=TaskType.FAULT_CLASSIFICATION,
        input_requirements=["features", "labels (research mode)"],
        output_schema=["prediction", "probability", "feature_importances"],
        resource_requirements=["scikit-learn"],
        typical_latency_ms=600,
        deployment_targets=[Deployment.CLOUD, Deployment.EDGE_CPU],
        notes="feature importances feed the XAI evidence bundle",
    )

    def _check_deps(self) -> None:
        from sklearn.ensemble import RandomForestClassifier  # noqa: F401

    def fit(self, batch: SensorBatch, train_mask: np.ndarray) -> None:
        from sklearn.ensemble import RandomForestClassifier
        y = _step_labels(batch)
        if y is None:
            raise RuntimeError("m10 needs labels; fit in research mode first")
        mask = training_mask(batch, train_mask)
        y = y[mask]
        present = np.unique(y)
        self._class_ids = present
        # repo settings: n_estimators=200, max_depth=None
        self._clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        self._clf.fit(batch.features[mask], y)
        self._importances = dict(zip(batch.feature_names, self._clf.feature_importances_))
        self._fitted = True

    def _predict(self, batch: SensorBatch) -> ModelOutput:
        proba = self._clf.predict_proba(batch.features)
        pred_ids = self._clf.classes_[np.argmax(proba, axis=1)]
        pred = np.array([FAULT_CLASSES[i] for i in pred_ids], dtype=object)
        top = sorted(self._importances.items(), key=lambda kv: -kv[1])[:5]
        return ModelOutput(
            model_id=self.spec.model_id, task=self.spec.task,
            turbine_id=batch.turbine_id, timestamps=batch.timestamps,
            prediction=pred,
            probability=_proba_dict(proba, FAULT_CLASSES, self._clf.classes_),
            explanation="RF fault probabilities; top features: "
                        + ", ".join(f"{k}={v:.3f}" for k, v in top),
            extra={"estimator": self._clf, "feature_importances": self._importances},
        )


# ---------------------------------------------------------------------------
# m12 — SVM-RBF generator/stator
# ---------------------------------------------------------------------------
class SVMGeneratorStator(BaseWTModel):
    spec = ModelSpec(
        model_id="m12-svm-generator",
        repository="wt-pm-svm-rbf-generator-stator",
        task=TaskType.FAULT_CLASSIFICATION,
        input_requirements=["electrical features", "labels (research mode)"],
        output_schema=["prediction", "probability"],
        resource_requirements=["scikit-learn"],
        typical_latency_ms=900,
        deployment_targets=[Deployment.CLOUD],
        fallback="m11-xgboost-tabular",
        subsystem_focus=["generator", "converter"],
        notes="binary electrical-fault detector on generator/converter channels",
    )

    def _check_deps(self) -> None:
        from sklearn.svm import SVC  # noqa: F401

    def _elec(self, batch: SensorBatch) -> np.ndarray:
        cols = [i for i, n in enumerate(batch.feature_names)
                if any(n.startswith(c) for c in ELECTRICAL_CHANNELS)]
        return batch.features[:, cols]

    def fit(self, batch: SensorBatch, train_mask: np.ndarray) -> None:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        y = _step_labels(batch)
        if y is None:
            raise RuntimeError("m12 needs labels; fit in research mode first")
        mask = training_mask(batch, train_mask)
        elec_fault = np.isin(np.array(FAULT_CLASSES, dtype=object)[y[mask]], ["converter_fault"])
        if len(np.unique(elec_fault)) != 2:
            raise ValueError("SVM requires labelled healthy and electrical-fault training examples; no synthetic labels are invented")
        self._weak_labels = False
        X = self._elec(batch)[mask]
        # subsample for SVM tractability while keeping every fault example;
        # identical pipeline to the repo
        rng = np.random.default_rng(0)
        pos = np.flatnonzero(elec_fault)
        neg = rng.choice(np.flatnonzero(~elec_fault),
                         size=min(4000, int((~elec_fault).sum())), replace=False)
        idx = np.concatenate([pos, neg])
        self._pipe = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=42),
        )
        self._pipe.fit(X[idx], elec_fault[idx].astype(int))
        self._fitted = True

    def _predict(self, batch: SensorBatch) -> ModelOutput:
        proba = self._pipe.predict_proba(self._elec(batch))
        p_fault = proba[:, list(self._pipe.classes_).index(1)] if 1 in self._pipe.classes_ else np.zeros(len(proba))
        pred = np.where(p_fault > 0.5, "converter_fault", "not_electrical").astype(object)
        return ModelOutput(
            model_id=self.spec.model_id, task=self.spec.task,
            turbine_id=batch.turbine_id, timestamps=batch.timestamps,
            prediction=pred, subsystem=Subsystem.GENERATOR,
            probability={"converter_fault": p_fault, "not_electrical": 1 - p_fault},
            explanation=("SVM-RBF electrical-fault probability (generator/converter channels only)"
                         + (" [weak labels: generator-current residual]" if getattr(self, "_weak_labels", False) else "")),
        )


# ---------------------------------------------------------------------------
# m01 — 1D CNN bearing vibration (Keras)
# ---------------------------------------------------------------------------
class CNN1DBearingVibration(BaseWTModel):
    spec = ModelSpec(
        model_id="m01-1dcnn-bearing",
        repository="wt-pm-1d-cnn-bearing-vibration",
        task=TaskType.FAULT_CLASSIFICATION,
        input_requirements=["vib_waveforms (surrogate)", "labels (research mode)"],
        output_schema=["prediction", "probability"],
        resource_requirements=["tensorflow"],
        typical_latency_ms=6000,
        deployment_targets=[Deployment.CLOUD, Deployment.EDGE_GPU],
        fallback="m11-xgboost-tabular",
        subsystem_focus=["drivetrain"],
        notes="trained on surrogate waveforms derived from 10-min vibration RMS; "
              "real high-frequency DAQ data would slot in unchanged",
    )

    def _check_deps(self) -> None:
        import tensorflow  # noqa: F401
        load_repo_module("wt-pm-1d-cnn-bearing-vibration")

    def fit(self, batch: SensorBatch, train_mask: np.ndarray) -> None:
        mod = load_repo_module("wt-pm-1d-cnn-bearing-vibration")
        waves, idx = batch.vib_waveforms, batch.vib_index
        if waves is None:
            raise RuntimeError("no vibration view available")
        labels = _step_labels(batch)
        if labels is None:
            raise ValueError("CNN requires labelled training examples")
        keep = training_mask(batch, train_mask)[idx]
        if not keep.any():
            raise ValueError("no training waveforms")
        model = mod.build_1d_cnn((waves.shape[1], 1), num_classes=len(FAULT_CLASSES))
        import tensorflow as tf
        yoh = tf.keras.utils.to_categorical(labels[idx][keep], num_classes=len(FAULT_CLASSES))
        model.fit(waves[keep, ..., None], yoh, epochs=4, batch_size=64, verbose=0)
        self._model = model
        self._fitted = True

    def _predict(self, batch: SensorBatch) -> ModelOutput:
        waves, idx = batch.vib_waveforms, batch.vib_index
        proba = self._model.predict(waves[..., None], verbose=0)
        p = hold_probabilities(proba, idx, batch.n_steps)
        return ModelOutput(
            model_id=self.spec.model_id, task=self.spec.task,
            turbine_id=batch.turbine_id, timestamps=batch.timestamps,
            prediction=np.asarray(FAULT_CLASSES, object)[p.argmax(1)],
            probability={c: p[:, i] for i, c in enumerate(FAULT_CLASSES)},
            subsystem=Subsystem.DRIVETRAIN,
            explanation="CNN canonical fault probabilities from vibration surrogates; uniform during warmup",
        )


# ---------------------------------------------------------------------------
# m07 — SNN event vibration
# ---------------------------------------------------------------------------
class SNNEventVibration(BaseWTModel):
    spec = ModelSpec(
        model_id="m07-snn-vibration",
        repository="wt-pm-snn-event-vibration",
        task=TaskType.FAULT_CLASSIFICATION,
        input_requirements=["vib_waveforms (event-encoded)"],
        output_schema=["prediction", "probability(spike-rate)"],
        resource_requirements=["snntorch", "torch"],
        typical_latency_ms=5000,
        deployment_targets=[Deployment.CLOUD, Deployment.EDGE_CPU, Deployment.EDGE_GPU],
        fallback="m01-1dcnn-bearing",
        subsystem_focus=["drivetrain"],
        notes="delta/threshold event encoding; differentiable membrane logits as class evidence",
    )

    def _check_deps(self) -> None:
        import snntorch  # noqa: F401
        mod = load_repo_module("wt-pm-snn-event-vibration")
        if not hasattr(mod, "EventSNN"):
            raise ImportError("EventSNN not defined (snntorch missing at repo import)")

    @staticmethod
    def _events(waves: np.ndarray, bins: int = 64, threshold=None) -> np.ndarray:
        """Delta-threshold event encoding: fraction of threshold crossings per bin."""
        d = np.abs(np.diff(waves, axis=1))
        thr = np.percentile(d, 90) if threshold is None else threshold
        ev = (d > thr).astype(np.float32)
        L = ev.shape[1] // bins * bins
        return ev[:, :L].reshape(ev.shape[0], bins, -1).mean(axis=2)

    def fit(self, batch: SensorBatch, train_mask: np.ndarray) -> None:
        import torch
        mod = load_repo_module("wt-pm-snn-event-vibration")
        waves, idx = batch.vib_waveforms, batch.vib_index
        labels = _step_labels(batch)
        if labels is None or waves is None:
            raise ValueError("SNN requires labelled training waveforms")
        keep = training_mask(batch, train_mask)[idx]
        if not keep.any():
            raise ValueError("no training waveforms")
        waves, y = waves[keep], labels[idx][keep]
        self._event_threshold = float(np.percentile(np.abs(np.diff(waves, axis=1)), 90))
        X = self._events(waves, threshold=self._event_threshold)
        net = mod.EventSNN(input_size=X.shape[1], hidden=64, output_size=len(FAULT_CLASSES))
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        xb = torch.tensor(X, dtype=torch.float32)
        yb = torch.tensor(y)
        for _ in range(40):
            opt.zero_grad()
            spk, m1, m2 = net(xb)
            loss = torch.nn.functional.cross_entropy(m2, yb)
            loss.backward()
            opt.step()
        self._net = net
        self._fitted = True

    def _predict(self, batch: SensorBatch) -> ModelOutput:
        import torch
        waves, idx = batch.vib_waveforms, batch.vib_index
        X = self._events(waves, threshold=self._event_threshold)
        self._net.eval()
        with torch.no_grad():
            _, _, membrane = self._net(torch.tensor(X, dtype=torch.float32))
            proba = torch.softmax(membrane, dim=1).numpy()
        p = hold_probabilities(proba, idx, batch.n_steps)
        return ModelOutput(
            model_id=self.spec.model_id, task=self.spec.task,
            turbine_id=batch.turbine_id, timestamps=batch.timestamps,
            prediction=np.asarray(FAULT_CLASSES, object)[p.argmax(1)],
            probability={c: p[:, i] for i, c in enumerate(FAULT_CLASSES)},
            subsystem=Subsystem.DRIVETRAIN,
            explanation="SNN membrane class evidence; event threshold fixed on training data",
        )


def nt_xent(z1, z2, temperature=0.5):
    import torch
    if len(z1) < 2 or temperature <= 0:
        raise ValueError("NT-Xent needs at least two pairs and positive temperature")
    z = torch.nn.functional.normalize(torch.cat([z1, z2]), dim=1)
    logits = z @ z.T / temperature
    logits = logits.masked_fill(torch.eye(len(z), dtype=torch.bool, device=z.device), float('-inf'))
    targets = (torch.arange(len(z), device=z.device) + len(z1)) % len(z)
    return torch.nn.functional.cross_entropy(logits, targets)


# ---------------------------------------------------------------------------
# m08 — Contrastive SSL vibration encoder (feature provider)
# ---------------------------------------------------------------------------
class ContrastiveSSLVibration(BaseWTModel):
    spec = ModelSpec(
        model_id="m08-contrastive-ssl",
        repository="wt-pm-contrastive-ssl-vibration",
        task=TaskType.FEATURE_EXTRACTION,
        input_requirements=["vib_waveforms (unlabelled)"],
        output_schema=["extracted_features"],
        resource_requirements=["torch"],
        typical_latency_ms=4000,
        deployment_targets=[Deployment.CLOUD, Deployment.EDGE_GPU],
        notes="upstream encoder with corrected NT-Xent positive/negative-pair loss",
    )

    def _check_deps(self) -> None:
        import torch  # noqa: F401
        load_repo_module("wt-pm-contrastive-ssl-vibration")

    def fit(self, batch: SensorBatch, train_mask: np.ndarray) -> None:
        import torch
        mod = load_repo_module("wt-pm-contrastive-ssl-vibration")
        waves = batch.vib_waveforms[training_mask(batch, train_mask)[batch.vib_index]]
        enc = mod.ContrastiveEncoder(input_dim=waves.shape[1], proj_dim=32)
        opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
        xb = torch.tensor(waves[:, None, :], dtype=torch.float32)
        rng = np.random.default_rng(0)
        for _ in range(15):
            # two stochastic augmentations: jitter + scaling
            a1 = xb + 0.05 * torch.randn_like(xb)
            a2 = xb * float(rng.uniform(0.9, 1.1)) + 0.05 * torch.randn_like(xb)
            z1, z2 = enc(a1), enc(a2)
            loss = nt_xent(z1, z2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        enc.eval()
        self._enc = enc
        self._fitted = True

    def _predict(self, batch: SensorBatch) -> ModelOutput:
        import torch
        with torch.no_grad():
            z = self._enc(torch.tensor(batch.vib_waveforms[:, None, :], dtype=torch.float32)).numpy()
        return ModelOutput(
            model_id=self.spec.model_id, task=self.spec.task,
            turbine_id=batch.turbine_id,
            timestamps=batch.timestamps[batch.vib_index],
            explanation="contrastive embeddings of vibration surrogates",
            extra={"embeddings": z, "embedding_index": batch.vib_index},
        )


# ---------------------------------------------------------------------------
# m09 — DBN feature extraction (adds CD-1 pretraining the repo omits)
# ---------------------------------------------------------------------------
class DBNFeatureExtraction(BaseWTModel):
    spec = ModelSpec(
        model_id="m09-dbn-features",
        repository="wt-pm-dbn-feature-extraction",
        task=TaskType.FEATURE_EXTRACTION,
        input_requirements=["features"],
        output_schema=["extracted_features"],
        resource_requirements=["torch"],
        typical_latency_ms=3000,
        deployment_targets=[Deployment.CLOUD],
        notes="platform supplies the contrastive-divergence loop; RBM/DBN classes are the repo's",
    )

    def _check_deps(self) -> None:
        import torch  # noqa: F401
        load_repo_module("wt-pm-dbn-feature-extraction")

    def fit(self, batch: SensorBatch, train_mask: np.ndarray) -> None:
        import torch
        mod = load_repo_module("wt-pm-dbn-feature-extraction")
        X = batch.features[train_mask]
        self._std = _standardiser(X)
        Xs = torch.sigmoid(torch.tensor(self._std(X), dtype=torch.float32))
        dbn = mod.DBN(layers=[X.shape[1], 32, 16])
        # CD-1 greedy layer-wise pretraining (missing from the repo)
        v = Xs
        for rbm in dbn.rbms:
            opt = torch.optim.SGD(rbm.parameters(), lr=0.05)
            for _ in range(20):
                p_h, h = rbm.sample_h(v)
                p_v, v_neg = rbm.sample_v(h)
                p_h_neg, _ = rbm.sample_h(v_neg)
                pos = v.t() @ p_h
                neg = v_neg.t() @ p_h_neg
                opt.zero_grad()
                rbm.W.grad = -(pos - neg) / len(v)
                rbm.v_bias.grad = -(v - v_neg).mean(dim=0)
                rbm.h_bias.grad = -(p_h - p_h_neg).mean(dim=0)
                opt.step()
            with torch.no_grad():
                v, _ = rbm.sample_h(v)
        self._dbn = dbn
        self._fitted = True

    def _predict(self, batch: SensorBatch) -> ModelOutput:
        import torch
        v = torch.sigmoid(torch.tensor(self._std(batch.features), dtype=torch.float32))
        with torch.no_grad():
            for rbm in self._dbn.rbms:
                v = torch.sigmoid(v @ rbm.W + rbm.h_bias)
        return ModelOutput(
            model_id=self.spec.model_id, task=self.spec.task,
            turbine_id=batch.turbine_id, timestamps=batch.timestamps,
            explanation="DBN deep features (CD-1 pretrained RBM stack)",
            extra={"embeddings": v.numpy(), "embeddings_shape": list(v.shape)},
        )


# ---------------------------------------------------------------------------
# m23 — AeroZip compressor
# ---------------------------------------------------------------------------
class AeroZipCompressor(BaseWTModel):
    spec = ModelSpec(
        model_id="m23-aerozip",
        repository="wt-pm-aerozip-autoencoder-compressor",
        task=TaskType.COMPRESSION,
        input_requirements=["features"],
        output_schema=["extracted_features(latent)", "reconstruction_error"],
        resource_requirements=["torch"],
        typical_latency_ms=1200,
        deployment_targets=[Deployment.CLOUD, Deployment.EDGE_CPU, Deployment.EDGE_GPU],
        notes="latent bottleneck for telemetry; actual backhaul byte savings unmeasured",
    )

    def _check_deps(self) -> None:
        import torch  # noqa: F401
        load_repo_module("wt-pm-aerozip-autoencoder-compressor")

    def fit(self, batch: SensorBatch, train_mask: np.ndarray) -> None:
        import torch
        mod = load_repo_module("wt-pm-aerozip-autoencoder-compressor")
        X = batch.features[train_mask]
        self._std = _standardiser(X)
        Xs = torch.tensor(self._std(X), dtype=torch.float32)
        latent = max(X.shape[1] // 8, 4)
        model = mod.AeroZipCompressor(input_dim=X.shape[1], latent_dim=latent)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(60):
            opt.zero_grad()
            lat, rec = model(Xs)
            loss = torch.nn.functional.mse_loss(rec, Xs)
            loss.backward()
            opt.step()
        self._model = model
        self._ratio = mod.compression_ratio(X.shape[1], latent)  # repo's fn
        self._fitted = True

    def _predict(self, batch: SensorBatch) -> ModelOutput:
        import torch
        Xs = torch.tensor(self._std(batch.features), dtype=torch.float32)
        with torch.no_grad():
            lat, rec = self._model(Xs)
            err = ((rec - Xs) ** 2).mean(dim=1).numpy()
        return ModelOutput(
            model_id=self.spec.model_id, task=self.spec.task,
            turbine_id=batch.turbine_id, timestamps=batch.timestamps,
            explanation=f"AeroZip latent code, compression ratio {self._ratio:.1f}:1",
            extra={"latent": lat.numpy(), "recon_error": err,
                   "compression_ratio": self._ratio},
        )
