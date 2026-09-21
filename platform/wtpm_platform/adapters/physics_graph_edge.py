"""Adapters for physics/probabilistic, digital-twin, graph, XAI and edge/safety.

m18 pg-bnn-wind-turbine    — physics-guided Bayesian net (uncertainty)
m19 digital-twin-surrogate — FEA surrogate; what-if engine wraps it
m20 gnn-turbines-cascade   — farm-graph cascade risk (PyG or repo fallback)
m21 xai-shap-interpretable — SHAP wrapper (used by XAIEngine)
m24 quantized-mobilenet-edge — INT8/TFLite conversion recipe for edge nets
m25 tinyml-esp32-safety-relay — DecisionTree -> C++ header + in-loop safety gate
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from wtpm_platform.base import BaseWTModel, ModelSpec, load_repo_module
from wtpm_platform.contracts import (
    Deployment, ModelOutput, SensorBatch, Subsystem, TaskType,
)
from wtpm_platform.data import SAFETY_CHANNELS, degradation_target
from wtpm_platform.adapters.anomaly import _standardiser, robust_calibrate


# ---------------------------------------------------------------------------
# m18 — Physics-guided BNN
# ---------------------------------------------------------------------------
class PGBNNWindTurbine(BaseWTModel):
    spec = ModelSpec(
        model_id="m18-pg-bnn",
        repository="wt-pm-pg-bnn-wind-turbine",
        task=TaskType.ANOMALY_DETECTION,
        input_requirements=["torque/rpm/power channels", "features"],
        output_schema=["anomaly_score", "uncertainty"],
        resource_requirements=["torch"],
        typical_latency_ms=2500,
        deployment_targets=[Deployment.CLOUD],
        notes="predicts power from context under P=τω physics loss; "
              "MC-sampled BayesianLinear gives epistemic uncertainty",
    )

    MC = 12

    def _check_deps(self) -> None:
        import torch  # noqa: F401
        load_repo_module("wt-pm-pg-bnn-wind-turbine")

    def _xy(self, batch: SensorBatch):
        names = list(batch.channel_names)
        ctx = [n for n in ("wind_speed_ms", "ambient_temp_c", "rotor_speed_rpm",
                           "pitch_angle_deg", "main_shaft_torque_knm") if n in names]
        X = np.column_stack([batch.channel(c) for c in ctx])
        y = batch.channel("power_kw")
        tau = batch.channel("main_shaft_torque_knm") * 1e3
        omega = batch.channel("rotor_speed_rpm") * 2 * np.pi / 60.0
        return X, y, tau, omega

    def fit(self, batch: SensorBatch, train_mask: np.ndarray) -> None:
        import torch
        mod = load_repo_module("wt-pm-pg-bnn-wind-turbine")
        X, y, tau, omega = self._xy(batch)
        Xtr, ytr = X[train_mask], y[train_mask]
        self._stdx = _standardiser(Xtr)
        self._ymu, self._ysd = ytr.mean(), ytr.std() + 1e-9
        xb = torch.tensor(self._stdx(Xtr), dtype=torch.float32)
        yb = torch.tensor((ytr - self._ymu) / self._ysd, dtype=torch.float32)
        pw = torch.tensor(y[train_mask], dtype=torch.float32)
        tq = torch.tensor(tau[train_mask], dtype=torch.float32)
        om = torch.tensor(omega[train_mask], dtype=torch.float32)
        # small BNN from the repo's BayesianLinear blocks
        l1, l2 = mod.BayesianLinear(X.shape[1], 32), mod.BayesianLinear(32, 1)
        lossf = mod.PhysicsGuidedLoss(lambda_physics=1e-7)  # repo's loss (kW scale)
        opt = torch.optim.Adam(list(l1.parameters()) + list(l2.parameters()), lr=5e-3)
        for _ in range(150):
            opt.zero_grad()
            pred = l2(torch.relu(l1(xb))).squeeze(-1)
            loss = lossf(pred, yb, pw / 1e3, tq / 1e3, om * 60 / (2 * np.pi))
            loss.backward()
            opt.step()
        self._l1, self._l2 = l1, l2
        with torch.no_grad():
            preds = torch.stack([l2(torch.relu(l1(xb))).squeeze(-1) for _ in range(self.MC)])
        err = np.abs(preds.mean(0).numpy() - yb.numpy())
        self._cal = robust_calibrate(err)
        self._fitted = True

    def _predict(self, batch: SensorBatch) -> ModelOutput:
        import torch
        X, y, _, _ = self._xy(batch)
        xb = torch.tensor(self._stdx(X), dtype=torch.float32)
        yn = (y - self._ymu) / self._ysd
        with torch.no_grad():
            preds = torch.stack([self._l2(torch.relu(self._l1(xb))).squeeze(-1)
                                 for _ in range(self.MC)]).numpy()
        mean, std = preds.mean(0), preds.std(0)
        err = np.abs(mean - yn)
        return ModelOutput(
            model_id=self.spec.model_id, task=self.spec.task,
            turbine_id=batch.turbine_id, timestamps=batch.timestamps,
            anomaly_score=self._cal(err),
            uncertainty=std * self._ysd,
            explanation="physics-guided BNN power residual; uncertainty = MC weight sampling",
            extra={"power_pred_kw": mean * self._ysd + self._ymu},
        )


# ---------------------------------------------------------------------------
# m19 — Digital twin surrogate + what-if engine
# ---------------------------------------------------------------------------
class DigitalTwinSurrogate(BaseWTModel):
    spec = ModelSpec(
        model_id="m19-digital-twin",
        repository="wt-pm-digital-twin-surrogate",
        task=TaskType.SURROGATE,
        input_requirements=["operating-condition channels"],
        output_schema=["prediction(stress/deflection/fatigue proxies)"],
        resource_requirements=["torch"],
        typical_latency_ms=1000,
        deployment_targets=[Deployment.CLOUD, Deployment.EDGE_GPU],
        notes="trained against physics-proxy load targets from the simulator "
              "(thrust ~ rho v^2, fatigue ~ vib RMS); enables what-if analysis",
    )

    IN_CH = ("wind_speed_ms", "rotor_speed_rpm", "pitch_angle_deg",
             "yaw_error_deg", "ambient_temp_c", "main_shaft_torque_knm",
             "power_kw", "grid_frequency_hz")

    def _check_deps(self) -> None:
        import torch  # noqa: F401
        load_repo_module("wt-pm-digital-twin-surrogate")

    def _x(self, batch: SensorBatch) -> np.ndarray:
        return np.column_stack([batch.channel(c) for c in self.IN_CH if c in batch.channel_names])

    @staticmethod
    def _targets(batch: SensorBatch) -> np.ndarray:
        """Physics-proxy load targets (labelled as proxies, not FEA truth)."""
        v = batch.channel("wind_speed_ms")
        rpm = batch.channel("rotor_speed_rpm")
        vib = batch.channel("bearing_vib_rms_mm_s")
        stress = 0.6 * v ** 2 + 0.4 * rpm ** 2 / 10          # von-Mises proxy
        deflect = 0.02 * v ** 2                               # tip deflection proxy
        fatigue = np.cumsum(np.maximum(vib - np.median(vib), 0)) / max(len(v), 1)
        return np.column_stack([stress, deflect, fatigue])

    def fit(self, batch: SensorBatch, train_mask: np.ndarray) -> None:
        import torch
        mod = load_repo_module("wt-pm-digital-twin-surrogate")
        X, Y = self._x(batch)[train_mask], self._targets(batch)[train_mask]
        self._stdx, = (_standardiser(X),)
        self._ymu, self._ysd = Y.mean(0), Y.std(0) + 1e-9
        xb = torch.tensor(self._stdx(X), dtype=torch.float32)
        yb = torch.tensor((Y - self._ymu) / self._ysd, dtype=torch.float32)
        model = mod.FEASurrogate(input_dim=X.shape[1], hidden=64, output_dim=3)
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        for _ in range(120):
            opt.zero_grad()
            loss = mod.residual_loss(model(xb), yb)   # repo's loss fn
            loss.backward()
            opt.step()
        self._model = model
        self._fitted = True

    def _predict(self, batch: SensorBatch) -> ModelOutput:
        import torch
        xb = torch.tensor(self._stdx(self._x(batch)), dtype=torch.float32)
        with torch.no_grad():
            out = self._model(xb).numpy() * self._ysd + self._ymu
        return ModelOutput(
            model_id=self.spec.model_id, task=self.spec.task,
            turbine_id=batch.turbine_id, timestamps=batch.timestamps,
            prediction=out[:, 0],  # stress proxy as headline
            explanation="digital-twin surrogate load proxies [stress, deflection, fatigue]",
            extra={"loads": out, "load_names": ["von_mises_proxy", "deflection_proxy", "fatigue_proxy"]},
        )

    # ---- what-if API used by DigitalTwinInterface --------------------------
    def what_if(self, batch: SensorBatch, overrides: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Re-run the surrogate under modified operating conditions."""
        import torch
        X = self._x(batch).copy()
        chs = [c for c in self.IN_CH if c in batch.channel_names]
        for ch, val in overrides.items():
            if ch in chs:
                X[:, chs.index(ch)] = val
        with torch.no_grad():
            out = self._model(torch.tensor(self._stdx(X), dtype=torch.float32)).numpy()
        return {"loads": out * self._ysd + self._ymu, "overrides": overrides}


# ---------------------------------------------------------------------------
# m20 — GNN turbine cascade
# ---------------------------------------------------------------------------
class GNNTurbineCascade(BaseWTModel):
    spec = ModelSpec(
        model_id="m20-gnn-cascade",
        repository="wt-pm-gnn-turbines-cascade",
        task=TaskType.GRAPH_ANALYSIS,
        input_requirements=["fleet batches + wake graph (meta['fleet'])"],
        output_schema=["prediction(cascade risk class per turbine)"],
        resource_requirements=["torch", "torch-geometric (optional: repo has fallback)"],
        typical_latency_ms=2000,
        deployment_targets=[Deployment.CLOUD],
        notes="node = turbine, edge = wake coupling from farm layout; "
              "repo ships its own linear fallback when PyG is absent",
    )

    RISK = ["low", "elevated", "high", "critical"]

    def _check_deps(self) -> None:
        import torch  # noqa: F401
        load_repo_module("wt-pm-gnn-turbines-cascade")

    @staticmethod
    def _node_features(batches: List[SensorBatch], scores: Dict[str, np.ndarray]) -> np.ndarray:
        feats = []
        for b in batches:
            s = scores.get(b.turbine_id, np.zeros(b.n_steps))
            tail = slice(-144, None)  # last day
            feats.append([
                float(np.mean(s[tail])), float(np.max(s[tail])),
                float(np.mean(b.channel("power_kw")[tail])),
                float(np.mean(b.channel("wind_speed_ms")[tail])),
                float(np.mean(b.channel("bearing_vib_rms_mm_s")[tail])),
                float(np.mean(b.channel("gearbox_oil_temp_c")[tail])),
            ])
        return np.asarray(feats, dtype=np.float32)

    def fit(self, batch: SensorBatch, train_mask: np.ndarray) -> None:
        # Trained lazily in predict_fleet (needs the whole farm, not one turbine).
        self._fitted = True

    def _predict(self, batch: SensorBatch) -> ModelOutput:
        raise RuntimeError("m20 is fleet-level; call predict_fleet(batches, scores, edges)")

    def predict_fleet(
        self,
        batches: List[SensorBatch],
        anomaly_scores: Dict[str, np.ndarray],
        edge_index: np.ndarray,           # (2, E) wake-coupling edges
    ) -> ModelOutput:
        import torch
        mod = load_repo_module("wt-pm-gnn-turbines-cascade")
        X = self._node_features(batches, anomaly_scores)
        Xn = (X - X.mean(0)) / (X.std(0) + 1e-9)
        # weak labels for training: risk band from own + upstream anomaly level
        own = X[:, 1]
        risk = np.digitize(own, [1.5, 3.0, 6.0])
        net = mod.TurbineCascadeGNN(in_channels=X.shape[1], hidden=32, out_classes=4)
        xb = torch.tensor(Xn)
        ei = torch.tensor(edge_index, dtype=torch.long)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        yb = torch.tensor(risk)
        for _ in range(150):
            opt.zero_grad()
            out = net(xb, ei)
            loss = torch.nn.functional.cross_entropy(out, yb)
            loss.backward()
            opt.step()
        with torch.no_grad():
            proba = torch.softmax(net(xb, ei), dim=1).numpy()
        pred = np.array([self.RISK[i] for i in proba.argmax(1)], dtype=object)
        ts = np.array([b.timestamps[-1] for b in batches])
        return ModelOutput(
            model_id=self.spec.model_id, task=self.spec.task,
            turbine_id="FLEET", timestamps=ts, subsystem=Subsystem.FARM,
            prediction=pred,
            probability={c: proba[:, i] for i, c in enumerate(self.RISK)},
            explanation="GNN cascade-risk class per turbine over the wake graph",
            extra={"turbine_ids": [b.turbine_id for b in batches], "edge_index": edge_index},
        )


# ---------------------------------------------------------------------------
# m21 — SHAP XAI wrapper
# ---------------------------------------------------------------------------
class XAIShapInterpretable(BaseWTModel):
    spec = ModelSpec(
        model_id="m21-xai-shap",
        repository="wt-pm-xai-shap-interpretable",
        task=TaskType.EXPLAINABILITY,
        input_requirements=["a fitted tree model + feature matrix"],
        output_schema=["explanation", "extra['shap_values']"],
        resource_requirements=["shap"],
        typical_latency_ms=5000,
        deployment_targets=[Deployment.CLOUD],
        notes="wraps the repo's explain_with_shap dispatch (TreeExplainer vs generic)",
    )

    def _check_deps(self) -> None:
        import shap  # noqa: F401

    def fit(self, batch: SensorBatch, train_mask: np.ndarray) -> None:
        self._fitted = True  # stateless wrapper

    def _predict(self, batch: SensorBatch) -> ModelOutput:
        raise RuntimeError("m21 is invoked through XAIEngine.explain(), not predict()")

    def explain(self, estimator, X: np.ndarray, feature_names: List[str],
                row: int, class_index: Optional[int] = None) -> Dict[str, float]:
        import shap
        # Same explainer dispatch rule as the repo's explain_with_shap()
        explainer = shap.TreeExplainer(estimator) if hasattr(estimator, "get_booster") \
            else shap.TreeExplainer(estimator)   # RF is also a tree model
        sv = explainer.shap_values(X[row:row + 1])
        if isinstance(sv, list):  # multiclass: list of (1, F)
            ci = class_index if class_index is not None else 0
            vals = np.asarray(sv[ci])[0]
        else:
            vals = np.asarray(sv)[0]
            if vals.ndim == 2:   # (F, n_classes)
                ci = class_index if class_index is not None else int(np.argmax(np.abs(vals).sum(0)))
                vals = vals[:, ci]
        order = np.argsort(-np.abs(vals))
        return {feature_names[i]: float(vals[i]) for i in order[:8]}


# ---------------------------------------------------------------------------
# m24 — Quantized edge conversion
# ---------------------------------------------------------------------------
class QuantizedMobileNetEdge(BaseWTModel):
    spec = ModelSpec(
        model_id="m24-quantized-edge",
        repository="wt-pm-quantized-mobilenet-edge",
        task=TaskType.EDGE_INFERENCE,
        input_requirements=["a Keras model to quantize"],
        output_schema=["extra['tflite_path', 'size_bytes']"],
        resource_requirements=["tensorflow"],
        typical_latency_ms=20000,
        deployment_targets=[Deployment.EDGE_CPU, Deployment.EDGE_GPU],
        notes="repo = INT8 TFLite conversion recipe; applied to the platform's "
              "small edge net (full 224x224 MobileNet has no data source here)",
    )

    def _check_deps(self) -> None:
        import tensorflow  # noqa: F401

    def fit(self, batch: SensorBatch, train_mask: np.ndarray) -> None:
        self._fitted = True

    def _predict(self, batch: SensorBatch) -> ModelOutput:
        raise RuntimeError("m24 is invoked through EdgeInferenceManager.quantize()")

    def quantize(self, keras_model, sample_input: np.ndarray, out_path: str) -> Dict[str, object]:
        """INT8 conversion using the repo's converter settings."""
        import tensorflow as tf
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]   # repo's setting
        def rep():
            for i in range(min(100, len(sample_input))):
                yield [sample_input[i:i + 1].astype(np.float32)]
        converter.representative_dataset = rep
        tflite = converter.convert()
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(tflite)
        return {"tflite_path": out_path, "size_bytes": len(tflite)}


# ---------------------------------------------------------------------------
# m25 — TinyML ESP32 safety relay
# ---------------------------------------------------------------------------
class TinyMLSafetyRelay(BaseWTModel):
    spec = ModelSpec(
        model_id="m25-tinyml-safety",
        repository="wt-pm-tinyml-esp32-safety-relay",
        task=TaskType.SAFETY,
        input_requirements=[f"safety channels {SAFETY_CHANNELS}"],
        output_schema=["prediction(trip/continue)", "extra['model_h']"],
        resource_requirements=["scikit-learn", "micromlgen (for C++ export)"],
        typical_latency_ms=50,
        deployment_targets=[Deployment.MCU, Deployment.EDGE_CPU],
        notes="depth-5 DecisionTree as in the repo; exported to model.h AND "
              "mirrored in-loop as the SafetyManager's gate",
    )

    def _check_deps(self) -> None:
        from sklearn.tree import DecisionTreeClassifier  # noqa: F401

    def _x(self, batch: SensorBatch) -> np.ndarray:
        return np.column_stack([batch.channel(c) for c in SAFETY_CHANNELS
                                if c in batch.channel_names])

    def fit(self, batch: SensorBatch, train_mask: np.ndarray) -> None:
        from sklearn.tree import DecisionTreeClassifier
        X = self._x(batch)
        # trip label: extreme vibration or oil temp against the *healthy* band
        hi = degradation_target(batch, healthy_mask=train_mask)
        y = (hi > 1.0).astype(int)
        if y.sum() == 0:  # ensure both classes exist for the tree
            y[np.argmax(hi)] = 1
        self._clf = DecisionTreeClassifier(max_depth=5)  # repo's depth
        self._clf.fit(X, y)
        self._fitted = True

    def _predict(self, batch: SensorBatch) -> ModelOutput:
        X = self._x(batch)
        trip = self._clf.predict(X)
        pred = np.where(trip == 1, "TRIP", "continue").astype(object)
        return ModelOutput(
            model_id=self.spec.model_id, task=self.spec.task,
            turbine_id=batch.turbine_id, timestamps=batch.timestamps,
            prediction=pred,
            explanation=f"depth-5 safety tree on {SAFETY_CHANNELS}",
            extra={"trip": trip},
        )

    def export_esp32(self, out_path: str) -> str:
        """C++ header via micromlgen — exactly the repo's export path."""
        from micromlgen import port
        code = port(self._clf)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            f.write(code)
        return out_path
