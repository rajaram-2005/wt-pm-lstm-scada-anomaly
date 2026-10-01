# 1D-CNN Bearing Vibration

> **Category:** Component Fault Classification  
> **Platform adapter:** `m01-1dcnn-bearing`  
> **Status:** integrated (surrogate waveforms)

## Description

1D Convolutional Neural Network processing raw high-frequency vibration signals for gearbox bearing fault identification.

This is model **m01** of the WT-PM wind-turbine predictive-maintenance suite. The repository stays
independently usable (`python model.py`). The unified **wt-pm** platform wraps the original
`model.py` through adapter `m01-1dcnn-bearing` — it does not replace or fork this code.

## Model

`build_1d_cnn(input_shape, num_classes)` in [`model.py`](model.py) returns a compiled Keras
`Sequential` classifier:

| Block | Layers |
|---|---|
| Feature extractor | 3 × (`Conv1D` kernel 3 + ReLU → `MaxPooling1D` 2) with 32 → 64 → 128 filters |
| Classifier head | `Flatten` → `Dense(128)` + ReLU → `Dropout(0.5)` → `Dense(num_classes)` + softmax |
| Compile | Adam · `categorical_crossentropy` · accuracy |

- **Input:** one window of shape `(samples, channels)`, e.g. `(1024, 1)` for a 1024-sample, single-axis vibration signal.
- **Labels:** one-hot encoded (e.g. `tf.keras.utils.to_categorical`), because the loss is categorical cross-entropy.
- **Size:** 2,096,068 parameters for `(1024, 1)` and 4 classes.

## Structure

```
├── model.py           # 1D-CNN architecture (build_1d_cnn); "python model.py" prints its summary
├── requirements.txt   # Dependencies
├── .gitignore         # Environment, data and model-artifact exclusions
├── README.md          # Project overview
├── index.html         # Landing page (GitHub Pages)
└── docs/
    └── index.html     # Copy of the landing page
```

## Setup (standalone)

```bash
pip install -r requirements.txt
python model.py
```

`python model.py` builds the reference configuration (one 1024-sample window, 4 classes) and prints the
Keras model summary. It loads no data and trains nothing. Verified with Python 3.11 and TensorFlow 2.21
(Keras 3); on CPU-only machines TensorFlow prints some startup/CUDA messages to stderr — those come from
TensorFlow, not from this code.

`requirements.txt` lists `tensorflow numpy scipy matplotlib`; `model.py` itself only imports TensorFlow/Keras.

To use the model from your own code:

```python
from model import build_1d_cnn
from tensorflow.keras.utils import to_categorical

model = build_1d_cnn((1024, 1), num_classes=4)
model.fit(X, to_categorical(y, num_classes=4), epochs=10, batch_size=64)  # X: (n, 1024, 1) floats, y: (n,) class ids
```

## Platform contract

| | |
|---|---|
| Input (adapter view) | 1024-sample vibration waveforms — **surrogates** synthesised from `bearing_vib_rms_mm_s` and flagged `vibration_is_surrogate=True` |
| Output (WTDataSchema) | `prediction`, `probability` over `healthy`, `bearing_wear`, `imbalance`, `other` |
| Integration role | Drivetrain fault voter. Hermes uses classifier votes in `diagnose`. Honest surrogate data. |
| Deployment / fallback | cloud, edge-GPU (needs TensorFlow) · falls back to `m11-xgboost-tabular` |

Records follow `WTDataSchema` (`wt-pm.platform.v1`): `timestamp`, `turbine_id` and `model_id` are required;
this adapter also fills `subsystem` (`drivetrain`), `prediction`, `probability`, `inference_time_ms` and
`explanation`. The remaining optional fields (`anomaly_score`, `degradation_state`, `rul_hours`,
`uncertainty`, …) are left to other models.

## Hermes agent + explainable AI

The platform's Hermes loop (Thought → Action → Observation) uses this model through its `diagnose` tool:
the adapter's class probabilities count as one classifier vote, and the model is listed under
`models_responsible` in the XAI evidence bundle. Observations are the adapter's real outputs — never
invented. XAI is m21 SHAP + contrastive healthy-band z + counterfactuals (SHAP is computed on the tree
classifiers, not on this CNN); the Hermes trace is the operator-readable explanation.

To run this model inside the platform (Python ≥ 3.10):

```bash
pip install "wt-pm[tf] @ git+https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly.git#subdirectory=platform"
wt-pm fetch-models --dir ./external      # clones this repo and its siblings into ./external
export WTPM_EXTERNAL_DIR=$PWD/external   # adapters import external/<repo>/model.py
wt-pm inspect                            # m01-1dcnn-bearing should report "available": true
wt-pm serve --port 8100                  # HTTP API, e.g. GET /health and GET /registry
```

Use a separate virtual environment for this, or drop the `[tf]` extra if `tensorflow` is already installed:
`tensorflow` and `tensorflow-cpu` provide the same `tensorflow` package and should not be installed side by side.

More on how the 25 models fit together: [platform architecture](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/ARCHITECTURE.md)
and the [per-model table](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/platform/docs/MODELS.md).

## Honest limits

- Architecture stub: `model.py` defines the network only — no data loader and no training loop.
  No trained weights or field datasets ship here.
- The platform feeds the model **surrogate** waveforms synthesised from 10-minute RMS values, not real
  high-frequency DAQ recordings (real waveforms would slot in unchanged), and trains it briefly
  (4 epochs, batch size 64) on simulator labels.
- `imbalance` is part of the output vocabulary, but the platform's current labelling only produces
  `healthy`, `bearing_wear` and `other`, so that class has no training examples yet.
- Metrics on the platform are held-out **simulator** records (fidelity rung 1, see the
  [fidelity ladder](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/docs/fidelity_ladder.md)),
  not certified asset performance.
