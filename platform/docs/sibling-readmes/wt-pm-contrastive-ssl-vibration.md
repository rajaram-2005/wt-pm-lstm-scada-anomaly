# Contrastive SSL Vibration

> **Category:** Component Fault Classification  
> **Platform adapter:** `m08-contrastive-ssl`  
> **Status:** partial (SimCLR encoder; platform adapter supplies full NT-Xent loss)

## Description

Self-supervised contrastive learning framework for feature extraction from unannotated wind turbine vibration streams.

This is model **m08** of the WT-PM wind-turbine predictive-maintenance suite (third repository alphabetically in the collection). The repository stays independently usable (`python model.py`). The unified **wt-pm** platform wraps the original `model.py` through adapter `m08-contrastive-ssl` — it does not replace or fork this code.

## Model

[`model.py`](model.py) defines `ContrastiveEncoder(input_dim=1024, proj_dim=64)` (a 1D convolutional feature encoder and projection head) and `nt_xent_loss(z1, z2, temperature=0.5)`:

| Block | Layers |
|---|---|
| `encoder` (conv feature extractor) | `Conv1d(1, 32, kernel_size=7, stride=2)` → `ReLU` → `Conv1d(32, 64, kernel_size=3)` → `ReLU` |
| `encoder` (pooling & dense) | `AdaptiveAvgPool1d(1)` → `Flatten` → `Linear(64, 128)` → `ReLU` |
| `proj` (projection head) | `Linear(128, proj_dim)` |
| `forward(x)` | Returns `F.normalize(z, dim=1)` with shape `(batch_size, proj_dim)` |
| `nt_xent_loss(z1, z2)` | Positive-pair alignment loss (simplified placeholder) |

- **Input:** 1D vibration signal tensor `x` of shape `(batch_size, 1, samples)` (e.g. `(batch_size, 1, 1024)`), float32.
- **Output:** Unit L2-normalized latent embedding `z` of shape `(batch_size, proj_dim)`.
- **Loss:** Temperature-scaled NT-Xent (SimCLR). Upstream `model.py` provides a simplified positive-pair alignment loss; the platform adapter implements the full normalized temperature-scaled cross entropy with negative batch pairs.
- **Size:** 23,040 parameters for the default `(input_dim=1024, proj_dim=64)` configuration (18,912 parameters when instantiated with `proj_dim=32` by the platform adapter).

## Structure

```
├── model.py           # Contrastive encoder architecture (ContrastiveEncoder, nt_xent_loss); "python model.py" prints its summary
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

`python model.py` builds the reference configuration (`input_dim=1024, proj_dim=64`), prints the PyTorch module architecture and parameter count (`23,040`), and runs a smoke-test forward pass and loss evaluation on synthetic waveforms. It loads no external data and trains nothing. Verified with Python 3.11 and PyTorch 2.14.

`requirements.txt` lists `torch torchvision numpy`; `model.py` itself imports only `torch`, `torch.nn`, and `torch.nn.functional`.

To use the model from your own code:

```python
import torch
from model import ContrastiveEncoder, nt_xent_loss

model = ContrastiveEncoder(input_dim=1024, proj_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# x1, x2: two augmented views of vibration waveforms (batch_size, 1, 1024)
x1 = torch.randn(16, 1, 1024)
x2 = x1 + 0.05 * torch.randn_like(x1)

z1 = model(x1)  # (16, 64), L2-normalized
z2 = model(x2)  # (16, 64), L2-normalized

loss = nt_xent_loss(z1, z2, temperature=0.5)
loss.backward()
optimizer.step()
```

## Platform contract

| | |
|---|---|
| Input (adapter view) | Vibration waveforms `batch.vib_waveforms` — **surrogates** synthesised from 10-minute SCADA RMS `bearing_vib_rms_mm_s` (flagged `vibration_is_surrogate=True`) |
| Output (`ModelOutput` / `WTDataSchema`) | `explanation` (`"contrastive embeddings of vibration surrogates"`), plus `extra={"embeddings": z, "embedding_index": batch.vib_index}` |
| Integration role | Unsupervised feature provider (`TaskType.FEATURE_EXTRACTION`, routed in `support`). Extracted embeddings feed downstream tabular classifiers. Not a decision maker. |
| Deployment / fallback | `cloud`, `edge_gpu` (requires `torch`) · no fallback model configured |

Records emitted via `ModelOutput.to_records()` follow `WTDataSchema` (`wt-pm.platform.v1`): `timestamp`, `turbine_id` and `model_id` (`m08-contrastive-ssl`) are required; this adapter also fills `subsystem` (`drivetrain`), `inference_time_ms` and `explanation`, while `embeddings` and `embedding_index` live on `ModelOutput.extra` (and are surfaced under `result["support"]["m08-contrastive-ssl"]` in `Orchestrator.analyse()`). Decision fields (`prediction`, `probability`, `anomaly_score`, `degradation_state`, `rul_hours`, `uncertainty`) are left to the anomaly, classification, and prognostics models.

## Hermes agent + explainable AI

In the platform's `Orchestrator.analyse()` pipeline, `ModelRouter` places `m08-contrastive-ssl` in the `support` bucket (`routed["support"]`) so it executes alongside representation, edge, and twin models, reporting status and extracted representations in `model_health` and `support["m08-contrastive-ssl"]`. The Hermes agent loop (Thought → Action → Observation) uses diagnostic, anomaly, RUL, twin, and safety outputs to answer the operator's operational questions; `m08-contrastive-ssl` serves as a feature provider rather than a diagnostic voter. Observations in Hermes are always real adapter outputs — never invented. XAI is `m21` SHAP (`TreeExplainer` on the tree classifiers) + contrastive healthy-band z + counterfactuals; the Hermes trace is the operator-readable explanation.

To run this model inside the platform (Python ≥ 3.10):

```bash
pip install "wt-pm[torch] @ git+https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly.git#subdirectory=platform"
wt-pm fetch-models --dir ./external      # clones this repo and its siblings into ./external
export WTPM_EXTERNAL_DIR=$PWD/external   # adapters import external/<repo>/model.py
wt-pm inspect                            # m08-contrastive-ssl should report "available": true
wt-pm agent --days 8                     # fits m08-contrastive-ssl + runs the Hermes trace
wt-pm serve --port 8100                  # HTTP API, e.g. GET /health and GET /registry
```

If `torch` is already installed in your environment, you can drop the `[torch]` extra when installing `wt-pm`.

More on how the 25 models fit together: [platform architecture](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/ARCHITECTURE.md) and the [per-model table](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/platform/docs/MODELS.md).

## Honest limits

- Architecture stub: `model.py` defines `ContrastiveEncoder` and `nt_xent_loss` only — no data loader, dataset readers, or data augmentation pipelines ship here. No trained weights or field datasets ship in this repository.
- Partial upstream loss: `model.py` defines a simplified positive-pair placeholder loss; negative pairs and full cross-entropy contrastive learning are implemented in the platform adapter `m08-contrastive-ssl`.
- Waveform fidelity: on the platform, input vibration streams are **surrogate** waveforms synthesised from 10-minute SCADA RMS vibration values (`bearing_vib_rms_mm_s`, flagged with `vibration_is_surrogate=True`), not raw megahertz DAQ sensor telemetry. Real high-frequency vibration streams slot in identically when available.
- Metrics on the platform come from held-out **simulator** records (fidelity rung 1, see the [fidelity ladder](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/docs/fidelity_ladder.md)), not certified asset performance.
