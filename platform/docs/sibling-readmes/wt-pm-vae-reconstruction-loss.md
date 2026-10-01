# VAE Reconstruction

> **Category:** SCADA Anomaly Detection  
> **Platform adapter:** `m22-vae-reconstruction`  
> **Status:** integrated (uses repo's vae_loss)

## Description

Variational Autoencoder using reconstruction loss to isolate rare non-linear sensor anomalies in SCADA logs.

This is model **m22** of the WT-PM wind-turbine predictive-maintenance suite. The repository stays
independently usable (`python model.py`). The unified **wt-pm** platform wraps the original
`model.py` through adapter `m22-vae-reconstruction` — it does not replace or fork this code.

## Model

[`model.py`](model.py) is a complete VAE:

| Symbol | What it is |
|---|---|
| `VAEAnomalyDetector(input_dim=64, hidden_dim=32, latent_dim=8)` | `fc1` encoder → `fc_mu`/`fc_logvar` → reparameterised sample → `fc2`/`fc3` decoder |
| `encode / reparameterize / decode / forward` | standard VAE plumbing returning `(recon, mu, logvar)` |
| `vae_loss(recon_x, x, mu, logvar)` | summed MSE reconstruction + KL divergence |

`python model.py` defines the architecture; it loads no data and trains nothing.

## Structure

```
├── model.py           # Architecture / training entry point (unchanged by the platform)
├── requirements.txt   # Dependencies
├── .gitignore         # Environment, data and model-artifact exclusions
├── README.md          # Project overview (this file)
├── index.html         # Landing page (GitHub Pages)
└── docs/
    └── index.html     # Copy of the landing page
```

## Setup (standalone)

```bash
pip install -r requirements.txt
python model.py
```

`python model.py` defines `VAEAnomalyDetector` and `vae_loss`; it loads no data and trains nothing. `requirements.txt` lists `torch numpy pandas`; `model.py` itself only imports PyTorch.

To use the model from your own code, import from `model.py` directly — the entry points above
are the exact symbols the platform adapter calls.

## Platform contract

| | |
|---|---|
| Input (adapter view) | tabular feature windows |
| Output (WTDataSchema) | anomaly_score — reconstruction error; uncertainty |
| Integration role | Reconstruction-error anomaly stream (WHAT). Hermes `anomaly` tool. |
| Deployment / fallback | cloud, edge-GPU · falls back to `m14-isolation-forest` |

Records follow `WTDataSchema` (`wt-pm.platform.v1`): `timestamp`, `turbine_id` and `model_id` are required;
this adapter also fills `anomaly_score`, `uncertainty`, `inference_time_ms` and `explanation`. The remaining optional fields (`prediction`, `probability`, `degradation_state`, `rul_hours`) are left to other models.

## Hermes agent + explainable AI

The VAE's reconstruction error is one anomaly vote in Hermes `anomaly`; observations are the adapter's real outputs — never invented. XAI is m21 SHAP + contrastive healthy-band z + counterfactuals; the Hermes trace is the operator-readable explanation.

To run this model inside the platform (Python ≥ 3.10):

```bash
pip install "git+https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly.git#subdirectory=platform"
wt-pm fetch-models --dir ./external      # clones this repo and its siblings into ./external
export WTPM_EXTERNAL_DIR=$PWD/external   # adapters import external/<repo>/model.py
wt-pm inspect                            # m22-vae-reconstruction should report "available": true
wt-pm serve --port 8100                  # HTTP API, e.g. GET /health and GET /registry
```

More on how the 25 models fit together: [platform architecture](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/ARCHITECTURE.md)
and the [per-model table](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/platform/docs/MODELS.md).

## Honest limits

- Trained with the repo's own `vae_loss` on simulator feature windows; no field SCADA logs ship here.
- Metrics on the platform are held-out **simulator** records (fidelity rung 1), not certified asset performance.
