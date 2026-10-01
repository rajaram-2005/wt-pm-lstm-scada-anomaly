# DBN-FEATURE-EXTRACTION
> **Category:** RUL Prognostics  
> **Platform adapter:** `m09-dbn-features`  
> **Status:** integrated (platform supplies its own CD-1 loop; Model 5 ships one too)

## Description
Deep Belief Network using stacked Restricted Boltzmann Machines for automated feature hierarchy in wear tracking.

This is model **m09** of the WT-PM wind-turbine predictive-maintenance suite (fifth repository alphabetically in the collection). The repository stays independently usable (`python model.py`). The unified **wt-pm** platform wraps the original `model.py` through adapter `m09-dbn-features` — it does not replace or fork this code.

**Model 5** extends the original scaffold (3-layer DBN, randomly initialised RBM stack with no training loop) to a **5-layer stack** with greedy CD-1 pretraining and supervised fine-tuning:

```
Input (64 sensor channels in (0, 1))
  └─ RBM(64→48)  + CD-1 pretraining    # detector bank
  └─ RBM(48→32)  + CD-1 pretraining    # mid-level wear patterns
  └─ RBM(32→16)  + CD-1 pretraining    # degradation hierarchy
  └─ RBM(16→8)   + CD-1 pretraining    # compact codes
  └─ Linear(8→1)                       # normalized RUL regression head
```

- **Pretraining:** greedy layer-wise CD-1 (free-energy contrastive divergence), 15 epochs per RBM (SGD, lr 0.1). Visible units see thresholded sensor bits and sampled hidden states are chained upward — the classic DBN protocol; reconstruction MSE drops from the ~0.25 no-skill level to 0.09 / 0.07 / 0.06 / 0.07 across the stack.
- **Fine-tuning:** Adam with differential rates (stack 1e-3, head 1e-2, weight decay 1e-4), MSE loss, early stopping (patience 20) on the validation window.
- **Data:** synthetic fleet campaign — 48 turbines with turbine-specific wear exponents and service lives, inspected every 6 campaign days (2,140 inspections, 64 channels). Sensors report the nonlinear wear state/rate pair plus load and thermal oscillations. **Chronological** campaign-day split (65% / 15% / 20%): training never sees an inspection recorded after its cutoff.

## Structure
```
├── model.py           # Model 5 architecture + fleet synthesis + training script
├── requirements.txt   # Dependencies
├── results/           # Run outputs (metrics JSON, best .pt weights) — created at runtime
├── .gitignore         # Environment and data exclusions
└── README.md          # Project overview
```

## Setup & Execution
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run Model 5 (fleet synthesis → CD-1 pretraining → fine-tuning → evaluation):
   ```bash
   python model.py
   ```

Outputs:
- Printed evaluation table (RMSE / MAE / R² on train/val/test, in normalized RUL and campaign days)
- `results/model5_metrics.json` — full metrics summary
- `results/model5_weights.pt` — best weights (excluded from git via `.gitignore`)

## Model

[`model.py`](model.py) defines `RBM(visible, hidden)` and `DBN(layers=[64, 48, 32, 16, 8])`:

| Block | Definition |
|---|---|
| `RBM` | `W (visible, hidden)`, `v_bias`, `h_bias`; `sample_h(v)` / `sample_v(h)` return `(prob, sample)`; `free_energy(v)` drives the CD-1 update |
| `DBN.rbms` | `ModuleList` of 4 stacked RBMs (5,521 parameters total with the head) |
| `DBN.forward(x)` | Mean-field hidden probabilities upward, then `classifier` → normalized RUL |
| `DBN.pretrain(x)` | Greedy layer-wise CD-1 (additive helper; returns per-RBM reconstruction MSE) |
| `DBN.encode(x)` | Deep feature representation without the classifier |

- **Input:** feature tensor `x` of shape `(batch_size, 64)` (float32 in (0, 1) — the visible-probability range the RBMs expect).
- **Output:** `forward` returns `(batch_size, 1)` normalized RUL; `encode` returns `(batch_size, 8)` deep features.
- **Size:** 5,521 parameters for the default 5-layer configuration. The platform adapter instantiates `DBN(layers=[38, 32, 16])` on its 38-column feature table.

`python model.py` runs the full pipeline (deterministic seeds). Verified with Python 3.11 and PyTorch 2.14.

## Platform contract

| | |
|---|---|
| Input (adapter view) | tabular features (`batch.features`) |
| Output (`ModelOutput` / `WTDataSchema`) | `extra={"embeddings": v, "embeddings_shape": ...}` — deep features from the CD-1-pretrained stack |
| Integration role | Feature provider (`TaskType.FEATURE_EXTRACTION`, routed in `support`). Extracted embeddings feed downstream tabular models. Not a decision maker. |
| Deployment / fallback | `cloud` (requires `torch`) · no fallback model configured |

The platform adapter `m09-dbn-features` calls the original API surface (`DBN(layers=[...])`, `rbm.W` / `rbm.v_bias` / `rbm.h_bias`, `sample_h` / `sample_v`) and runs its own CD-1 loop over them — Model 5 keeps all of these stable (verified against the adapter's call sequence). Records emitted via `ModelOutput.to_records()` follow `WTDataSchema` (`wt-pm.platform.v1`); embeddings surface under `result["support"]["m09-dbn-features"]` in `Orchestrator.analyse()`. The Hermes agent loop uses diagnostic, anomaly and RUL voters as tools; `m09-dbn-features` serves as a feature provider. Observations in Hermes are always real adapter outputs — never invented.

To run this model inside the platform (Python ≥ 3.10):

```bash
pip install "wt-pm[torch] @ git+https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly.git#subdirectory=platform"
wt-pm fetch-models --dir ./external      # clones this repo and its siblings into ./external
export WTPM_EXTERNAL_DIR=$PWD/external   # adapters import external/<repo>/model.py
wt-pm inspect                            # m09-dbn-features should report "available": true
```

More on how the 25 models fit together: [platform architecture](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/ARCHITECTURE.md) and the [per-model table](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/platform/docs/MODELS.md).

## Model Info
- **Repo name:** `wt-pm-dbn-feature-extraction`
- **Category:** RUL Prognostics
- **Model:** 5-layer DBN (~5.5K parameters)
- **Dependencies:** `torch numpy`

## Honest limits

- Synthetic fleet only: the campaign is generated from turbine-specific wear curves (wear exponent and life are drawn per turbine); no field datasets or trained weights ship in this repository.
- Reported metrics come from held-out **campaign days** of the synthetic fleet (test R² 0.874, RMSE 0.038 normalized ≈ 10.2 days at the fleet-mean 270-day life). Platform metrics come from **simulator** records (fidelity rung 1, see the [fidelity ladder](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/docs/fidelity_ladder.md)), not certified asset performance.
- Day-scale errors use the fleet-mean life (270 days) as the conversion of normalized RUL; per-turbine day errors vary with individual service life.
- Bernoulli RBMs are pretrained on thresholded sensor bits (they need contrasty inputs to learn detectors) and fine-tuned on mean-field probabilities; the platform adapter feeds its own sigmoid-standardized features to the same API.
