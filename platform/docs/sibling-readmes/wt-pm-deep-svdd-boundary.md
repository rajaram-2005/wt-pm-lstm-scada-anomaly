# DEEP-SVDD-BOUNDARY
> **Category:** SCADA Anomaly Detection  
> **Platform adapter:** `m13-deep-svdd`  
> **Status:** integrated (platform supplies its own training loop; Model 6 ships one too)

## Description
Deep Support Vector Data Description mapping telemetry into a hypersphere boundary for abnormal state isolation.

This is model **m13** of the WT-PM wind-turbine predictive-maintenance suite (sixth repository alphabetically in the collection). The repository stays independently usable (`python model.py`). The unified **wt-pm** platform wraps the original `model.py` through adapter `m13-deep-svdd` — it does not replace or fork this code.

**Model 6** extends the original scaffold (3-layer MLP with biases, no data, no training loop) to a **6-layer bias-free encoder** trained with the one-class Deep SVDD objective (Ruff et al., ICML 2018), plus a full data → training → evaluation pipeline:

```
Input (64 standardized + ZCA-whitened SCADA features)
  └─ Linear(64 → 128)  → LeakyReLU(0.1)     # no bias in any layer
  └─ Linear(128 → 96)  → LeakyReLU(0.1)
  └─ Linear(96 → 64)   → LeakyReLU(0.1)
  └─ Linear(64 → 48)   → LeakyReLU(0.1)
  └─ Linear(48 → 32)   → LeakyReLU(0.1)
  └─ Linear(32 → rep_dim)                    # hypersphere representation z
      score = ‖z − c‖²                       # c = fixed centre from init_center
```

- **Why bias-free:** with bias terms a network can output the constant `c` for every input (zero loss, no information) — the "hypersphere collapse" trivial solution. Removing all biases and keeping `c` fixed rules it out, as in the Deep SVDD paper. Kaiming init keeps representation scale through six layers.
- **Preprocessing:** standardize, then ZCA-whiten with statistics from the healthy training window. The 64 features are mean/std/min/max views of 16 channels and are strongly correlated; whitening stops a few physical signals from dominating the distance. (Without it, every SVDD variant tested plateaued around AUROC 0.80.)
- **Training:** `init_center` on the training window, then `svdd_loss` with Adam (lr 1e-4, weight decay 1e-6, batch 256) for a **fixed 20 epochs**. Deep SVDD keeps lowering healthy-data loss long after fault contrast stops improving, and a healthy-only validation set gives no label-free signal to stop on — so the budget is committed up front. The AUROC-vs-epoch curve is still recorded as a diagnostic (`auroc_by_epoch_diagnostic`), never used for selection.
- **Alarm:** threshold = 99th percentile of healthy validation distances; an alarm fires after 3 consecutive hourly exceedances.
- **Data:** synthetic 24-turbine farm, 120 days, hourly. 16 physical channels sampled every 10 min (wind, power curve, rotor speed, pitch, yaw error, ambient/nacelle/gearbox-oil/gearbox-bearing/generator-bearing temperatures, three winding temperatures — all with first-order thermal lag — drivetrain and tower vibration, phase-current imbalance) summarized per hour as mean/std/min/max → **64 features**. Six incipient fault types are injected into 12 turbines (2 each), **test window only**, each ramping in over 4–10 days: gearbox bearing overheating, generator winding hot-spot, pitch misalignment, yaw misalignment, drivetrain bearing wear, rotor imbalance.
- **Split (chronological):** days 0–70 train (healthy only — one-class), 70–85 validation (healthy; threshold), 85–120 test (12 healthy + 12 faulty turbines; 20,160 records, 6,607 faulty).

## Structure
```
├── model.py           # Model 6 architecture + farm synthesis + training/evaluation script
├── requirements.txt   # Dependencies
├── results/           # Run outputs (metrics JSON, seed sweep, .pt weights)
├── index.html         # Project page
├── docs/              # Project page (GitHub Pages copy)
├── .gitignore         # Environment and data exclusions
└── README.md          # Project overview
```

## Setup & Execution
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run Model 6 (farm synthesis → whitening → Deep SVDD → evaluation, ~25 s on CPU):
   ```bash
   python model.py              # seed 42
   python model.py --seed 3     # any other seed: regenerates the farm and the init
   ```

Outputs:
- Printed comparison table (Model 6 vs the original scaffold vs a Mahalanobis reference) and per-fault detection delays
- `results/model6_metrics.json` — full metrics, including every fault event
- `results/model6_weights.pt` — weights, centre, standardization + whitening statistics and threshold (excluded from git via `.gitignore`)

## Results

All three detectors use the identical pipeline (same data, whitening, threshold rule and alarm persistence); only the scoring model differs. "Scaffold" is the original 3-layer network with biases trained with the same 20-epoch schedule. Mahalanobis distance is the linear reference (squared norm in the whitened space).

**Seed 42** (`results/model6_metrics.json`, deterministic):

| detector | AUROC | precision | recall | F1 | events detected | median delay | false alarms /100 healthy turbine-days |
|---|---|---|---|---|---|---|---|
| **Model 6** (6-layer, bias-free) | 0.865 | 1.000 | 0.560 | **0.718** | **12/12** | **65.5 h** | **0.0** |
| original scaffold (3-layer, biases) | **0.871** | 1.000 | 0.572 | 0.727 | 10/12 | 65.5 h | 0.0 |
| Mahalanobis (linear) | 0.870 | 0.999 | 0.481 | 0.649 | 11/12 | 116 h | 0.71 |

**Six seeds** (42, 1–5; `results/model6_seed_sweep.json`) — each seed regenerates the fault turbines, onsets, ramps, noise and network init:

| detector | AUROC (mean ± sd) | F1 (mean ± sd) | events detected | median delay (all detected events) | false alarms /100 turbine-days |
|---|---|---|---|---|---|
| **Model 6** | **0.885 ± 0.016** | **0.782 ± 0.038** | **68/72** | **56.5 h** | 0.04 |
| original scaffold | 0.864 ± 0.013 | 0.702 ± 0.047 | 59/72 | 70.0 h | 0.04 |
| Mahalanobis (linear) | 0.877 ± 0.009 | 0.661 ± 0.059 | 63/72 | 95.0 h | 1.03 |

Seed 42 happens to be Model 6's weakest AUROC of the six; across seeds it leads on AUROC, F1, events detected and delay (ties the scaffold on false alarms), catching 9 more fault events than the scaffold and about half a day sooner. Per-fault delays at seed 42 (hours after onset, severity at detection): generator winding 25 h / 33 h (sev 0.12 / 0.17), rotor imbalance 32 h / 43 h, drivetrain bearing 32 h / 66 h, gearbox bearing 65 h / 109 h, pitch misalignment ~155 h, yaw misalignment 114 h / 407 h — yaw is the hardest (its signature is a power deficit that overlaps natural wind variability).

Precision is ~1.0 because the 99th-percentile threshold plus 3-hour persistence is conservative; recall is limited mainly by the start of each ramp, where the fault is labelled (severity > 0) but still physically negligible.

## Model

[`model.py`](model.py) defines:

| Block | Definition |
|---|---|
| `DeepSVDDNetwork(input_dim=64, rep_dim=32, hidden=(128, 96, 64, 48, 32))` | 6 bias-free `Linear` layers with `LeakyReLU(0.1)` between them; `forward(x)` returns `z`; `model.c` holds the centre once initialised |
| `init_center(model, loader, device="cpu", eps=0.1)` | Forward pass over `loader` (batches as 1-tuples `(x,)`); sets and returns `model.c` = mean representation, with every coordinate pushed to at least `eps` in magnitude |
| `svdd_loss(outputs, c)` | Mean over the batch of squared distances `‖z − c‖²` |
| `anomaly_score(model, x)` | Additive helper: per-sample `‖φ(x) − c‖²` (eval mode, no grad) |
| `train_svdd(model, X_train, X_val=None, epochs=20, ...)` | Additive helper: fixed-budget mini-batch Deep SVDD training |

- **Input:** feature tensor `x` of shape `(batch_size, input_dim)` (`float32`).
- **Output:** representation `z` of shape `(batch_size, rep_dim)`; anomaly score is `‖z − c‖²`.
- **Size:** 32,256 parameters for the default `(input_dim=64, rep_dim=32)` configuration (28,416 when the platform instantiates it on its 38-column feature table with `rep_dim=16`). The original scaffold was 7,296 / 5,104.
- **Import is side-effect free:** no seeding, no training, no file I/O on `import model` — the pipeline runs only under `python model.py`.

`requirements.txt` lists `torch` and `numpy`. Verified with Python 3.11 and PyTorch 2.14.

## Platform contract

| | |
|---|---|
| Input (adapter view) | tabular features (`batch.features`) |
| Output (`ModelOutput` / `WTDataSchema`) | `anomaly_score` — robust-calibrated distance to the hypersphere centre |
| Integration role | One-class anomaly stream (`TaskType.ANOMALY_DETECTION`), routed as a Hermes `anomaly` voter |
| Deployment / fallback | `cloud`, `edge_gpu`, `edge_cpu` (requires `torch`) · falls back to `m14-isolation-forest` |

The platform adapter `m13-deep-svdd` calls the API exactly as defined here, unchanged by Model 6: `DeepSVDDNetwork(input_dim=..., rep_dim=16)`, `init_center(model, [(Xs,)])` (note the 1-tuple batches), and `svdd_loss(model(Xs), model.c)` — 60 full-batch Adam steps (lr 1e-3, weight decay 1e-5) on its own robust-standardized features. The adapter does not use this repo's whitening or training helpers. Verified against the platform's `test_deep_svdd_detects_ramp` (passes); across 20 unseeded adapter runs the late-vs-early anomaly-score margin was 117,887 minimum / 377,858 median (scaffold: 32,331 / 83,097; the test requires > 1.0). Records emitted via `ModelOutput.to_records()` follow `WTDataSchema` (`wt-pm.platform.v1`).

To run this model inside the platform (Python ≥ 3.10):

```bash
pip install "wt-pm[torch] @ git+https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly.git#subdirectory=platform"
wt-pm fetch-models --dir ./external      # clones this repo and its siblings into ./external
export WTPM_EXTERNAL_DIR=$PWD/external   # adapters import external/<repo>/model.py
wt-pm inspect                            # m13-deep-svdd should report "available": true
```

More on how the 25 models fit together: [platform architecture](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/ARCHITECTURE.md) and the [per-model table](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/platform/docs/MODELS.md).

## Model Info
- **Repo name:** `wt-pm-deep-svdd-boundary`
- **Category:** SCADA Anomaly Detection
- **Model:** 6-layer bias-free Deep SVDD encoder (~32K parameters default / ~28K in platform configuration)
- **Dependencies:** `torch`, `numpy`

## Honest limits

- **Synthetic data only.** The farm simulator is physically motivated but simple; numbers above are not field results. On real SCADA (fidelity rung 1 of the platform's [fidelity ladder](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/docs/fidelity_ladder.md)) expect lower recall and more false alarms.
- **The deep model's margin over a linear baseline is modest.** Mahalanobis distance on the same whitened features reaches AUROC 0.877 vs Model 6's 0.885. Model 6's clearer advantage is at the alarm level — higher F1, more events caught, earlier detection and ~25× fewer false-alarm events.
- **Training length matters and is fixed, not tuned per run.** AUROC peaks around 10 epochs and drifts down slowly after (seed 42: 0.876 at 10, 0.865 at 20, 0.853 at 40). The 20-epoch budget was chosen before looking at test results and kept for every seed.
- **One-class assumption:** boundary quality depends on the training window being genuinely healthy; contamination pulls the centre toward faults.
- `init_center` pushes centre coordinates smaller than `eps=0.1` in magnitude out to ±`eps` (standard Deep SVDD practice; a bias-free network can otherwise reach the origin trivially).
- Inside the platform, scores are calibrated by the adapter (robust z), not by this repository, and the adapter's 60-step schedule (no whitening) is its own choice.
