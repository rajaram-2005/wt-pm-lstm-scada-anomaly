# CONVLSTM-WEAR-PROGNOSTICS
> **Category:** RUL Prognostics

## Description
Convolutional LSTM extracting joint spatiotemporal representations for continuous mechanical degradation tracking.

**Model 4** extends the original scaffold (2-layer ConvLSTM) to a **4-layer stack** for deeper temporal-hierarchical wear feature extraction:

```
Input (10, 16, 16, 1)   # 10-frame window of 16x16 wear maps
  └─ ConvLSTM2D(32) + BatchNorm     # fine spatiotemporal features
  └─ ConvLSTM2D(48) + BatchNorm     # mid-level degradation patterns
  └─ ConvLSTM2D(48) + BatchNorm     # global wear-rate representation
  └─ ConvLSTM2D(32) + BatchNorm     # regression-ready summary
  └─ Flatten → Dense(64, relu) → Dense(1)   # normalized RUL output
```

- **Optimizer:** Adam (1e-3), MSE loss, MAE tracked
- **Callbacks:** EarlyStopping (patience 3, restore best) + ReduceLROnPlateau
- **Data:** synthetic accelerated-wear trajectory rendered as 2-D wear maps; sliding windows (stride 2) with **chronological** train/val/test split (15% / 20%) to prevent time leakage

## Structure
```
├── model.py           # Model 4 architecture + data synthesis + training script
├── requirements.txt   # Dependencies
├── results/           # Run outputs (metrics JSON, best .h5 weights) — created at runtime
├── .gitignore         # Environment and data exclusions
└── README.md          # Project overview
```

## Setup & Execution
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run Model 4 (data synthesis → training → evaluation):
   ```bash
   python model.py
   ```

Outputs:
- Printed evaluation table (RMSE / MAE / R² on train/val/test, in cycle units)
- `results/model4_metrics.json` — full metrics summary
- `results/model4_weights.h5` — best weights (excluded from git via `.gitignore`)

## Model Info
- **Repo name:** `wt-pm-convlstm-wear-prognostics`
- **Category:** RUL Prognostics
- **Model:** 4-layer ConvLSTM (~0.96M parameters)
- **Dependencies:** `tensorflow numpy pandas`

> Scaffold generated locally for inspection — no GitHub API call made.
