# TCN Power Curve

> **Category:** SCADA Anomaly Detection  
> **Platform adapter:** `m03-tcn-power-curve`  
> **Status:** integrated

## Description

Temporal Convolutional Network for parallel sequence processing and aerodynamic power-curve degradation tracking.

This repository stays independently usable (`python model.py`). The unified
**wt-pm** platform wraps the original `model.py` through adapter `m03-tcn-power-curve` —
it does not replace or fork this code.

## Platform contract

| | |
|---|---|
| Input (adapter view) | sequence windows, power_kw |
| Output (WTDataSchema) | anomaly_score (power residual) |
| Integration role | Anomaly stream in fused WHAT. Hermes tool `anomaly`. |

Shared record fields: `timestamp`, `turbine_id`, `subsystem`, `prediction`,
`probability`, `anomaly_score`, `degradation_state`, `RUL`, `uncertainty`,
`model_id`, `inference_time`, `explanation`.

## Hermes agent + explainable AI

The platform Hermes loop (Thought → Action → Observation) may call this
model as a **tool**. Observations are the adapter’s real outputs — never
invented. XAI is m21 SHAP + contrastive healthy-band z + counterfactuals;
the Hermes trace is the operator-readable explanation.

```bash
pip install "git+https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly.git#subdirectory=platform"
wt-pm inspect          # see m03-tcn-power-curve availability
wt-pm agent --days 8   # Hermes trace including this model when routed
wt-pm serve --port 8100
```

## Structure

```
├── model.py           # Core architecture / training entry (unchanged)
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup (standalone)

```bash
pip install -r requirements.txt
python model.py
```

## Honest limits

Architecture stub unless noted complete. No trained weights or field
datasets ship here. Metrics on the platform are held-out **simulator**
records (fidelity rung 1), not certified asset performance.
