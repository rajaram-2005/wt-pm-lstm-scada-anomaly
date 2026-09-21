# Particle Filter RUL

> **Category:** RUL Prognostics  
> **Platform adapter:** `m16-particle-filter-rul`  
> **Status:** complete

## Description

Stochastic particle filtering fused with deep networks to continuously update probabilistic RUL decay distributions.

This repository stays independently usable (`python model.py`). The unified
**wt-pm** platform wraps the original `model.py` through adapter `m16-particle-filter-rul` —
it does not replace or fork this code.

## Platform contract

| | |
|---|---|
| Input (adapter view) | RUL observations from m17/m02 + m15 states |
| Output (WTDataSchema) | rul_hours, uncertainty |
| Integration role | Hermes `rul` tool. Adapter guards Gaussian likelihood underflow (not patched in this repo). |

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
wt-pm inspect          # see m16-particle-filter-rul availability
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
