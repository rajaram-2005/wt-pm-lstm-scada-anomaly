# TinyML ESP32 Safety Relay

> **Category:** Edge AI  
> **Platform adapter:** `m25-tinyml-safety`  
> **Status:** integrated

## Description

C++-exported TinyML decision trees on bare-metal ESP32 microcontrollers triggering immediate safety trips.

This repository stays independently usable (`python model.py`). The unified
**wt-pm** platform wraps the original `model.py` through adapter `m25-tinyml-safety` —
it does not replace or fork this code.

## Platform contract

| | |
|---|---|
| Input (adapter view) | bearing_vib, gearbox_oil_temp, rotor_speed, power |
| Output (WTDataSchema) | TRIP/continue + model.h |
| Integration role | Hermes `safety` tool. Hard limits still outrank the learned tree. |

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
wt-pm inspect          # see m25-tinyml-safety availability
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
