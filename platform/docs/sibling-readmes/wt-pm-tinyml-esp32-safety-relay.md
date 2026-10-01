# TinyML ESP32 Safety Relay

> **Category:** Edge AI  
> **Platform adapter:** `m25-tinyml-safety`  
> **Status:** integrated

## Description

C++ exported TinyML decision trees for bare-metal execution on ESP32 microcontrollers triggering immediate safety trips.

This is model **m25** of the WT-PM wind-turbine predictive-maintenance suite. The repository stays
independently usable (`python model.py`). The unified **wt-pm** platform wraps the original
`model.py` through adapter `m25-tinyml-safety` — it does not replace or fork this code.

## Model

`export_esp32_header(X, y)` in [`model.py`](model.py) fits a depth-5
`DecisionTreeClassifier` and uses **micromlgen**'s `port()` to emit `model.h` — a C++ header an
ESP32 Arduino sketch can `#include` and call `predict()` on, with no Python at runtime.

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

`python model.py` defines `export_esp32_header`; nothing is exported until features/labels are supplied. `requirements.txt` lists `scikit-learn micromlgen`.

To use the model from your own code, import from `model.py` directly — the entry points above
are the exact symbols the platform adapter calls.

## Platform contract

| | |
|---|---|
| Input (adapter view) | the safety-critical SCADA channels (overspeed, temperatures, vibration limits) |
| Output (WTDataSchema) | prediction — TRIP / continue; extra['model_h'] — the exported C++ header |
| Integration role | Safety gate: a pure-Python mirror of the same tree runs in-loop as the platform's last-word gate (`wt-pm edge` emits `model.h`). Hermes `safety` tool. |
| Deployment / fallback | cloud, MCU, edge-CPU · scikit-learn + micromlgen · no fallback (it IS the gate) |

Records follow `WTDataSchema` (`wt-pm.platform.v1`): `timestamp`, `turbine_id` and `model_id` are required;
this adapter also fills `prediction` (TRIP/continue), `inference_time_ms` and `explanation`. The remaining optional fields (`probability`, `anomaly_score`, `degradation_state`, `rul_hours`, `uncertainty`) are left to other models.

## Hermes agent + explainable AI

Hermes calls the `safety` tool; the gate's TRIP/continue decision and the hard limits it checked are quoted verbatim in the trace — never invented.

To run this model inside the platform (Python ≥ 3.10):

```bash
pip install "git+https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly.git#subdirectory=platform"
wt-pm fetch-models --dir ./external      # clones this repo and its siblings into ./external
export WTPM_EXTERNAL_DIR=$PWD/external   # adapters import external/<repo>/model.py
wt-pm inspect                            # m25-tinyml-safety should report "available": true
wt-pm serve --port 8100                  # HTTP API, e.g. GET /health and GET /registry
```

More on how the 25 models fit together: [platform architecture](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/ARCHITECTURE.md)
and the [per-model table](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/platform/docs/MODELS.md).

## Honest limits

- Depth-5 tree, as in the repo: deliberately small enough to fit an ESP32, not to maximise accuracy.
- The Python mirror and the exported C++ tree are the same fitted object; the platform verifies they agree.
- Metrics on the platform are held-out **simulator** records (fidelity rung 1), not certified safety evidence.
