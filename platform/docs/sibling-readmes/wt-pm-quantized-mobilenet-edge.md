# Quantized MobileNet Edge

> **Category:** Edge AI  
> **Platform adapter:** `m24-quantized-edge`  
> **Status:** integrated (recipe on platform edge net)

## Description

Quantized MobileNet variant designed for compressed high-frequency feature extraction on edge hardware.

This is model **m24** of the WT-PM wind-turbine predictive-maintenance suite. The repository stays
independently usable (`python model.py`). The unified **wt-pm** platform wraps the original
`model.py` through adapter `m24-quantized-edge` — it does not replace or fork this code.

## Model

`convert_to_tflite(saved_model_dir="mobilenet_saved", out_path="mobilenet_int8.tflite")` in
[`model.py`](model.py) is an **INT8 TFLite conversion recipe**: `TFLiteConverter.from_saved_model`
with default optimisations, a random-tensor representative dataset, `TFLITE_BUILTINS_INT8` ops and
int8 input/output types, written to disk. A commented stub shows the intended MobileNetV2 base.

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

`python model.py` defines `convert_to_tflite`; it converts nothing until a SavedModel directory is supplied. `requirements.txt` lists `tensorflow numpy`.

To use the model from your own code, import from `model.py` directly — the entry points above
are the exact symbols the platform adapter calls.

## Platform contract

| | |
|---|---|
| Input (adapter view) | any Keras SavedModel + representative data (platform passes its small telemetry edge net) |
| Output (WTDataSchema) | prediction, probability, extra['size_bytes'] — plus the INT8 `.tflite` artifact in edge mode |
| Integration role | Edge conversion service: the INT8 recipe is applied to the platform's small telemetry edge net; full MobileNet is out of scope (no image data exists in the collection). |
| Deployment / fallback | cloud, edge-CPU, edge-GPU · tensorflow · no fallback |

Records follow `WTDataSchema` (`wt-pm.platform.v1`): `timestamp`, `turbine_id` and `model_id` are required;
this adapter also fills `prediction`, `probability`, `inference_time_ms` and `explanation`. The remaining optional fields (`anomaly_score`, `degradation_state`, `rul_hours`, `uncertainty`) are left to other models.

## Hermes agent + explainable AI

Edge artifacts (`model.tflite`, size report) are produced by `wt-pm edge`; observations in the Hermes trace are the adapter's real outputs — never invented.

To run this model inside the platform (Python ≥ 3.10):

```bash
pip install "git+https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly.git#subdirectory=platform"
wt-pm fetch-models --dir ./external      # clones this repo and its siblings into ./external
export WTPM_EXTERNAL_DIR=$PWD/external   # adapters import external/<repo>/model.py
wt-pm inspect                            # m24-quantized-edge should report "available": true
wt-pm serve --port 8100                  # HTTP API, e.g. GET /health and GET /registry
```

More on how the 25 models fit together: [platform architecture](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/ARCHITECTURE.md)
and the [per-model table](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/platform/docs/MODELS.md).

## Honest limits

- Conversion recipe only — no MobileNet weights ship, and none are trained: **there is no image data anywhere in the collection**, so the full MobileNet path is honestly marked out of scope.
- The representative dataset in the recipe is random tensors; real calibration data would slot in unchanged.
- Metrics on the platform are held-out **simulator** records (fidelity rung 1), not certified asset performance.
