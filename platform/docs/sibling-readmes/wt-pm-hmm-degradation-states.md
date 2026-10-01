# HMM Degradation States

> **Category:** RUL Prognostics  
> **Platform adapter:** `m15-hmm-degradation`  
> **Status:** integrated (severity-ordered states)

## Description

Hidden Markov Model evaluating discrete unobservable degradation state transitions from continuous SCADA signals.

This is model **m15** of the WT-PM wind-turbine predictive-maintenance suite. The repository stays
independently usable (`python model.py`). The unified **wt-pm** platform wraps the original
`model.py` through adapter `m15-hmm-degradation` — it does not replace or fork this code.

## Model

`train_hmm(csv_path="scada_seq.csv", n_states=4)` in [`model.py`](model.py) fits a
`GaussianHMM` (`diag` covariance, 100 iterations, fixed seed 42) to a sequence table,
decodes the most likely state path and prints the transition matrix.

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

`python model.py` defines `train_hmm`; it loads no data until a CSV path is supplied. `requirements.txt` lists `hmmlearn numpy pandas`.

To use the model from your own code, import from `model.py` directly — the entry points above
are the exact symbols the platform adapter calls.

## Platform contract

| | |
|---|---|
| Input (adapter view) | health-indicator series built by the platform from the SCADA batch |
| Output (WTDataSchema) | degradation_state, probability — state posterior per timestamp |
| Integration role | Answers HOW SEVERE. States are re-ordered by mean health indicator so state index is monotone in severity; the decay rate feeds the particle filter (m16). |
| Deployment / fallback | cloud · hmmlearn · no fallback |

Records follow `WTDataSchema` (`wt-pm.platform.v1`): `timestamp`, `turbine_id` and `model_id` are required;
this adapter also fills `degradation_state`, `probability`, `inference_time_ms` and `explanation`. The remaining optional fields (`prediction`, `anomaly_score`, `rul_hours`, `uncertainty`) are left to other models.

## Hermes agent + explainable AI

Hermes uses the severity-ordered state sequence in its severity assessment; the state index is quoted verbatim in observations — never invented. XAI is m21 SHAP + contrastive healthy-band z + counterfactuals; the Hermes trace is the operator-readable explanation.

To run this model inside the platform (Python ≥ 3.10):

```bash
pip install "git+https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly.git#subdirectory=platform"
wt-pm fetch-models --dir ./external      # clones this repo and its siblings into ./external
export WTPM_EXTERNAL_DIR=$PWD/external   # adapters import external/<repo>/model.py
wt-pm inspect                            # m15-hmm-degradation should report "available": true
wt-pm serve --port 8100                  # HTTP API, e.g. GET /health and GET /registry
```

More on how the 25 models fit together: [platform architecture](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/ARCHITECTURE.md)
and the [per-model table](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/platform/docs/MODELS.md).

## Honest limits

- The repo's `train_hmm` reads a CSV; the platform passes the same estimator mask-filtered rows instead of a file.
- hmmlearn orders states arbitrarily — the adapter re-orders them post-hoc by mean health indicator; the repo itself is untouched.
- Metrics on the platform are held-out **simulator** records (fidelity rung 1), not certified asset performance.
