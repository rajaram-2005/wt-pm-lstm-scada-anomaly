# wt-pm-lstm-scada-anomaly

**Model 13 of the WT-PM research ecosystem** — LSTM sequence anomaly detection
on wind-turbine SCADA data — **and the host of the unified 25-model platform**.

| | Model 13 (this package) | Unified platform |
| --- | --- | --- |
| Path | `src/wt_pm_lstm/` | `platform/` |
| Install | `pip install -e .` | `pip install -e platform` |
| CLI | `wtpm` | `wt-pm` |
| Role | Detector + schema/registry/eval/API | Orchestrator over all 25 `wt-pm-*` models |

**GitHub architecture (layers, mermaid, 25 adapters, commands):**
[ARCHITECTURE.md](ARCHITECTURE.md) · [ecosystem](docs/ecosystem.html) · [platform report](platform/docs/ARCHITECTURE_REPORT.md)

### Where this repository sits in the collection

The 25 repositories are numbered **alphabetically 01–25** — that is the number
printed on each repository's page — and each one also has a **task-based
platform adapter id**. This repository is **model 13 of 25** and adapter
**`m05-lstm-scada-anomaly`**. Both numbers appear on every page so the
ecosystem graph and the platform registry can be read against each other.

- **Single source of truth:** [`deployment/models.catalog.json`](deployment/models.catalog.json) —
  number, adapter, status, contract, advanced concepts, platform-layer
  participation and the 31 functional edges for all 25 models.
- **Advanced concepts index:** [all 25 model pages](https://rajaram-2005.github.io/wt-pm-lstm-scada-anomaly/all-models.html)
  (fusion, calibrated uncertainty, drift and data trust, explainability,
  physics constraints, the safety gate, the evaluation protocol, the Hermes
  agent) with 96 indexed concepts.
- **Regenerate / verify:** `python deployment/build_model_pages.py --out build/pages`
  and `python deployment/check_ecosystem.py --dir build/pages --strict`.

The WT-PM ecosystem is a 25-model wind-turbine predictive-maintenance research
collection spanning vibration intelligence, SCADA analytics, anomaly detection,
fault classification, RUL prognostics, physics-guided learning, digital twins,
graph learning, explainable AI, and edge deployment. This repository is the
SCADA-sequence branch of that collection, written as a **reference
implementation of the shared contract**: the data schema, model registry,
evaluation protocol, uncertainty reporting, drift monitoring and inference API
that the other models can adopt rather than reinvent. A collection of 25 models
is only a platform if the models can compose.

The 24 sibling repositories are **not replaced**. The platform calls their
original `model.py` through adapters (`external/`, gitignored).

---

## What it does

Ten-minute SCADA records in, timestamped anomaly records out:

```jsonc
{
  "schema_version": "wt-pm.anomaly.v1",
  "turbine_id": "WT-07",
  "timestamp": 1700390600,
  "score": 34.8,
  "threshold": 9.63,
  "alarm": true,
  "uncertainty_std": 0.41,
  "attribution": {"bearing_vib_rms_mm_s": 31.2, "gearbox_oil_temp_c": 5.9},
  "recommended_action": "inspect at next scheduled service",
  "explanation": "fused score 34.8 = 3.6x threshold; largest robust residual deviations: bearing_vib_rms_mm_s +30.9σ (vibration), gearbox_oil_temp_c +2.4σ (thermal)"
}
```

Each record names the channels it blames, states how far past the threshold it
is, carries the ensemble's disagreement, and recommends an action. Records are
emitted per alarm *event*, not per sample.

## Quick start

```bash
pip install -e .                    # numpy only; no framework required
wtpm demo --days 60 --epochs 30     # full protocol -> artifacts/demo/REPORT.md
wtpm serve                          # HTTP API + scoring console on :8000
pytest -q                           # gradient checks, contract tests, protocol tests
```

The **unified 25-model platform** installs as its own terminal command:

```bash
pip install -e platform             # provides the `wt-pm` executable
# or: pip install "git+https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly.git#subdirectory=platform"
wt-pm --version
wt-pm inspect
wt-pm serve --port 8100
```

```python
from wt_pm_lstm.config import RunConfig
from wt_pm_lstm.pipeline import run_experiment

result = run_experiment(RunConfig())
print(result.summary()["f1"], result.summary()["false_alarms_per_day"])
for record in result.detection.records:
    print(record.timestamp, record.recommended_action, record.explanation)
```

## Local / company Docker deployment (on demand)

Run the full 25-adapter research demo on a company computer without a cloud
hosting subscription. Requires Docker and about 4 GB RAM available to it.

```bash
docker compose -f compose.local.yaml build  # once, with internet
docker compose -f compose.local.yaml up -d --no-build --pull never --wait --wait-timeout 600
# Open http://localhost:8100 on this computer.
docker compose -f compose.local.yaml stop   # when finished
```

The service is **localhost-only**, does not auto-restart, and verifies all 25
adapters before becoming ready. The image can run offline after the initial
build. In-memory models/results reset on stop; company production use requires
additional security, persistence and validation.
[Local setup, offline transfer and company-use guide](docs/deploy-local.md).

## Website on GitHub Pages

[Open the project website](https://rajaram-2005.github.io/wt-pm-lstm-scada-anomaly/)
· [Pages publishing instructions](docs/deploy-pages.md)

GitHub Pages serves the static website only. The Python dashboard and model
inference need a separate server; Pages cannot replace that runtime.

For the full-platform SCADA flow, canonical turbine tag map, `/scada` request
examples, secure website proxy pattern, staged field-validation checklist and
current integration limits, see the [SCADA and website integration protocol](docs/SCADA_WEBSITE_INTEGRATION_PROTOCOL.md).

## Deploy the dashboard/API

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/tree/main)

Deploy the Docker-based command center on Render using the included
`render.yaml`. You must sign in to Render and approve the deployment.
[Deployment guide and verification steps](docs/deploy-render.md).

**Need every adapter?** Use the opt-in [full CPU deployment profile](docs/deploy-render.md#enable-all-25-adapters-on-the-existing-render-service)
(`Dockerfile.full`), including the original sibling sources and all required
frameworks. Budget at least 4 GB RAM and approve hosting costs before switching.
The free/demo profile intentionally does not run all 25.

**Public research demo only:** simulated data, no authentication, no persistent
state, and no plant connections. Optional model frameworks and sibling
repositories are not bundled. Do not upload sensitive SCADA data.

## Results (rung 1 — simulated data, see `docs/fidelity_ladder.md`)

Reference run: 60 days, 30 epochs, 2 ensemble members, stride 2.

| metric | value |
| --- | --- |
| precision / recall / F1 (raw point-wise) | 0.751 / 0.632 / **0.686** |
| F1 95 % CI (moving-block bootstrap) | [0.530, 0.811] |
| PR-AUC / ROC-AUC | 0.684 / 0.772 |
| event recall (6 h tolerance) | 4 of 7 fault families |
| median detection latency | 20.5 samples (3.4 h) |
| false-alarm time | 48 min/day |
| point-adjusted F1 (secondary, caveated) | 0.876 |

Per family, as a multiple of the alarm threshold: bearing wear ×31, pitch
misalignment ×10, yaw error ×5.2, gearbox thermal drift ×1.5 (detected late),
sensor freeze detected by the sensor-health rule, while converter fault and
sensor drift stay below threshold. **Three of seven families are missed, and the
README says so rather than quoting the average and moving on.**

## How it works

```text
SCADA window (36 x 12)
   |
   +-- sequence autoencoder (LSTM/GRU, NumPy, hand-derived backward)
   |     reconstruction residual  ----+
   |     one-step forecast residual --+-- per-channel robust z
   |                                  |
   +-- normal-behaviour model (ridge on degree-2 context) -> level z
   |
   +-- trailing-window mean of the signed residual -> trend statistic
   |
   +-- sensor-health rule: a scored channel that stops changing
   |
   fuse per channel -> reduce (max) -> EWMA -> hysteresis -> records
```

Five things worth knowing before reading the code:

1. **Exogenous channels are not scored.** Scoring the wind channel measures the
   weather. Restricting scoring to the eight response channels moved ROC-AUC
   from 0.52 (chance) to 0.77.
2. **Fusion happens per channel, reduction last.** A single-channel fault
   averaged over eight channels disappears: 3.3 σ becomes 1.4 σ.
3. **Residual heads cannot see slow smooth faults.** A fouled oil cooler
   produces a new, stable level that is easy to reconstruct and easy to
   forecast. That is what the normal-behaviour model and the trend statistic are
   for; more capacity never fixes it.
4. **A frozen sensor has no residual.** It is a different failure mode and gets
   its own alarm rule, labelled differently in the records.
5. **The threshold is an operational budget** — ten minutes of alarm time per
   day on healthy machines — not a magic quantile.

### Optional PG-BNN physics cross-check

In the unified platform's full profile, model `m18-pg-bnn` uses the original
`wt-pm-pg-bnn-wind-turbine` Bayesian layers and `P = τω` loss to forecast power
from wind, temperature, rotor speed, pitch and torque. Its `/analyse` response
includes `physics.pg_bnn`: measured and predicted power, measured-minus-model
residual, the efficiency-adjusted mechanical-power residual, and Monte-Carlo
weight spread. Hermes repeats those diagnostics in its explanation. Inference
uses a fixed, isolated posterior sample bank for reproducible comparisons and
calibrates residuals on a held-out healthy tail of the permitted training data.

The approximate `mean ± 1.96 × MC standard deviation` band is **epistemic only**;
it is not a calibrated coverage interval, an aleatoric noise estimate or a
safety guarantee. The minimal deployment profile may not include PyTorch or the
sibling source, so model 18 can be unavailable there. See
[`platform/docs/CORRECTNESS_AUDIT.md`](platform/docs/CORRECTNESS_AUDIT.md) for the
remaining limits.

## The interfaces (what other models can consume)

| interface | module | notes |
| --- | --- | --- |
| data schema | `schema.py` | 12 channels, quality codes, `wt-pm.scada.v1`, versioned |
| anomaly contract | `schema.AnomalyRecord` | `wt-pm.anomaly.v1`, JSON-serialisable |
| model registry | `registry.py` | `.npz` + JSON sidecar, config fingerprint, no pickle |
| experiment tracking | `registry.save_run` | `runs/<name>-<fingerprint>/` with full provenance |
| evaluation protocol | `evaluate.py` | point + event + alarm-time metrics, block bootstrap |
| uncertainty | `detect.py` | ensemble disagreement per window, in every record |
| drift detection | `drift.py` | PSI + Page-Hinkley, reported in every run |
| inference API | `api.py` | `POST /score`, `GET /schema`, `GET /health` (stdlib only) |
| reporting | `report.py` | `REPORT.md` per run, including what the run does *not* establish |

The HTTP API is deliberately dependency-free: the reference engine must run
wherever the research pipeline runs, including next to a DAQ.

```bash
wtpm serve --port 8000
curl -s localhost:8000/schema | head
curl -s "localhost:8000/sample?fault=bearing_wear" > batch.json
curl -s -X POST localhost:8000/score \
     -H 'Content-Type: application/json' \
     --data-binary @batch.json | python -m json.tool | head -40
```

## Reliability engineering

* **Gradients are verified, not assumed.** `tests/test_nn.py` checks every
  hand-derived LSTM/GRU gradient against a central finite difference of the
  training loss, including the 1×1 GRU trace that hides `dh_next` errors.
* **Protocol invariants are tests, not conventions.** Faults cannot reach the
  training band (`split_indices` refuses), the same configuration reproduces the
  same numbers, and a saved bundle scores *identically* to the object it came
  from — a bundle that silently scores differently is a reproducibility failure
  no metric would reveal.
* **Three silent bugs found and pinned by tests during development:** early
  stopping restored its best checkpoint by rebinding the parameter dict while
  the network cells kept the final epoch's weights (the best epoch never ran);
  the bundle loader broke the same parameter aliasing; and per-step metadata was
  not sliced with the data, so every fault event was attributed to the wrong
  family. Each now has a regression test.

## Repository layout

```text
src/wt_pm_lstm/
  schema.py      data contract + channel roles      config.py   run configuration
  dataio.py      CSV/JSON boundary                  simulate.py physics-informed generator
  windows.py     splits, scaling, windowing         baseline.py normal-behaviour model
  nn.py          NumPy LSTM/GRU + Adam              train.py    training loop, early stopping
  detect.py      four-component fusion, alarms      evaluate.py metrics + uncertainty
  drift.py       PSI, Page-Hinkley                  registry.py bundles, model cards, runs
  pipeline.py    end-to-end protocol                report.py   Markdown report
  api.py         HTTP contract                      cli.py      `wtpm` command line
docs/architecture.md      how it works, and its boundary conditions
docs/fidelity_ladder.md   what a metric can claim, and at which rung
tests/                    gradient checks, contract tests, protocol invariants
```

## Status and limitations

This is a research reference implementation at **fidelity rung 1**. It is not
certified for operational use, it has not seen field data, and its metrics are
not transferable to a real asset without re-calibration. Faults are injected in
the test band only, the alarm threshold is a single global value rather than an
operating-point-dependent one, and the drift monitor flags every multi-week run
because a nine-day calibration window does not represent a sixty-day operating
period — the report says so explicitly instead of hiding it.

Gearbox thermal drift, converter fault and sensor drift remain missed or late
relative to the 6-hour tolerance; the normal-behaviour model sees them only
marginally (0.93–1.06 mean |z| against 0.685 healthy). Detecting them reliably
needs either stronger physical evidence in the data source or an explicit
degradation model — not a larger network.

## License

MIT.
