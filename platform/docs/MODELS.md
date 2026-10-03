# All 25 WT-PM models — numbering, I/O, platform role, advanced concepts

**Single source of truth:** [`deployment/models.catalog.json`](../../deployment/models.catalog.json)
in the reference repository. It holds, for every one of the 25 models: collection
number, adapter id, repository, status, architecture, contract, advanced-concept
notes, platform-layer participation and honest limits — plus the 31 functional
edges and the 8 advanced layers.

**Generated pages:** `deployment/build_model_pages.py` renders `README.md`,
`index.html` and `docs/index.html` for all 25 repositories from that catalog, and
a copy of all 25 pages onto the static site
([all-models.html](https://rajaram-2005.github.io/wt-pm-lstm-scada-anomaly/all-models.html)).

**Verification:** `deployment/check_ecosystem.py --dir <clones> --remote --strict`
checks that each page states the right number/adapter/status, links all 25 pages,
carries the advanced-concepts section, ships the same page on `gh-pages`, and that
GitHub reports Pages as *built*.

## Two numbering schemes, both printed on every page

| Scheme | Range | Used by | This repository |
|---|---|---|---|
| Collection number | `01`…`25`, **alphabetical** by repository name | repository pages, PR titles, the wiring scripts | **13** |
| Adapter id | `m01`…`m25`, **task-based** | `ModelRegistry`, the `wt-pm` CLI, adapters, fusion weights | `m05-lstm-scada-anomaly` |

Earlier documents used the two interchangeably (for example "model 05" for this
repository in this file, "model 13" in `ARCHITECTURE.md`). The catalog now records
both fields per model, and every generated page prints both, so the mapping cannot
drift again.

## Rolling the pages out to the 24 sibling repositories

The pages are generated here and pushed from here; siblings stay independently
runnable and their `model.py` is never touched.

```bash
python deployment/build_model_pages.py --out build/pages      # render all 25
bash deployment/rollout_all.sh --dry-run                      # what would happen
bash deployment/rollout_all.sh                                # branch, PR, merge, gh-pages
python deployment/check_ecosystem.py --dir build/pages --remote --strict
```

`rollout_all.sh` clones each sibling into `build/rollout/`, applies the rendered
README + landing page on a branch, opens a PR, merges it, rebuilds `gh-pages` as a
clean single-file site and re-verifies. It needs a token with **write** access to
`rajaram-2005/wt-pm-*`; a read-only token fails with HTTP 403 at the push step.

### When the token cannot push

[`deployment/model-pages/`](../../deployment/model-pages/) holds **one
`git am`-able patch per sibling repository** (24 files, generated from the same
catalog). They apply to each repository's current `main` without any clone-side
tooling, so a maintainer - or a PR from a machine with write access - can land the
same pages:

```bash
bash deployment/model-pages/generate.sh --check   # still applies to origin/main?
bash deployment/model-pages/apply-all.sh --dry-run
bash deployment/model-pages/apply-all.sh          # git am -> PR -> merge -> gh-pages
```

The reference repository (this one) publishes its own site through the `pages`
workflow: a push to `main` stages `docs/launch/` + `docs/ecosystem.html`, verifies
all 25 pages against the catalog and deploys. Manual `workflow_dispatch` from a
feature branch is limited by the `github-pages` environment's branch policy
(`main` and `gh-pages` are the allowed deployment branches).

## Models

| # | adapter | repository | task | status |
|---|---------|------------|------|--------|
| 01 | `m01-1dcnn-bearing` | wt-pm-1d-cnn-bearing-vibration | fault classification | integrated (surrogate waveforms) |
| 02 | `m23-aerozip` | wt-pm-aerozip-autoencoder-compressor | compression | integrated |
| 03 | `m08-contrastive-ssl` | wt-pm-contrastive-ssl-vibration | feature extraction | partial (platform supplies NT-Xent) |
| 04 | `m02-convlstm-wear` | wt-pm-convlstm-wear-prognostics | RUL | partial (no 64×64 wear maps exist) |
| 05 | `m09-dbn-features` | wt-pm-dbn-feature-extraction | feature extraction | integrated (CD-1 pretraining) |
| 06 | `m13-deep-svdd` | wt-pm-deep-svdd-boundary | anomaly | integrated (6-layer bias-free encoder) |
| 07 | `m19-digital-twin` | wt-pm-digital-twin-surrogate | surrogate | integrated (physics-proxy targets) |
| 08 | `m20-gnn-cascade` | wt-pm-gnn-turbines-cascade | graph | integrated (wake graph from simulator) |
| 09 | `m04-gru-scada-telemetry` | wt-pm-gru-scada-telemetry | anomaly | integrated |
| 10 | `m15-hmm-degradation` | wt-pm-hmm-degradation-states | degradation | integrated (severity-ordered states) |
| 11 | `m06-informer-forecast` | wt-pm-informer-long-sequence | forecasting / anomaly | partial (placeholder attention) |
| 12 | `m14-isolation-forest` | wt-pm-isolation-forest-telemetry | anomaly | integrated (universal fallback) |
| 13 | `m05-lstm-scada-anomaly` | **wt-pm-lstm-scada-anomaly** (this repo) | anomaly + contract hub | integrated — reference implementation |
| 14 | `m17-mlp-rul` | wt-pm-mlp-rul-regression | RUL | integrated (proxy target) |
| 15 | `m16-particle-filter-rul` | wt-pm-particle-filter-rul | RUL | complete |
| 16 | `m18-pg-bnn` | wt-pm-pg-bnn-wind-turbine | anomaly + uncertainty | integrated (P = τω loss) |
| 17 | `m24-quantized-edge` | wt-pm-quantized-mobilenet-edge | edge | integrated (recipe on platform edge net) |
| 18 | `m10-random-forest` | wt-pm-random-forest-telemetry | fault classification | integrated |
| 19 | `m07-snn-vibration` | wt-pm-snn-event-vibration | fault classification | integrated (surrogate events) |
| 20 | `m12-svm-generator` | wt-pm-svm-rbf-generator-stator | fault classification | integrated (electrical specialist) |
| 21 | `m03-tcn-power-curve` | wt-pm-tcn-power-curve | anomaly | integrated |
| 22 | `m25-tinyml-safety` | wt-pm-tinyml-esp32-safety-relay | safety | integrated |
| 23 | `m22-vae-reconstruction` | wt-pm-vae-reconstruction-loss | anomaly | integrated (repo's `vae_loss`) |
| 24 | `m21-xai-shap` | wt-pm-xai-shap-interpretable | explainability | integrated |
| 25 | `m11-xgboost-tabular` | wt-pm-xgboost-tabular-faults | fault classification | integrated (primary tabular voter) |

## Advanced layers

Defined once in the catalog, cited by every model page that participates:

| Layer | What it does | Models |
|---|---|---|
| Fusion engine | Weighted / confidence-weighted / max / median, inverse-variance weighting under uncertainty, coverage-aware probability fusion | 3, 4, 6, 9, 10, 11, 12, 13, 16, 18, 20, 21, 22, 23, 25 |
| Calibrated uncertainty | Intervals and confidences on every alert; expected calibration error at evaluation time | 13, 15, 16, 23 |
| Drift & data trust | Training-vs-live distribution comparison; freeze / spike / stuck-at / range sensor flags | 2, 3, 5, 6, 13 |
| Explainability | SHAP TreeExplainer, contrastive healthy-band z, counterfactuals, Hermes narrative | 1, 18, 24, 25 |
| Physics constraints | Digital-twin residuals and what-if; P = τω soft constraint with physics-consistent uncertainty | 7, 16 |
| Safety gate | Depth-5 tree → `model.h` on ESP32 plus a mirrored in-loop gate; excluded from fusion | 22 |
| Evaluation protocol | Leak-free window masks, ROC-AUC / PR-AUC / calibration / RUL metrics, fidelity ladder | 1, 3, 5, 13, 15 |
| Hermes agent | Thought → Action → Observation over `sensor_quality, anomaly, diagnose, explain, rul, safety, what_if, finish` | 7, 9, 13, 15, 16, 22, 24, 25 |

**Hermes tools** (Thought → Action → Observation): `sensor_quality`, `anomaly`,
`diagnose`, `explain`, `rul`, `safety`, `what_if`, `finish`.

**XAI stack:** m21 SHAP + healthy-band contrastive z + feature counterfactual +
Hermes narrative (observations never invented).
