# Model wiring fix bundles

Patch bundles that bring every one of the 24 sibling `wt-pm-*` repos to the
same **wired** state: a platform-aware README, a landing page with the correct
platform-adapter card and a working "View on GitHub" link, and a clean
one-file `gh-pages` branch.

- **Batch 1 (2026-10-01, models 1–6)** — also cleaned misplaced files and
  upgraded Model 6 (Deep SVDD). Shipped in platform PR #6.
- **Batch 2 (2026-10-01, models 7–25)** — same wiring for the remaining 18
  repos. Model 13 (`wt-pm-lstm-scada-anomaly`) is this repository and was
  already wired.

> **Superseded (2026-10-03).** The pages these patches produce are now generated
> from a single catalog, with advanced-concept content added to all 25 models:
> `deployment/models.catalog.json` + `deployment/build_model_pages.py`, rolled out
> by `deployment/rollout_all.sh` and verified by
> `deployment/check_ecosystem.py`. The batch-2 patches remain as the historical
> record of the models 7-25 wiring; running the rollout instead of
> `apply-all-7to25.sh` reaches the same repositories with the newer pages.

Model numbers are the alphabetical collection numbering used by
[docs/ecosystem.html](../../docs/ecosystem.html); the platform adapter ids
(`m01`…`m25`) are a second, task-based numbering — the table below gives both.

## Batch 1 — models 1–6

| File | Target repo | What it fixes |
|---|---|---|
| `wt-pm-1d-cnn-bearing-vibration.patch` | m01 (1d-cnn-bearing) | Landing page card + working "View on GitHub" link |
| `wt-pm-aerozip-autoencoder-compressor.patch` | m23 (aerozip) | Landing page card + GitHub link |
| `wt-pm-contrastive-ssl-vibration.patch` | m08 (contrastive-ssl) | Landing page card + GitHub link |
| `wt-pm-convlstm-wear-prognostics.patch` | m02 (convlstm-wear) | Landing page card (was missing platform adapter / status / GitHub link) |
| `wt-pm-dbn-feature-extraction.patch` | m09 (dbn-features) | **Removes** the cross-repo staging artifacts that were accidentally committed to `docs/` (audit docs for models 1–4 and model 6, patches for other repos, the full Model 6 payload under `docs/model-6-deep-svdd/`, stray `models.lock.*.json`). The 5-layer DBN + `results/model5_metrics.json` is untouched. Landing page card corrected. |
| `wt-pm-deep-svdd-boundary.patch` | m13 (deep-svdd) | **Upgrades** Model 6 from the 29-line 3-layer scaffold to the full 6-layer bias-free Deep SVDD encoder (556-line `model.py` with 24-turbine synthetic farm, ZCA whitening, fixed 20-epoch schedule, per-fault delay reporting, scaffold + Mahalanobis comparisons, two results JSON files). New README, landing page, and slimmed `requirements.txt`. Adapter API (`DeepSVDDNetwork(input_dim, rep_dim)`, `init_center`, `svdd_loss`) is preserved. |

Applied by `apply-all.sh` (branch `arena/01a0f76b-models-1to6-fixes`).

## Batch 2 — models 7–25 (18 repos)

Each patch makes one commit on the target repo:

1. **README.md → full platform-aware version** (matches the models 1–6
   READMEs and `platform/docs/sibling-readmes/`): adapter id, status, model
   entry points, platform contract (`WTDataSchema` / `wt-pm.platform.v1`),
   Hermes/XAI role, standalone setup and honest limits.
2. **index.html + docs/index.html → wired landing page**: platform-adapter
   card, status, files list, working "View on GitHub" button (was the
   placeholder `#`), honest footer instead of the scaffold note.
3. `model.py` and `requirements.txt` are **untouched** — every repo stays
   independently runnable (`python model.py`).

| # | File | Target repo | Adapter | Status shown |
|---|---|---|---|---|
| 7 | `wt-pm-digital-twin-surrogate.patch` | digital-twin-surrogate | `m19-digital-twin` | integrated (physics-proxy targets) |
| 8 | `wt-pm-gnn-turbines-cascade.patch` | gnn-turbines-cascade | `m20-gnn-cascade` | integrated (wake graph from simulator) |
| 9 | `wt-pm-gru-scada-telemetry.patch` | gru-scada-telemetry | `m04-gru-scada-telemetry` | integrated |
| 10 | `wt-pm-hmm-degradation-states.patch` | hmm-degradation-states | `m15-hmm-degradation` | integrated (severity-ordered states) |
| 11 | `wt-pm-informer-long-sequence.patch` | informer-long-sequence | `m06-informer-forecast` | partial (placeholder attention) |
| 12 | `wt-pm-isolation-forest-telemetry.patch` | isolation-forest-telemetry | `m14-isolation-forest` | integrated (universal fallback) |
| 14 | `wt-pm-mlp-rul-regression.patch` | mlp-rul-regression | `m17-mlp-rul` | integrated (proxy target) |
| 15 | `wt-pm-particle-filter-rul.patch` | particle-filter-rul | `m16-particle-filter-rul` | complete |
| 16 | `wt-pm-pg-bnn-wind-turbine.patch` | pg-bnn-wind-turbine | `m18-pg-bnn` | integrated (BNN assembled from repo blocks) |
| 17 | `wt-pm-quantized-mobilenet-edge.patch` | quantized-mobilenet-edge | `m24-quantized-edge` | integrated (recipe on platform edge net) |
| 18 | `wt-pm-random-forest-telemetry.patch` | random-forest-telemetry | `m10-random-forest` | integrated |
| 19 | `wt-pm-snn-event-vibration.patch` | snn-event-vibration | `m07-snn-vibration` | integrated (surrogate events) |
| 20 | `wt-pm-svm-rbf-generator-stator.patch` | svm-rbf-generator-stator | `m12-svm-generator` | integrated (electrical specialist) |
| 21 | `wt-pm-tcn-power-curve.patch` | tcn-power-curve | `m03-tcn-power-curve` | integrated |
| 22 | `wt-pm-tinyml-esp32-safety-relay.patch` | tinyml-esp32-safety-relay | `m25-tinyml-safety` | integrated |
| 23 | `wt-pm-vae-reconstruction-loss.patch` | vae-reconstruction-loss | `m22-vae-reconstruction` | integrated (uses repo's vae_loss) |
| 24 | `wt-pm-xai-shap-interpretable.patch` | xai-shap-interpretable | `m21-xai-shap` | integrated |
| 25 | `wt-pm-xgboost-tabular-faults.patch` | xgboost-tabular-faults | `m11-xgboost-tabular` | integrated (primary tabular voter) |

Model 13 (`wt-pm-lstm-scada-anomaly`) is the platform host — no patch needed.

Every batch-2 patch was verified with `git am` on a fresh clone of the
target's current `origin/main` before being committed here.

Applied by `apply-all-7to25.sh` (branch `arena/01a0f786-models-7to25-wiring`).

## How to apply

From the root of a clone of `wt-pm-lstm-scada-anomaly`, with the target
sibling repos cloned as siblings of this directory (or set `REPO_PARENT`):

```bash
bash deployment/model-fixes/apply-all.sh          # batch 1 (models 1–6) — already merged upstream
bash deployment/model-fixes/apply-all-7to25.sh    # batch 2 (models 7–25)
```

Each script:
1. Checks out each sibling repo's `main`, pulls, and creates its fix branch.
2. Applies the corresponding `.patch` via `git am`.
3. Pushes the branch and opens a PR on each repo via `gh pr create`.
4. Rebuilds each repo's `gh-pages` branch as a clean one-file orphan branch
   containing only `index.html` and force-pushes it.

## After applying batch 2

- Merge the 18 PRs that the script opens.

- Confirm each GitHub Pages site (e.g.
  https://rajaram-2005.github.io/wt-pm-xgboost-tabular-faults/) shows the
  correct adapter id, status, and a working "View on GitHub" button.
- Pull `main` on this platform repo and re-pin the lockfile:
  ```bash
  python deployment/fetch_models.py --dir /tmp/external --write-lock
  python deployment/check_full.py /tmp/external       # verify all 25 adapters load
  ```
- Commit the updated `deployment/models.lock.json`.
