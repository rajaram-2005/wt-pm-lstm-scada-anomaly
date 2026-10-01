# Model 1–6 fix bundles — 2026-10-01

These patches correct the cross-repo confusion from the previous session:

| File | Target repo | What it fixes |
|---|---|---|
| `wt-pm-1d-cnn-bearing-vibration.patch` | m01 (1d-cnn-bearing) | Landing page card + working "View on GitHub" link |
| `wt-pm-aerozip-autoencoder-compressor.patch` | m23 (aerozip) | Landing page card + GitHub link |
| `wt-pm-contrastive-ssl-vibration.patch` | m08 (contrastive-ssl) | Landing page card + GitHub link |
| `wt-pm-convlstm-wear-prognostics.patch` | m02 (convlstm-wear) | Landing page card (was missing platform adapter / status / GitHub link) |
| `wt-pm-dbn-feature-extraction.patch` | m09 (dbn-features) | **Removes** the cross-repo staging artifacts that were accidentally committed to `docs/` (audit docs for models 1–4 and model 6, patches for other repos, the full Model 6 payload under `docs/model-6-deep-svdd/`, stray `models.lock.*.json`). The 5-layer DBN + `results/model5_metrics.json` is untouched. Landing page card corrected. |
| `wt-pm-deep-svdd-boundary.patch` | m13 (deep-svdd) | **Upgrades** Model 6 from the 29-line 3-layer scaffold to the full 6-layer bias-free Deep SVDD encoder (556-line `model.py` with 24-turbine synthetic farm, ZCA whitening, fixed 20-epoch schedule, per-fault delay reporting, scaffold + Mahalanobis comparisons, two results JSON files). New README, landing page, and slimmed `requirements.txt`. Adapter API (`DeepSVDDNetwork(input_dim, rep_dim)`, `init_center`, `svdd_loss`) is preserved. |

## How to apply

Run `bash deployment/model-fixes/apply-all.sh` from the root of a clone of
`wt-pm-lstm-scada-anomaly`, with the six sibling repos cloned as siblings of
this directory (i.e. `../wt-pm-1d-cnn-bearing-vibration`, etc.). Set
`REPO_PARENT` if they live elsewhere.

The script:
1. Checks out each sibling repo's `main`, pulls, and creates branch
   `arena/01a0f76b-models-1to6-fixes`.
2. Applies the corresponding `.patch` via `git am`.
3. Pushes the fix branch and opens a PR on each repo via `gh pr create`.
4. Rebuilds each repo's `gh-pages` branch as a clean one-file orphan branch
   containing only `index.html` (the previous `gh-pages` had a stale
   full-repo snapshot with broken links) and force-pushes it.

After the PRs merge, re-run `deployment/fetch_models.py` to re-pin
`deployment/models.lock.json` to the new `main` SHAs (the lock currently
points to pre-fix SHAs and will report mismatches until re-pinned).

## After applying

- Merge the six PRs that the script opens.
- Confirm each GitHub Pages site (e.g. https://rajaram-2005.github.io/wt-pm-deep-svdd-boundary/) shows the correct badge, adapter id, and a working "View on GitHub" button that lands on the repo (not `#`).
- Pull `main` on this platform repo and run:
  ```bash
  python deployment/fetch_models.py --dir /tmp/external --write-lock   # re-pin SHAs
  python deployment/check_full.py /tmp/external                       # verify all 25 adapters load
  ```
- Commit the updated `deployment/models.lock.json`.
