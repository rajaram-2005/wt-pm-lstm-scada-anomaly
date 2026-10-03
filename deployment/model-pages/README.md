# Ready-to-commit patches: the 25 model pages

One patch per sibling repository, carrying the page rendered from
[`deployment/models.catalog.json`](../models.catalog.json) — the same catalog
the reference repository's own pages come from.

| | |
|---|---|
| Patches | 24 (`wt-pm-<name>.patch`), byte-identical in structure to the historical `deployment/model-fixes/` bundle |
| Contents | `README.md`, `index.html`, `docs/index.html` — replaced wholesale |
| Untouched | `model.py`, `requirements.txt`; every repository stays independently runnable |
| Format | `git format-patch` output — apply with `git am`, no clone-side tooling needed |

## Why this exists alongside `rollout_all.sh`

`deployment/rollout_all.sh` and `deployment/model-pages/apply-all.sh` both clone,
branch, commit, push, open a PR, merge and rebuild `gh-pages`. This bundle is for
the case where the pushing has to happen from somewhere else: the patches are
self-contained, verifiable offline, and can be attached to a PR, applied by hand,
or replayed by a maintainer on a machine that has write access.

## Use

```bash
# 1. verify the patches still apply to each repository's current main
bash deployment/model-pages/generate.sh --check

# 2. see what would happen
bash deployment/model-pages/apply-all.sh --dry-run

# 3. push branch -> PR -> merge -> clean gh-pages, then verify
bash deployment/model-pages/apply-all.sh
python deployment/check_ecosystem.py --remote --strict
```

Without push access (HTTP 403 at the push step) the patches can still be applied
manually:

```bash
git clone https://github.com/rajaram-2005/wt-pm-gru-scada-telemetry
cd wt-pm-gru-scada-telemetry
git am /path/to/deployment/model-pages/wt-pm-gru-scada-telemetry.patch
git push origin HEAD:advanced-pages
```

## Regenerating

After any catalog change:

```bash
python deployment/build_model_pages.py --out build/pages
bash deployment/model-pages/generate.sh          # rewrites all 24 patches
bash deployment/model-pages/generate.sh --check  # only verifies they apply
```
