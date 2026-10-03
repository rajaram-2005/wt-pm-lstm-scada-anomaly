# Changelog

## Unreleased - 25-page collection pages with advanced concepts

- `deployment/models.catalog.json`: single source of truth for all 25 models -
  both numbering schemes (collection 01-25, adapter `m01`-`m25`), architecture,
  contract, 3-4 advanced-concept notes each, platform-layer participation,
  honest limits, 31 functional edges, 8 advanced layers, 96-concept index.
- `deployment/build_model_pages.py`: renders `README.md`, `index.html` and
  `docs/index.html` for all 25 repositories from that catalog, plus the merged
  `all-models.html` index and the 25 pages on the static site.
- `deployment/check_ecosystem.py`: verifies every page's number, adapter, status,
  cross-links, advanced section, `gh-pages` copy and Pages build status.
- `deployment/rollout_all.sh` / `rollout_one.sh`: clone -> branch -> PR -> merge
  -> clean `gh-pages` rebuild for the 24 sibling repositories.
- `deployment/model-pages/`: one `git am`-able patch per sibling repository
  (24), generated from the catalog and verified against each `origin/main`,
  for when the token cannot push. `generate.sh`, `apply-all.sh`, README.
- Published: the reference site now serves the merged index and all 25 model
  pages (`/all-models.html`, `/model-01-…` … `/model-25-…`) via the `pages`
  workflow on `main`.
- `docs/ecosystem.html`: advanced-layer definitions, per-model page links and an
  explicit explanation of the two numbering schemes.
- `docs/launch/`: the static site now ships all 25 model pages plus the merged
  index (`all-models.html`), and both Pages workflows verify them.
- `platform/tests/test_model_pages.py`: locks catalog <-> `ModelRegistry`
  agreement and re-renders/verifies all 25 pages in CI.
- `platform/docs/MODELS.md`: rewritten around the catalog, both numbering
  schemes and the advanced layers.

## 0.1.0

- Model 13 reference engine (`wtpm`) + unified platform (`wt-pm`)
- 25 adapters, fusion, Hermes agent, XAI, edge/ESP32 export
- SCADA bridge: `wt-pm scada`, `wt-pm watch`, `POST /scada`
- Docker compose (serve + drop-folder watcher)
- OIDC PyPI publish workflow; JSON/TCP mock bus; `--stub` siblings; HoT field sample; SIL software gate
