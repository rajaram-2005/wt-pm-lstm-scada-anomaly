# GitHub Pages: static website

Public URL: https://rajaram-2005.github.io/wt-pm-lstm-scada-anomaly/

Pages hosts the launch site, pitch, architecture explorer and media. It cannot
run Python, train models, receive SCADA uploads, or replace the Render API.
The website links to the separate Render dashboard and its actual model health.
Publishing this site does not make more models available on Render. For adding a
private, authenticated backend to a static site, follow the
[SCADA/website integration protocol](SCADA_WEBSITE_INTEGRATION_PROTOCOL.md);
do not upload plant data to the public research demo.

## Deployment path (`main`)

The `github-pages` environment allows `main` and `gh-pages` as deployment
branches, with **Settings → Pages** configured to deploy from **GitHub Actions**.
Merging a pull request into `main` automatically runs `.github/workflows/pages.yml`
(`build` followed by `deploy`) and publishes the site without needing extra
branch rules.

To trigger a manual redeploy from `main`:

```bash
gh workflow run pages.yml --ref main
```

To deploy from a non-`main` branch via `workflow_dispatch`, a repository
administrator must first add that branch under **Settings → Environments →
github-pages → Deployment branches and tags** (the session integration receives
HTTP 403 on the environment branch-policy API).

## What the workflow stages and verifies

The workflow stages `docs/launch/` plus `docs/ecosystem.html` — the launch page,
the pitch, the ecosystem graph, the merged `all-models.html` index and the 25
model pages (`model-01-…` … `model-25-…`, all rendered from
`deployment/models.catalog.json`) — checks local links/assets, verifies that all
25 pages agree with the catalog, uploads the site and deploys with
`actions/deploy-pages`.
Pull requests run the `build` job (including `check_pages.py` and
`check_ecosystem.py --site _site --strict`); pushes to `main` also run `deploy`.

Confirm the **deploy** job succeeds and the live site serves `/`,
`/all-models.html`, `/ecosystem.html`, and `/model-01-…` through `/model-25-…`.

## Local static check

```bash
mkdir -p artifacts/pages
cp -a docs/launch/. artifacts/pages/
cp docs/ecosystem.html artifacts/pages/ecosystem.html
python deployment/check_pages.py artifacts/pages
python deployment/check_ecosystem.py --site artifacts/pages --strict
```

Regenerate the 25 model pages (and the merged index) after editing the catalog:

```bash
python deployment/build_model_pages.py --site docs/launch
```
