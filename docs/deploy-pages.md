# GitHub Pages: static website

Public URL: https://rajaram-2005.github.io/wt-pm-lstm-scada-anomaly/

Pages hosts the launch site, pitch, architecture explorer and media. It cannot
run Python, train models, receive SCADA uploads, or replace the Render API.
The website links to the separate Render dashboard and its actual model health.
Publishing this site does not make more models available on Render.

## One-time branch authorization (repository administrator)

The Pages environment currently allows `main` and `gh-pages`. To publish the
prepared update from this session without pushing to those branches:

1. Open repository **Settings → Environments → github-pages**.
2. Under **Deployment branches and tags → Selected branches and tags**, add a
   **Branch** rule for `arena/01a1001d-wt-pm-lstm-scada-anomaly`.
3. Keep the existing rules and protection settings; do not enable all branches.
4. Under **Settings → Pages**, keep the source as **GitHub Actions** (already set).

The integration could read these settings but received HTTP 403 when attempting
to add the exact branch rule. An administrator must do that step.

## Publish

After authorization, open **Actions → pages → Run workflow** and choose
`arena/01a1001d-wt-pm-lstm-scada-anomaly`. Or use:

```bash
gh workflow run pages.yml --ref arena/01a1001d-wt-pm-lstm-scada-anomaly
```

The workflow stages `docs/launch/` plus `docs/ecosystem.html` — the launch page,
the pitch, the ecosystem graph, the merged `all-models.html` index and the 25
model pages (`model-01-…` … `model-25-…`, all rendered from
`deployment/models.catalog.json`) — checks local links/assets, verifies that all
25 pages agree with the catalog, uploads the site and deploys with
`actions/deploy-pages`.
Pull requests only build; manual publishing is limited to `main` and the exact
session branch. It does not push to `main` or `gh-pages`.

A successful build alone is **not** proof of publication. Confirm the **deploy**
job succeeds and the live homepage contains **Open dashboard on Render** before
calling the update published. If the deploy job is rejected by a branch policy,
check the environment rule above.

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
