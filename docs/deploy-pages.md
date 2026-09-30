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
   **Branch** rule for `arena/01a0cc04-wt-pm-lstm-scada-anomaly`.
3. Keep the existing rules and protection settings; do not enable all branches.
4. Under **Settings → Pages**, keep the source as **GitHub Actions** (already set).

The integration could read these settings but received HTTP 403 when attempting
to add the exact branch rule. An administrator must do that step.

## Publish

After authorization, open **Actions → pages → Run workflow** and choose
`arena/01a0cc04-wt-pm-lstm-scada-anomaly`. Or use:

```bash
gh workflow run pages.yml --ref arena/01a0cc04-wt-pm-lstm-scada-anomaly
```

The workflow stages `docs/launch/` plus `docs/ecosystem.html`, checks local
links/assets, uploads the site and deploys with `actions/deploy-pages`.
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
```
