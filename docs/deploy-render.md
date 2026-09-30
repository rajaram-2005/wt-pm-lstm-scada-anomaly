# Deploy the WT-PM dashboard/API to Render

This deploys the **Python command center**, not the static launch website.
It is a public **research demo with simulated SCADA**, not an operational
monitoring or turbine-control service.

## Create the service

1. Open [Deploy to Render](https://render.com/deploy?repo=https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/tree/arena/01a0cc04-wt-pm-lstm-scada-anomaly).
2. Sign in to Render. If prompted, authorize Render's GitHub integration for
   `rajaram-2005/wt-pm-lstm-scada-anomaly`. Do not share access tokens in chat.
3. Review the Blueprint. It creates one Docker web service named
   `wt-pm-command-center`, using the **Free** plan in Frankfurt. Confirm the
   branch is `arena/01a0cc04-wt-pm-lstm-scada-anomaly` and the Blueprint path is
   `render.yaml`. Review any pricing shown by Render before approving.
4. Approve the deployment. Render builds the image, starts the server, and
   checks `/ready`. The probe returns HTTP 503 while the demo models fit, then
   HTTP 200 when initialization is complete.
5. Open the **actual `https://….onrender.com` URL shown on the service page**.
   Click **Run analysis** to exercise the pipeline. The hostname is assigned by
   Render; the repository does not reserve a particular public URL.

If the deploy link cannot select the branch, use **New → Blueprint** in the
Render dashboard, connect the repository, and select the branch and YAML path
above manually.

## Verify

Replace the placeholder with the service URL from Render:

```bash
APP_URL=https://YOUR-SERVICE.onrender.com
curl --fail "$APP_URL/ready"                  # {"status": "ready"}
curl --fail "$APP_URL/health"                 # status + model availability
curl --fail "$APP_URL/"                       # dashboard HTML
curl --fail -X POST "$APP_URL/analyse" \
  -H 'Content-Type: application/json' -d '{}'  # analysis + Hermes explanation
```

## Deployment settings

- `render.yaml` specifies the Dockerfile, branch, free plan and readiness probe.
- `PORT` controls the listening port; the server binds to `0.0.0.0`. An explicit
  CLI `--port` overrides the environment variable.
- The image runs as non-root UID `10001`. Locally mounted SCADA output folders
  must be writable by this UID when using Docker Compose.
- Numerical thread counts are limited to one to reduce resource usage.
- Automatic deploys are **off**. To update, push to the configured branch, then
  use **Manual Deploy → Deploy latest commit** in Render. Use Render's rollback
  controls to restore a previous successful deployment if needed.
- Free instances can sleep when idle; waking the service requires fitting the
  demo models again. Render's current plan limits apply. If the service runs out
  of memory, review logs and consider a larger plan only after approving its cost.

## Important limits

- **No authentication or per-user isolation.** Anyone with the URL can call the
  API, run analysis, acknowledge alerts, or submit feedback. Add authentication,
  request limits and rate limiting before exposing it to untrusted workloads.
- **Do not upload sensitive or real plant data.** Do not attach a SCADA watcher,
  plant network, or control/writeback connection to this public demo.
- Analysis state, alerts and feedback are in memory and reset on restart or
  idle shutdown. There is no persistent database or disk in this Blueprint.
- This image installs the reference package and lightweight platform
  dependencies. The dashboard lists all model adapters, but optional frameworks
  and sibling research repositories are **not** bundled; unavailable adapters
  are reported in `/health`. This is not a deployment of all 25 trained models.
- The standard-library HTTP server is suitable for this demo, not a hardened
  production API. Production use needs an authenticated gateway, resource and
  request limits, persistent state, monitoring, and field-data validation.

## Local check

```bash
pip install -e '.[dev]' -e platform
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest platform/tests/test_deployment.py
PORT=10000 wt-pm serve --days 6
# or, with Docker installed:
docker build -t wt-pm .
docker run --rm -e PORT=10000 -p 10000:10000 wt-pm
```

## Enable all 25 adapters on the existing Render service

The default `Dockerfile` / `render.yaml` deliberately stay lightweight. They do
**not** include the full stack. A normal redeploy of that profile will still
leave most adapters unavailable.

The opt-in `Dockerfile.full` installs CPU PyTorch, TensorFlow/Keras, XGBoost,
HMM, SHAP, snntorch, PyG and the edge-export dependencies. It fetches all 24
original sibling repositories at revisions in `deployment/models.lock.json`,
verifies each `model.py` checksum, and sets `WTPM_EXTERNAL_DIR=/app/external`.
It never uses `fetch-models --stub`.

### Existing service: preserve your public URL

1. In Render, open **wt-pm-command-center → Settings → Instance Type**.
   Review pricing and choose **at least 4 GB RAM** for headroom. The tested
   six-day demo peaked around **1.6 GiB** on the development machine; this is
   not a memory guarantee for other inputs. The free 512 MiB tier is not suitable.
2. In **Settings → Build & Deploy**, change **Dockerfile Path** to
   `./Dockerfile.full`. Keep Docker context `.` and the branch
   `arena/01a0cc04-wt-pm-lstm-scada-anomaly`.
3. Clear any **Docker Command** override, so the image's `wt-pm serve --days 6`
   command is used. Keep the health-check path `/ready` and `PORT=10000`.
4. Set/confirm `WTPM_REQUIRE_ALL_MODELS=1`, `WTPM_WORKERS=1`, and
   `WTPM_EXTERNAL_DIR=/app/external` if the service already has overrides for
   those variables. These defaults are included in the full image.
5. **Manual Deploy → Deploy latest commit**. The larger image takes longer to
   build. Wait for model fitting and the full startup analysis to complete.
6. Open `/health`: expect `profile: "full"`, and counts `registered`,
   `available`, `fitted`, `ran` all equal to **25**. `/ready` becomes HTTP 200
   only after all 25 execute without errors or fallback substitution.
7. Click **Run full analysis**. The model panel distinguishes fitted models
   from models that actually ran, and exposes failures and limitations on hover.

**Blueprint-managed settings:** `render.yaml` still describes the free demo.
Do not re-sync that Blueprint after manually switching to the full profile,
which could restore the lightweight settings. After approving the paid plan,
update the existing Blueprint's Dockerfile/plan to match, or manage the service
settings independently. `render.full.yaml` is an opt-in template for a **new**
full service (paid `pro` plan); using it as a new Blueprint does not upgrade the
existing URL. No paid resource or billing change is applied by this repository.

If startup fails, `/health` reports `status: "error"` and per-model fit reasons;
`/ready` stays HTTP 503. Inspect the Render logs rather than treating dependency
imports or fallback results as proof of success. Roll back to `Dockerfile` and
unset `WTPM_REQUIRE_ALL_MODELS` (or set it to `0`) to restore the lean profile.

### Full-profile verification

With Docker installed:

```bash
docker build -f Dockerfile.full -t wt-pm-full .
docker run --rm --memory=3g --cpus=2 --entrypoint python wt-pm-full \
  deployment/check_full.py
docker run --rm -e PORT=10000 -p 10000:10000 wt-pm-full
```

Without Docker (Python 3.11, git required):

```bash
python -m venv .venv
. .venv/bin/activate
pip install 'torch==2.8.0+cpu' --index-url https://download.pytorch.org/whl/cpu
pip install -r deployment/requirements-full.txt
pip install -e '.[dev]' -e platform
python deployment/fetch_models.py --dest external
export WTPM_EXTERNAL_DIR="$PWD/external"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
python deployment/check_full.py --report artifacts/full-models/verification.json
WTPM_TEST_FULL=1 python -m pytest tests platform/tests -ra
WTPM_REQUIRE_ALL_MODELS=1 WTPM_WORKERS=1 wt-pm serve --days 6
```

`check_full.py` fails if any model is unavailable, absent from the actual run,
errors, or uses a fallback. It additionally checks that SHAP values and an INT8
edge-model result exist. `--imports-only` checks dependencies, **not** execution.
The `full-models` GitHub Actions workflow builds and tests the actual CPU image.

### What “all 25 working” means here

All 25 **research adapters** can execute their supported role on simulated
SCADA. This does not produce 25 independent validated fault detectors:

- Models have different roles: classification, forecasting, embeddings,
  compression, degradation, RUL, graph analysis, explanations and edge inference.
- m16 executes after upstream RUL estimates are available; it is not a standalone
  detector. A synthetic `fused-rul` fallback is never counted as a model.
- m21 must actually compute SHAP values from a fitted tree to count as having run.
- m24 now trains a small tabular SCADA demo net, converts it to full INT8, and
  invokes the TFLite interpreter. It is **not** a trained image-based MobileNet.
- m20's single-turbine dashboard path uses a self-loop; real wake-coupled farm
  analysis requires fleet data. Vibration windows and wear images are surrogates.
- The sibling architectures still have the limitations recorded in
  `platform/docs/INSPECTION.md` (including simplified Informer/contrastive losses).
- Smoke tests use the simulated training record to verify execution, not held-out
  accuracy. Passing them is not evidence of field performance or plant safety.

The public demo still has no authentication or tenant isolation. Do not attach
plant control connections or upload private SCADA records.
