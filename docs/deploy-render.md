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
