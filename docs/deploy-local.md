# Run all 25 research adapters locally, only when needed

Use a company-owned workstation or internal server instead of paying a cloud
host. Start the Docker container when an operator needs it, then stop it to
release its CPU and RAM. **No Render, GitHub Pages backend, or cloud API key is
required.** The website and Python backend run together in the same container.

This is still a **research/demo system**, not a certified plant monitoring or
control product. All-model readiness verifies execution on simulated SCADA,
not model accuracy on company data.

## Requirements and costs

- Prefer an **Intel/AMD x86-64** computer with at least **8 GB system RAM** and
  **4 GB available to Docker**, at least 2 CPU cores, and about **15 GB free disk**
  for the image/build. These are planning recommendations, not resource guarantees
  for arbitrarily large datasets. No GPU is needed for the six-day demo.
- Linux Docker Engine with **Docker Compose v2.20+**, or Docker Desktop using
  **Linux containers** on Windows/macOS. Confirm `docker compose version` works.
- The supplied image targets `linux/amd64`. Apple Silicon/other ARM hosts require
  emulation; performance and compatibility there have not been verified.
- An internet connection is needed for the **initial build** and later updates
  (Python dependencies, base images and pinned GitHub model sources). Once the
  image is present, ordinary starts and model inference can run without internet.
- There is no cloud-hosting subscription, but hardware, electricity, maintenance
  and possibly Docker Desktop licensing still cost money. Company IT should
  review Docker's current commercial licensing terms before using Desktop.

## 1. Get the prepared code

On the company computer, install Docker and Git, then run:

```bash
git clone --single-branch --branch arena/01a0cc04-wt-pm-lstm-scada-anomaly https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly.git
cd wt-pm-lstm-scada-anomaly
```

The commands below work in a terminal or PowerShell from that repository folder.
`compose.local.yaml` is a **standalone file**, not an override. Do not combine it
with `docker-compose.yml`: the older configuration runs the lean profile and a
separate SCADA watcher.

## 2. Build once (internet required)

```bash
docker compose -f compose.local.yaml build
```

This builds `Dockerfile.full`, installs the full CPU dependencies, and fetches
all 24 original sibling sources at checked revisions. No model stubs are used.
Allow time for the large initial downloads. Do not delete the image if you want
future starts to work offline.

## 3. Start whenever needed

```bash
docker compose -f compose.local.yaml up -d --no-build --pull never --wait --wait-timeout 600
```

Open **http://localhost:8100** on the **same computer**.

The app fits its demo models at startup and executes a complete verification
analysis. Initialization can take a few minutes depending on the computer.
`--wait` waits for `/ready`; full-profile readiness requires all 25 models to
execute without fallback substitution. Closing the browser does **not** stop
the container.

Verify at **http://localhost:8100/health**:

```json
{
  "status": "ok",
  "profile": "full",
  "counts": {"registered": 25, "available": 25, "fitted": 25, "ran": 25}
}
```

Then click **Run full analysis**. To verify via a script without installing
Python on the host:

```bash
docker compose -f compose.local.yaml exec -T dashboard python deployment/check_local.py --analyse
```

## 4. Stop when finished

```bash
docker compose -f compose.local.yaml stop
```

This stops the Python process and releases its runtime CPU/RAM. The container
and image remain on disk for reuse. Repeat the start command next time.
`restart: "no"` prevents automatic restarts when Docker/the computer restarts.
The application cannot monitor assets or send alerts while stopped.

To remove the stopped container/network, but keep the downloaded image:

```bash
docker compose -f compose.local.yaml down
```

**State limitation:** models, analysis results, feedback and alert history live
in memory. Stopping/restarting the container discards them and refits the demo
models on the next start. This setup does not persist trained models or a
production audit database. Save any needed results before stopping, for example
with your browser's Save As on `http://localhost:8100/last`.

## Privacy and company-network access

The port is bound to **127.0.0.1 only**. Other computers on the office network
cannot access it by default. There are no plant/SCADA mounts, watcher, or control
bus enabled in this profile. Startup uses generated data and the model inference
path does not require a cloud service. The Docker bridge is not an outbound
firewall; company IT can enforce network isolation separately.

For a shared internal deployment, ask company IT to put it behind an authenticated
HTTPS reverse proxy and restrict access to the appropriate office network or VPN.
Do **not** simply publish port 8100 to the internet or a broad company network:
the app itself has no login, rate limiting or per-user isolation. Avoid opening
untrusted websites on the same machine while using an unauthenticated local API.

Real company SCADA ingestion, model validation/recalibration, persistent storage,
access control and monitoring are separate deployment work. Do not connect this
demo to turbine controls or treat its simulated results as operational advice.

## Offline transfer to another company machine

On a connected build machine:

```bash
docker save -o wt-pm-full.tar wt-pm-full:local
```

Transfer that archive **and `compose.local.yaml`** through your company's approved
channel. On the destination with Docker installed:

```bash
docker load -i wt-pm-full.tar
docker compose -f compose.local.yaml up -d --no-build --pull never --wait --wait-timeout 600
```

The second machine does not need to build or clone source repositories. Keep the
image archive private as appropriate and outside Git. Use matching CPU support
(`linux/amd64`), and verify `/health` after startup.

## Troubleshooting and updates

```bash
docker compose -f compose.local.yaml ps
docker compose -f compose.local.yaml logs --tail=150 dashboard
```

- **Port 8100 is in use:** stop the other service, or edit only the host port to
  `127.0.0.1:8101:8100`, then browse `http://localhost:8101`.
- **Unhealthy/startup timeout:** inspect logs and `/health` for the actual fit
  error; make sure Docker has 4 GB RAM available. A timeout can leave a container
  running. Use `stop` if abandoning the attempt; do not disable full verification
  merely to make the status green.
- **Exit 137/out of memory:** increase Docker's memory allocation or reduce other
  workloads. Changing the Compose limit alone cannot create RAM on the host.
- **Fewer than 25:** confirm you used this local file, rebuilt the full image,
  and did not combine it with the old Compose file.
- **No image when offline:** build first, or import it using `docker load`.
- **Update:** connect to the internet, pull the desired reviewed code changes,
  stop the service, repeat the build command, then start it again. The image is
  not rebuilt or pulled during normal offline starts.
