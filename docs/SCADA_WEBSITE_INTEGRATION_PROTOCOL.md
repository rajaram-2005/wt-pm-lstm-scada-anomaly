# WT-PM SCADA and website integration protocol

This is the step-by-step integration guide for the **full WT-PM platform** (the
25-adapter platform hosted by this repository), including the PG-BNN physics
cross-check. It describes the supported CSV/JSON boundary, a safe OT/IT
architecture, website consumption, validation, and the work still required for
a real turbine pilot.

> **Read this before connecting plant data.** WT-PM is a research/reference
> implementation, not an OEM-certified SCADA product, protection relay, or
> turbine controller. Start in offline replay and read-only shadow mode. Do not
> connect its advisory output to pitch, brake, trip, derate, converter, or other
> control points. A green `/ready` response proves software startup only; it is
> not evidence of model accuracy, site calibration, safety, or permission to
> operate.

## 1. What is supported today

| Capability | Present in this repository | Important qualification |
|---|---|---|
| Convert a CSV historian export to the canonical 12-channel record | Yes: `wt-pm scada`, `wt-pm field`, `wt-pm watch` | `scada` and `watch` are research/smoke-test paths; see below. |
| Submit JSON rows to the platform | Yes: `POST /scada` | One turbine per request; an installed site bridge must read the vendor SCADA and prepare the rows. |
| Return analysis and candidate `WTPM.*` points | Yes: `/scada` returns `analyse` and `writeback` | The platform only formats results. It does not write to PLCs, OPC UA, Modbus, or MQTT. |
| Serve the built-in dashboard and API | Yes: `wt-pm serve` | The default service bootstraps on **simulated training data** and is a demo, not a site-trained model. |
| Run all 25 adapters | Optional full CPU profile | “All 25 ran” on simulated data does not establish field performance. Some adapters use surrogate inputs or proxy targets. |
| Direct OPC UA / Modbus / MQTT plant connector | **No production connector included** | Build or approve a site-specific, read-only gateway. `mock-bus` and the optional protocol code are emulators, not a live turbine integration. |
| Persist and restore all trained model states for production inference | **Not implemented as a complete platform lifecycle** | A site deployment needs versioned training, artifact storage, restore-on-start, and a rollback plan. |
| Secure public API authentication and user isolation | **Not included in the demo API** | Put it behind a private network and an authenticated TLS reverse proxy/API gateway. |

Two existing commands need particular caution:

- `wt-pm scada --in export.csv` fits research adapters on that supplied export and
  then analyzes the same export. This is useful for checking parsing and the
  output shape only; its scores are not independent validation or operational
  alarms. `--no-fit` now refuses to run because this CLI has no saved-model
  loader.
- `wt-pm watch` also fits on each incoming CSV before analyzing it. It is a
  folder-processing demonstration, **not a production online inference loop**.

The API `/scada` route does not fit on submitted live rows; it uses the models
already fitted at service startup. In the default `wt-pm serve` profile, those
models are bootstrapped using generated demo data. A field-trained model
artifact lifecycle is required before relying on real-turbine scores.

## 2. Recommended network and data-flow design

```text
Turbine sensors / OEM controller
          │
          ▼
Existing SCADA + historian (no WT-PM control access)
          │  read-only account / approved export
          ▼
Site-owned connector in an OT/IT DMZ
(OPC UA, vendor API, MQTT, historian CSV, etc. — site-specific)
          │  normalize units, timestamps, turbine identity and quality
          ▼
Private WT-PM edge/API service: POST /scada
          │
          ├──► approved result store / audit service
          │
          └──► authenticated website backend / dashboard
                    │
                    └──► operators review advisory result
```

Keep the connection **one-way for analytics**: telemetry flows to WT-PM and
advisory results flow to a separate result store or display. Do not route the
website browser to a PLC, and do not route WT-PM recommendations to control
registers. Keep safety interlocks and OEM protection logic independent.

A vendor connector should do only the following: read approved tags, apply
verified unit conversions, group samples by turbine, provide ordered timestamps,
attach the plant/site identity, submit bounded JSON batches, retry safely, and
record delivery status. It should never contain turbine control credentials.

## 3. Prepare the turbine channels

The platform's canonical channel names and engineering units are:

| Canonical field | Unit expected | PG-BNN (`m18-pg-bnn`) input? | Example use |
|---|---|---:|---|
| `wind_speed_ms` | m/s | Yes | Wind operating context |
| `ambient_temp_c` | °C | Yes | Temperature context |
| `rotor_speed_rpm` | rpm | Yes | Mechanical power and operating state |
| `pitch_angle_deg` | degrees | Yes | Blade operating context |
| `yaw_error_deg` | degrees | No | Yaw/fault and tabular features |
| `power_kw` | kW | Yes (prediction target and measured comparator) | Power residuals |
| `main_shaft_torque_knm` | kN·m | Yes | Mechanical-power cross-check |
| `generator_current_a` | A | No | Electrical/tabular models |
| `grid_frequency_hz` | Hz | No | Electrical context |
| `nacelle_temp_c` | °C | No | Thermal context |
| `gearbox_oil_temp_c` | °C | No | Gearbox condition |
| `bearing_vib_rms_mm_s` | mm/s RMS | No | Low-rate vibration indicator |

For the entire 12-channel contract, map all twelve measured channels. The tag
probe reports `ready_for_full_profile` only when it finds each canonical channel
and a time column; `ready_for_pg_bnn` checks the six m18 inputs and a time
column. These flags verify the **header map**, not measurement quality or field
fitness. The timestamp can be supplied as a CSV header such as `TimeStamp`,
`timestamp`, or a plant-specific field mapped to `__time__`.

### Sampling and time rules

1. Match the model's expected 10-minute SCADA cadence, or resample upstream to a
   documented cadence before inference. The default sequence view is 36 samples
   (about six hours at ten-minute cadence); changing the cadence changes the
   physical time span represented by each window.
2. Use one chronologically ordered record per turbine. Timestamps must be unique
   and strictly increasing within that turbine. Sort and deduplicate in the
   historian bridge, not by silently reordering after model features have been
   made.
3. Provide Unix timestamps in seconds (milliseconds, microseconds and
   nanoseconds are also normalized), or timezone-aware ISO-8601 strings such as
   `2026-10-03T12:10:00Z` or `2026-10-03T17:40:00+05:30`. Timezone-naive date
   strings are rejected because interpreting them in the server's local timezone
   can shift the turbine history. Convert site-local historian times to UTC in
   the bridge.
4. Keep values in the units shown above. A tag map changes **names only**; it
   does not convert °F to °C, W to kW, Nm to kN·m, percent to fraction, or
   acceleration to velocity RMS. Verify scale/sign against the OEM historian.
5. Preserve missing values as missing. WT-PM applies causal forward-fill only
   for short gaps and quality-masks unavailable rows; it does not know the
   historian's quality code unless the bridge interprets that code. The bridge
   must report stale/bad source data to the operator.
6. The input called `bearing_vib_rms_mm_s` is a low-rate summary, **not** a
   high-frequency accelerometer waveform. Where a model requires waveforms, the
   current low-rate SCADA path may create an explicitly marked surrogate. Do not
   report that as real waveform validation.
7. For the PG-BNN physics comparison, send **measured shaft torque** if
   available. The CSV adapter can estimate missing torque from power and RPM
   using an efficiency assumption; such estimated torque is not an independent
   physics measurement and should not be used to claim an independent
   physics-vs-data confirmation.

## 4. Map vendor tags and validate headers

Create a plant-specific JSON map. Start from the generated example, then replace
every `WT07.*` tag with the exact tag names from the approved historian export:

```bash
pip install -e platform
wt-pm scada --write-map tag_map.json
```

Example map:

```json
{
  "WT07.WindSpeed": "wind_speed_ms",
  "WT07.AmbTemp": "ambient_temp_c",
  "WT07.RotorRPM": "rotor_speed_rpm",
  "WT07.Pitch": "pitch_angle_deg",
  "WT07.YawErr": "yaw_error_deg",
  "WT07.Pwr_kW": "power_kw",
  "WT07.Torque": "main_shaft_torque_knm",
  "WT07.GenI": "generator_current_a",
  "WT07.GridHz": "grid_frequency_hz",
  "WT07.NacTemp": "nacelle_temp_c",
  "WT07.GbOilTemp": "gearbox_oil_temp_c",
  "WT07.BrgVib": "bearing_vib_rms_mm_s",
  "DateTime": "__time__",
  "Turbine": "__turbine__"
}
```

Probe the actual file before scoring it:

```bash
wt-pm scada --probe historian.csv --map tag_map.json
```

For a full channel map this exits successfully and reports `ready_for_full_profile`
and `time_mapped` as true. `ready_for_pg_bnn` is useful when checking only the
m18 subset. Unknown headers are listed so that the site team can decide whether
they are metadata, a quality field needing bridge logic, or a missing sensor.
The probe does not check units, ranges, timestamps in every row, calibration, or
model accuracy; actual row ingestion performs further checks.

Generate a harmless synthetic sample to test the tag tooling without connecting
to the turbine:

```bash
wt-pm field --sample /tmp/wtpm-sample.csv --rows 48
wt-pm scada --probe /tmp/wtpm-sample.csv --map tag_map.json
```

Do not use the sample's values as operating limits or expected turbine behavior.

## 5. Run the API locally and check readiness

Install the platform in an isolated environment, or use the repository's local
Docker instructions in [`deploy-local.md`](deploy-local.md):

```bash
pip install -e platform
wt-pm inspect
wt-pm serve --port 8100 --days 6
```

The HTTP service binds to `0.0.0.0` for container hosting. Keep the port private;
the application itself does not provide login/authentication. In another
terminal, verify startup:

```bash
curl --fail http://127.0.0.1:8100/ready
curl --fail http://127.0.0.1:8100/health
curl --fail http://127.0.0.1:8100/registry
```

`/ready` is HTTP 503 while models are starting and HTTP 200 after the configured
demo/full-profile initialization. `/health` lists each adapter's availability,
fit status, and last execution. In the optional full CPU profile, the strict
`WTPM_REQUIRE_ALL_MODELS=1` check requires all 25 adapters to execute on the
startup verification record. See [`deploy-local.md`](deploy-local.md) for the
full profile and its dependency requirements.

**Do not read `/ready` as a field qualification.** The service startup and full
profile smoke test use generated demo SCADA. Check whether `m18-pg-bnn` is
available, fitted and actually ran in `/health` and the analysis response; a
lean install may list it as unavailable because PyTorch and the sibling source
are optional.

## 6. Submit one turbine's rows to `POST /scada`

The JSON API accepts a non-empty `rows` array. Send one turbine per request; a
mixed-turbine request is rejected rather than blending two turbines into one
time series. Include the timestamp in UTC and the canonical engineering-unit
values. The example below is deliberately short to show the shape; the default
sequence models need a much longer history than two rows.

```json
{
  "turbine_id": "WT-07",
  "rows": [
    {
      "timestamp": 1791028800,
      "wind_speed_ms": 8.1,
      "ambient_temp_c": 12.4,
      "rotor_speed_rpm": 12.3,
      "pitch_angle_deg": 2.1,
      "yaw_error_deg": -0.4,
      "power_kw": 1150.0,
      "main_shaft_torque_knm": 950.0,
      "generator_current_a": 152.0,
      "grid_frequency_hz": 50.0,
      "nacelle_temp_c": 28.3,
      "gearbox_oil_temp_c": 54.7,
      "bearing_vib_rms_mm_s": 2.1
    },
    {
      "timestamp": 1791029400,
      "wind_speed_ms": 8.2,
      "ambient_temp_c": 12.4,
      "rotor_speed_rpm": 12.4,
      "pitch_angle_deg": 2.1,
      "yaw_error_deg": -0.3,
      "power_kw": 1164.0,
      "main_shaft_torque_knm": 951.0,
      "generator_current_a": 154.0,
      "grid_frequency_hz": 50.0,
      "nacelle_temp_c": 28.4,
      "gearbox_oil_temp_c": 54.8,
      "bearing_vib_rms_mm_s": 2.1
    }
  ]
}
```

Use a real historian-derived timestamp and values in practice; the values above
are illustrative only. The same API can accept vendor names by adding a `map`
object to the request, for example:

```json
{
  "turbine_id": "WT-07",
  "map": {
    "WT07.WindSpeed": "wind_speed_ms",
    "WT07.Pwr_kW": "power_kw",
    "DateTime": "__time__"
  },
  "rows": [
    {"DateTime": 1791028800, "WT07.WindSpeed": 8.1, "WT07.Pwr_kW": 1150.0}
  ]
}
```

Submit a saved JSON request from the **private service network**:

```bash
curl --fail-with-body -X POST http://127.0.0.1:8100/scada \
  -H 'Content-Type: application/json' \
  --data-binary @request.json
```

A successful response has this shape:

```json
{
  "analyse": { "turbine_id": "WT-07", "what": {}, "where": {}, "physics": {} },
  "writeback": {
    "schema": "wt-pm.scada.writeback.v1",
    "turbine_id": "WT-07",
    "timestamp": 1791029400,
    "tags": { "WTPM.ALARM": 0, "WTPM.FAULT": "healthy" },
    "point_list": []
  }
}
```

The response contains many more fields; consult `/last` for the most recent
analysis result. The API rejects malformed/mixed-turbine/unordered rows with
HTTP 400, input that cannot be analyzed with HTTP 422, a body larger than 8 MiB
with HTTP 413, and requests during startup with HTTP 503. A successful HTTP 200
means the software returned a result, not that it is correct for a particular
wind turbine.

## 7. Show results on a website

### Same-origin dashboard or private web application

The built-in dashboard is served from `/` on the same API origin. For a
separate website, the recommended pattern is a server-side reverse proxy so the
browser talks to its own site origin and the proxy forwards only approved API
routes to the private WT-PM service. For example:

```text
Browser: https://ops.example.com/api/scada
          │ authenticated HTTPS reverse proxy
          ▼
Private service: http://wtpm-internal:8100/scada
```

A front-end can then call the same-origin route (the reverse proxy must map
`/api/scada` to the internal `/scada` endpoint):

```javascript
async function analyseTurbine(turbineId, rows) {
  const response = await fetch("/api/scada", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ turbine_id: turbineId, rows })
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `WT-PM HTTP ${response.status}`);

  const result = body.analyse;
  document.querySelector("#turbine").textContent = result.turbine_id || turbineId;
  document.querySelector("#risk").textContent = String(result.risk_score ?? "—");
  document.querySelector("#fault").textContent = result.what?.fault || "unknown";
  document.querySelector("#action").textContent = result.action?.action || "review";
  // Show that these are advisory candidate tags; this does not write to SCADA.
  document.querySelector("#candidate-tags").textContent =
    JSON.stringify(body.writeback?.tags || {}, null, 2);
  return body;
}
```

Use `textContent` for data-derived UI text; do not interpolate tag values or
operator notes into `innerHTML`. Keep identity tokens and historian credentials
on the server side, never in website JavaScript.

### GitHub Pages or another static website

GitHub Pages can host static HTML/JavaScript but cannot run the Python model,
receive private historian traffic safely, store secrets, or authenticate users
by itself. A static page may call a separately secured backend, but for a private
SCADA deployment the backend should be on the site's private network and the
website should reach it through an authenticated gateway. Do not send turbine
history to the public Render research demo or expose an unauthenticated API to
the internet.

If a cross-origin browser call is unavoidable, configure exact origins, for
example:

```bash
WTPM_CORS_ORIGINS=https://ops.example.com,https://dashboard.example.com
```

The service will only return CORS permission for those exact origins; wildcard
`*` is not accepted. CORS is **not authentication** and does not stop scripts,
servers, command-line clients, or direct network users. The current API allows
only the `Content-Type` request header for cross-origin browser requests. Prefer
a same-origin authenticated proxy rather than placing credentials in a browser.

## 8. Candidate `WTPM.*` tags and their meaning

The result formatter emits advisory fields such as:

- `WTPM.ALARM`, `WTPM.SCORE`, `WTPM.RISK`, `WTPM.HEALTH`
- `WTPM.FAULT`, `WTPM.FAULT_CONF`, `WTPM.SUBSYSTEM`
- `WTPM.RUL_H`, `WTPM.RUL_UNC_H`, `WTPM.ACTION`, `WTPM.SAFETY`, `WTPM.TRUST`
- `WTPM.NARRATIVE`, `WTPM.TURBINE`

`WTPM.ACTION` and `WTPM.SAFETY` are software recommendations/assessments, not
validated turbine commands or certified safe-state determinations. Do not map
them to OEM control tags. If an organization later approves storing analytics
back into a historian, create a separate, namespaced, low-privilege result area;
include source timestamp, model version, quality, and audit identity. Keep it
read-only to the controller and review stale-result handling with the site
engineer. There is no automatic SCADA writeback in this repository.

## 9. Required model and data validation before a field pilot

Use the following staged protocol; do not skip directly from a synthetic demo to
plant operation.

1. **Offline schema replay.** Validate one turbine's timestamp ordering, unit
   conversions, tag mapping, missingness and expected ranges with the site
   historian engineer. Record map/version and source query.
2. **Create a field training dataset.** Obtain site-approved history spanning
   relevant operating regimes and known healthy periods. Use a separate,
   trusted maintenance/work-order source for fault labels. Supervised models
   without verified examples must be disabled or reported unavailable; do not
   treat unknown as healthy.
3. **Split by time and asset.** Keep train, calibration, validation and final
   test intervals chronologically disjoint; where possible hold out whole
   turbines/sites. Never randomly split overlapping windows. Fit scalers and
   thresholds only on the training/calibration portion allowed by the
   evaluation protocol.
4. **Select only applicable adapters.** The 25 entries have different roles;
   they are not 25 equivalent independent fault detectors. Check each adapter's
   required data, framework, training status and limitations. The m18 PG-BNN
   requires measured power plus wind, ambient temperature, RPM, pitch and torque;
   its MC epistemic spread and approximate band are not calibrated coverage.
5. **Set site thresholds.** Agree an alert budget with operations. Measure
   false alarms per turbine-day, event recall, detection delay, missed fault
   families, calibration and data-quality behavior on held-out real history.
   Compare with work orders and operator review, not only simulation metrics.
6. **Persist and version artifacts.** Before a restartable field service is
   accepted, persist model weights, fitted scalers, thresholds, input schema,
   tag-map version, training window, code/source revision, dependency versions,
   and validation report. Add integrity checks, restore-on-start tests, and a
   tested rollback to the last approved model. This full save/restore lifecycle
   is an engineering gap in the current 25-adapter demo.
7. **Run shadow mode.** Deploy in a private network with read-only historian
   access. Show alerts to trained reviewers without automatically changing
   operation. Track missing/stale data, sensor quality, drift, latency, alerts,
   reviewer decisions and maintenance outcomes for an agreed period and across
   operating regimes.
8. **Review and approve.** The turbine owner, OT security, OEM/controls engineer,
   reliability engineer and safety authority must sign off the use case, limits,
   failover, retention, monitoring and rollback. Keep all protection functions
   independent. If the service or input becomes unavailable, show “data/model
   unavailable”; never silently display the last old result as current.

The project fidelity ladder is [`fidelity_ladder.md`](fidelity_ladder.md): this
repository's published baseline is simulation (rung 1), not a field-tested
system. Passing platform tests or running all 25 adapters on synthetic SCADA
cannot advance that rung.

## 10. Security, privacy, and service operation

- Put the API in a private OT/IT DMZ or approved edge host. Do not expose port
  8100 directly to the public internet or the turbine control network.
- Use a read-only historian account, least-privilege firewall rules, TLS at the
  gateway, authentication/authorization, rate limits, request logging, and
  monitored backups. The WT-PM HTTP demo server itself has no authentication,
  persistent database, user isolation, or production-grade TLS.
- Do not use the public Render demo for real/private plant data. Use an
  organization-approved private deployment and retention policy.
- Keep network outage handling explicit: buffer bounded records, deduplicate
  retries by turbine/timestamp, monitor queue age, and alert when data is stale.
  Do not queue unbounded data in the browser.
- Store results outside WT-PM if they must survive a process restart; demo
  analysis state, feedback and alerts are in memory.
- Restrict website access by role; separate operator, reliability and
  administrator permissions. Record who acknowledged an advisory and when.
- Review output before display and clearly mark model version, source timestamp,
  uncertainty caveat, model availability, sensor trust and stale-data state.

## 11. Troubleshooting

| Symptom | Likely cause | Corrective action |
|---|---|---|
| `no SCADA columns mapped` | Vendor headers not recognized | Add exact source tags to `tag_map.json` and probe again. |
| `ready_for_full_profile` is false | Missing canonical headers or time tag | Map all 12 channels and a time column; inspect `missing_canonical`. |
| `ready_for_pg_bnn` is false | One or more of m18's six required channel headers absent | Map wind, ambient temperature, rotor speed, pitch, measured power and torque. |
| Timestamp parsing error | Invalid or timezone-naive time values | Convert to UTC epoch seconds or timezone-aware ISO-8601 upstream. |
| Timestamp order error | Duplicated/out-of-order samples | Sort/deduplicate within each turbine before submit. |
| “one turbine” error | A payload mixes turbine IDs | Send one request per turbine, or split by turbine in the gateway. |
| HTTP 413 | JSON body exceeds 8 MiB | Send smaller time windows/batches; keep a bounded queue. |
| HTTP 422 | Input passed parsing but could not form valid analysis features | Check numeric values, mapped units, finite channels and sequence length; examine private service logs. |
| `m18-pg-bnn` unavailable | Lean dependency profile, missing PyTorch, or sibling source not installed | Use the reviewed full profile and verify `/health`; do not infer availability from the catalog alone. |
| Browser CORS failure | Website origin is not allowlisted | Prefer the same-origin proxy or add the exact origin to `WTPM_CORS_ORIGINS`; still add authentication. |
| Website shows a result after data stops | UI retained the previous response | Add timestamp/age display and a stale-data warning; do not treat cached output as live. |

## 12. Deployment sign-off checklist

- [ ] Site OT/IT and turbine owner approve the data path and read-only access.
- [ ] Per-turbine tags, units, scale, sign, timestamp timezone and 10-minute
      cadence are verified against historian documentation.
- [ ] `wt-pm scada --probe ...` reports `ready_for_full_profile` (or the
      narrower approved subset) and all row-level checks pass.
- [ ] Field-trained, versioned model artifacts and restore/rollback tests exist;
      the demo/same-file-fit path is not used for live scoring.
- [ ] Held-out real-turbine metrics and operator shadow review meet an agreed
      false-alarm budget; unvalidated adapters are disabled or labelled.
- [ ] Website traffic uses HTTPS, authentication, authorization, request limits,
      safe output rendering and a private backend route.
- [ ] `WTPM.*` results remain advisory and cannot reach a control register.
- [ ] Data outage, stale result, model error, restart, rollback, audit and
      incident-response procedures have been tested.
- [ ] Operations and safety authorities approve the final pilot scope.

**Bottom line:** this repository provides a useful, testable CSV/JSON interface
and dashboard to build a turbine-specific analytics pilot. It does not offer a
universal plug-and-play connection to every SCADA vendor or website. A verified
vendor bridge, real-site training/validation, secure hosting, persistent model
artifacts and formal operational approval are necessary before plant use.
