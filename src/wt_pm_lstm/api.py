"""HTTP API: the cross-model interface, in the smallest honest form.

The point of this module is not "we have a REST endpoint". It is that model 13's
*output contract* is machine-readable and identical to the one a sibling model
(fault classifier, RUL, digital twin) should speak, so the fleet-level layer can
compose them without touching Python:

    POST /score   {channel: value} or {"timeline": {...}}  ->  anomaly records
    GET  /schema                                            ->  data contract
    GET  /health                                            ->  liveness + model id

Served with :mod:`http.server` from the standard library: the reference engine
must be deployable on the same machine as the DAQ, where a research-grade web
stack is not always available. Payloads are validated against the schema and
missing channels are refused rather than silently imputed.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple

import numpy as np

from wt_pm_lstm.config import RunConfig, config_fingerprint
from wt_pm_lstm.dataio import records_to_json
from wt_pm_lstm.pipeline import generate_record, fit_detector
from wt_pm_lstm.registry import model_id
from wt_pm_lstm.schema import (
    SCHEMA_VERSION,
    ANOMALY_SCHEMA_VERSION,
    QUALITY_MISSING,
    QUALITY_OK,
    SCHEMA_VERSION,
    TurbineTimeline,
    canonical_channels,
    describe_schema,
    get_channel,
)

MAX_BODY_BYTES = 8 * 1024 * 1024


def _data_config_copy(dcfg: Any, seed: int, n_steps: Optional[int] = None) -> Any:
    """A copy of the data config with a different seed and (optionally) length.

    The length matters: a fault injected at 55% of a short batch is invisible if
    the generator produces a multi-day record and only its tail is returned.
    """
    import copy

    out = copy.deepcopy(dcfg)
    out.seed = int(seed)
    if n_steps is not None:
        out.n_days = float(n_steps) * float(out.sample_minutes) / (24.0 * 60.0)
    return out


class ScoringService:
    """Holds a fitted detector and answers contract-shaped requests."""

    def __init__(self, cfg: Optional[RunConfig] = None, detector: Any = None, verbose: bool = False):
        self.cfg = cfg or RunConfig()
        if detector is None:
            n_steps = int(self.cfg.data.n_days * 24 * 60 / self.cfg.data.sample_minutes)
            from wt_pm_lstm.windows import split_indices

            plan = split_indices(
                n_steps, self.cfg.data.healthy_fraction, self.cfg.data.calibration_fraction, None, self.cfg.model.window
            )
            timeline = generate_record(self.cfg, plan)
            detector, _, _ = fit_detector(self.cfg, timeline, verbose=verbose)
        self.detector = detector
        self.model_id = model_id(self.cfg)
        self.lock = threading.Lock()

    # -- handlers ---------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "model_id": self.model_id,
            "engine": self.cfg.engine,
            "cell": self.cfg.model.cell,
            "schema_version": SCHEMA_VERSION,
            "anomaly_schema_version": ANOMALY_SCHEMA_VERSION,
            "threshold": float(self.detector.threshold.value),
            "scored_channels": [
                name for name, keep in zip(self.detector.channel_names, self.detector.score_mask) if keep
            ],
        }

    def schema(self) -> Dict[str, Any]:
        payload = describe_schema()
        payload["engine"] = self.cfg.engine
        return payload

    def sample(self, fault: str = "", n_samples: int = 432, seed: int = 7) -> Dict[str, Any]:
        """Return a contract-shaped sample batch straight from the simulator.

        Exists so the contract can be demonstrated end to end without a real
        SCADA export: the caller receives samples *and* the injected ground
        truth, and can check the detector's answer against it.
        """
        from wt_pm_lstm.simulate import FAULT_KINDS, SENSOR_FAULT_TARGETS, FaultSpec, simulate_timeline

        dcfg = self.cfg.data
        n = int(max(self.cfg.model.window + self.cfg.detect.trend_window + 12, n_samples))
        specs = []
        if fault:
            if fault not in FAULT_KINDS:
                raise ValueError(f"unknown fault {fault!r}; expected one of {list(FAULT_KINDS)}")
            onset = int(n * 0.55)
            specs.append(
                FaultSpec(
                    kind=fault,
                    onset_step=onset,
                    duration_steps=max(24, n - onset - 2),
                    magnitude=1.0,
                    channel=str(SENSOR_FAULT_TARGETS.get(fault, ("",))[0]) if fault in SENSOR_FAULT_TARGETS else "",
                )
            )
        timeline = simulate_timeline(_data_config_copy(dcfg, seed=seed, n_steps=n), fault_specs=specs)
        keep = slice(timeline.n_steps - n_samples, timeline.n_steps)
        samples = []
        for i in range(timeline.n_steps)[keep]:
            row = {"timestamp": int(timeline.timestamps[i])}
            for j, name in enumerate(timeline.channel_names):
                row[name] = None if not timeline.mask[i, j] else round(float(timeline.values[i, j]), 6)
            samples.append(row)
        labels = None
        if timeline.fault_label is not None:
            labels = timeline.fault_label[keep].astype(int).tolist()
        return {
            "schema_version": SCHEMA_VERSION,
            "turbine_id": "WT-SIM",
            "injected_fault": fault or None,
            "fault_starts_at_sample": (int(n_samples * 0.55) if fault else None),
            "note": (
                "samples are produced by the physics-informed simulator and are meant to be "
                "posted back to /score; they are not field data"
            ),
            "samples": samples,
            "fault_label": labels,
        }

    def score(self, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """Validate and score a batch of samples. Returns (http_status, body)."""
        try:
            timeline = self._timeline_from_payload(payload)
        except ValueError as exc:
            return 400, {"error": str(exc), "schema_version": SCHEMA_VERSION}
        with self.lock:  # the detector holds per-call state in components
            result = self.detector.score_timeline(timeline)
        return 200, {
            "anomaly_schema_version": ANOMALY_SCHEMA_VERSION,
            "model_id": self.model_id,
            "turbine_id": timeline.turbine_id,
            "n_samples": int(timeline.n_steps),
            "threshold": float(result.threshold),
            "n_alarm_events": len(result.records),
            "summary": result.summary(),
            "records": records_to_json(result.records),
        }

    # -- payload -> timeline ----------------------------------------------
    def _timeline_from_payload(self, payload: Dict[str, Any]) -> TurbineTimeline:
        if not isinstance(payload, dict):
            raise ValueError("body must be a JSON object")
        samples = payload.get("samples")
        if samples is None and "timeline" in payload:
            samples = payload["timeline"].get("samples")
        if samples is None and "channels" in payload:
            # A single sample given as {channel: value}.
            samples = [payload["channels"]]
        if not samples:
            raise ValueError("provide 'samples': [{'timestamp': int, <channel>: value}, ...]")

        channel_names = tuple(self.detector.channel_names)
        n = len(samples)
        values = np.zeros((n, len(channel_names)))
        mask = np.ones((n, len(channel_names)), dtype=bool)
        quality = np.full((n, len(channel_names)), QUALITY_OK, dtype=np.uint8)
        timestamps = np.zeros(n, dtype=np.int64)

        for i, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise ValueError(f"sample {i} is not an object")
            unknown = set(sample) - set(channel_names) - {"timestamp", "turbine_id"}
            if unknown:
                raise ValueError(f"sample {i}: unknown channels {sorted(unknown)}")
            missing = [c for c in channel_names if c not in sample]
            if missing:
                raise ValueError(
                    f"sample {i}: missing channels {missing} "
                    "(the contract has no default for a physical measurement)"
                )
            ts = sample.get("timestamp")
            if ts is None:
                raise ValueError(f"sample {i}: 'timestamp' (unix seconds) is required")
            timestamps[i] = int(ts)
            for j, name in enumerate(channel_names):
                raw = sample[name]
                if raw is None or (isinstance(raw, str) and raw.strip() == ""):
                    values[i, j] = 0.0
                    mask[i, j] = False
                    quality[i, j] = QUALITY_MISSING
                    continue
                value = float(raw)
                if not np.isfinite(value):
                    mask[i, j] = False
                    quality[i, j] = QUALITY_MISSING
                    continue
                values[i, j] = value
                if not get_channel(name).in_range(value):
                    quality[i, j] = 3  # out_of_range, kept not deleted
        min_samples = max(self.cfg.model.window + 1, 2)
        if n < min_samples:
            raise ValueError(
                f"{n} samples is below the minimum of {min_samples} "
                f"(a {self.cfg.model.window}-sample window plus one)"
            )
        return TurbineTimeline(
            turbine_id=str(payload.get("turbine_id", "WT-001")),
            site_id=str(payload.get("site_id", "unknown-site")),
            channel_names=channel_names,
            values=values,
            timestamps=timestamps,
            mask=mask,
            quality=quality,
            meta={"source": "api"},
        )


class _Handler(BaseHTTPRequestHandler):
    service: ScoringService
    server_version = "wtpm-scada/1.0"

    def _send(self, status: int, body: Dict[str, Any]) -> None:
        payload = json.dumps(body, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:  # noqa: N802 (http.server API)
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0].rstrip("/") or "/"
        if path == "/health":
            self._send(200, self.service.health())
        elif path == "/schema":
            self._send(200, self.service.schema())
        elif path == "/sample":
            query = dict(
                pair.split("=", 1) for pair in (self.path.split("?", 1)[1].split("&") if "?" in self.path else []) if "=" in pair
            )
            try:
                payload = self.service.sample(
                    fault=query.get("fault", ""),
                    n_samples=int(query.get("n", 432)),
                    seed=int(query.get("seed", 7)),
                )
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
                return
            self._send(200, payload)
        elif path == "/":
            # A tiny console, so the contract can be exercised from a browser
            # without a build step. It uses only relative URLs (the browser is
            # not the host running this process).
            body = _CONSOLE_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._send(
                404,
                {"error": f"unknown path {path!r}", "paths": ["/", "/health", "/schema", "/sample", "/score"]},
            )

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        if path != "/score":
            self._send(404, {"error": f"unknown path {path!r}"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send(413, {"error": f"body length must be 1..{MAX_BODY_BYTES} bytes"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError as exc:
            self._send(400, {"error": f"invalid JSON: {exc}"})
            return
        status, body = self.service.score(payload)
        self._send(status, body)

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter access log
        print(f"[api] {self.address_string()} {fmt % args}")


def make_server(host: str, port: int, service: ScoringService) -> ThreadingHTTPServer:
    handler = type("_BoundHandler", (_Handler,), {"service": service})
    return ThreadingHTTPServer((host, port), handler)


def serve(host: str = "0.0.0.0", port: int = 8000, cfg: Optional[RunConfig] = None, detector: Any = None) -> int:
    """Fit (or load) a detector and serve the contract until interrupted."""
    service = ScoringService(cfg=cfg, detector=detector)
    server = make_server(host, port, service)
    print(f"wtpm api listening on http://{host}:{port}")
    print(f"  model_id {service.model_id}")
    print(f"  POST /score   {{'turbine_id':..., 'samples':[{{'timestamp':..., <channel>: value}}, ...]}}")
    print("  GET  /schema  GET /health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.server_close()
    return 0





_CONSOLE_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>WT-PM model 13 — scoring console</title>
<style>
 body{font:14px/1.5 ui-sans-serif,system-ui,sans-serif;margin:0;padding:32px;background:#0f1720;color:#e6edf3}
 h1{font-size:18px;margin:0 0 4px} p.sub{color:#8b98a5;margin:0 0 20px}
 code,pre{font-family:ui-monospace,Menlo,monospace}
 button{background:#2f81f7;color:#fff;border:0;border-radius:6px;padding:8px 14px;font-size:13px;cursor:pointer;margin:0 6px 8px 0}
 button:hover{background:#1f6feb}
 pre{background:#161b22;border:1px solid #26303b;border-radius:8px;padding:14px;overflow:auto;max-height:50vh}
 .badge{background:#1a7f37;padding:2px 8px;border-radius:999px;font-size:12px}
 .badge.bad{background:#8b1a1a} .badge.warn{background:#9e6a03}
 table{border-collapse:collapse;margin-bottom:16px} td,th{border-bottom:1px solid #26303b;padding:4px 10px;text-align:left;font-size:13px}
 th{color:#8b98a5;font-weight:500}
</style></head><body>
<h1>WT-PM model 13 &middot; SCADA anomaly scoring console</h1>
<p class="sub">LSTM sequence anomaly detection &mdash; input contract <code>wt-pm.scada.v1</code>, output contract <code>wt-pm.anomaly.v1</code></p>
<table><tr><th>model</th><td id="mid">&hellip;</td></tr><tr><th>threshold</th><td id="thr">&hellip;</td></tr>
<tr><th>scored channels</th><td id="ch">&hellip;</td></tr></table>
<div id="buttons"></div>
<div><span id="state" class="badge">ready</span></div>
<pre id="out">Choose a scenario above. Samples come from the physics-informed simulator via GET /sample; the detector's answer comes from POST /score.</pre>
<script>
const out = document.getElementById('out'), state = document.getElementById('state');
const scenarios = [['healthy',''],['gearbox thermal','gearbox_thermal'],['bearing wear','bearing_wear'],
 ['pitch misalignment','pitch_misalignment'],['yaw error','yaw_error'],['converter fault','converter_fault'],
 ['sensor freeze','sensor_freeze'],['sensor drift','sensor_drift']];
fetch('/health').then(r => r.json()).then(h => {
  document.getElementById('mid').textContent = h.model_id;
  document.getElementById('thr').textContent = h.threshold.toFixed(3);
  document.getElementById('ch').textContent = h.scored_channels.join(', ');
}).catch(e => { out.textContent = 'health failed: ' + e; });
const bar = document.getElementById('buttons');
for (const [label, fault] of scenarios) {
  const b = document.createElement('button');
  b.textContent = label;
  b.onclick = () => run(fault, label);
  bar.appendChild(b);
}
async function run(fault, label) {
  state.className = 'badge warn'; state.textContent = 'scoring…';
  try {
    const batch = await (await fetch('/sample?n=432' + (fault ? '&fault=' + fault : ''))).json();
    const res = await fetch('/score', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({turbine_id: batch.turbine_id, samples: batch.samples})});
    const body = await res.json();
    const recs = body.records || [];
    const detected = recs.length > 0;
    state.className = 'badge' + (fault ? (detected ? '' : ' bad') : (detected ? ' bad' : ''));
    state.textContent = fault ? (detected ? 'alarm raised — fault detected' : 'no alarm — fault missed')
                              : (detected ? 'alarm raised on healthy data (false alarm)' : 'quiet on healthy data');
    out.textContent = JSON.stringify({
      scenario: label,
      injected_fault: batch.injected_fault || null,
      samples: batch.samples.length,
      fault_starts_at_sample: batch.fault_starts_at_sample,
      http_status: res.status,
      n_alarm_events: recs.length,
      alarms: recs.map(r => ({timestamp: r.timestamp, action: r.recommended_action,
                              channels: Object.keys(r.attribution || {}), why: r.explanation})),
      note: batch.injected_fault
        ? 'the simulator injected the fault at the sample index above; alarms after it are detections, alarms before it are false positives'
        : 'healthy record: any alarm is a false alarm'
    }, null, 2);
  } catch (e) { state.className = 'badge bad'; state.textContent = 'error'; out.textContent = String(e); }
}
</script></body></html>
"""


__all__ = ["ScoringService", "make_server", "serve"]
