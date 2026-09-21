"""HTTP API + dashboard for the unified platform (stdlib only, like model 05).

  GET  /health      liveness + per-adapter availability
  GET  /registry    the full model registry table
  GET  /plan        edge deployment plan
  POST /analyse     {"days": float, "seed": int} -> full 7-question record
  GET  /            dashboard
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

import numpy as np

_STATE: Dict[str, Any] = {"orchestrator": None, "batch": None, "last": None,
                          "fitted": False, "lock": threading.Lock()}


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _bootstrap(days: float, seed: int) -> None:
    from wtpm_platform.cli import _make_batch
    from wtpm_platform.contracts import OperatingContext
    from wtpm_platform.orchestrator import Orchestrator

    orch = Orchestrator()
    batch = orch.prepare(_make_batch(days, seed=seed))
    ctx = OperatingContext(mode="research", has_labels=True, has_vibration_waveform=True)
    orch.fit(batch, ctx, verbose=True)
    _STATE.update(orchestrator=orch, batch=batch, ctx=ctx, fitted=True)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: Any, ctype: str = "application/json") -> None:
        data = body if isinstance(body, bytes) else json.dumps(
            body, indent=2, default=_json_default).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        orch = _STATE["orchestrator"]
        if self.path == "/health":
            self._send(200, {
                "status": "ok" if _STATE["fitted"] else "starting",
                "models": None if orch is None else orch.registry.health_report(),
            })
        elif self.path == "/registry":
            if orch is None:
                return self._send(503, {"error": "starting"})
            self._send(200, [
                {"model_id": s.model_id, "repository": s.repository,
                 "task": s.task.value, "input": list(s.input_requirements),
                 "output": list(s.output_schema),
                 "deps": list(s.resource_requirements),
                 "latency_ms": s.typical_latency_ms,
                 "deployment": [d.value for d in s.deployment_targets],
                 "fallback": s.fallback, "notes": s.notes}
                for s in orch.registry.specs()])
        elif self.path == "/plan":
            if orch is None:
                return self._send(503, {"error": "starting"})
            self._send(200, orch.edge.deployment_plan())
        elif self.path == "/last":
            self._send(200, _STATE["last"] or {"info": "POST /analyse first"})
        elif self.path == "/":
            self._send(200, DASHBOARD.encode(), "text/html")
        else:
            self._send(404, {"error": "unknown path"})

    def do_POST(self):
        if self.path != "/analyse":
            return self._send(404, {"error": "unknown path"})
        if not _STATE["fitted"]:
            return self._send(503, {"error": "models still fitting, retry shortly"})
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"error": "invalid JSON"})
        with _STATE["lock"]:
            orch, ctx = _STATE["orchestrator"], _STATE["ctx"]
            batch = _STATE["batch"]
            if payload.get("seed") is not None:
                from wtpm_platform.cli import _make_batch
                batch = orch.prepare(_make_batch(
                    float(payload.get("days", 10.0)), seed=int(payload["seed"]),
                    turbine_id=payload.get("turbine_id", "WT-live")))
            result = orch.analyse(batch, ctx)
            _STATE["last"] = result
        self._send(200, result)


DASHBOARD = """<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>WT-PM Unified Platform</title><style>
body{font-family:system-ui;background:#0b1020;color:#dce4f5;margin:0;padding:24px}
h1{font-size:22px} .card{background:#121a30;border:1px solid #24304f;border-radius:12px;
padding:16px;margin:12px 0} button{background:#4cc9f0;border:0;border-radius:8px;
padding:8px 18px;font-weight:700;cursor:pointer} pre{white-space:pre-wrap;font-size:12px;
color:#8ea0c4;max-height:420px;overflow:auto} .big{font-size:30px;font-weight:800}
.row{display:flex;gap:14px;flex-wrap:wrap} .kv{background:#0f1628;border-radius:8px;
padding:10px 14px} .kv b{color:#4cc9f0;display:block;font-size:11px;text-transform:uppercase}
</style></head><body>
<h1>WT-PM Unified Platform — 25 models, one orchestrator</h1>
<div class=card><button onclick="run()">Run full analysis</button>
 <span id=status></span>
<div class=row id=summary></div></div>
<div class=card><b>Model health</b><pre id=health>loading…</pre></div>
<div class=card><b>Full record</b><pre id=out>—</pre></div>
<script>
async function health(){const r=await fetch('/health');document.getElementById('health').textContent=JSON.stringify(await r.json(),null,1)}
async function run(){
 document.getElementById('status').textContent=' running…';
 const r=await fetch('/analyse',{method:'POST',body:'{}'});const d=await r.json();
 document.getElementById('status').textContent=' done in '+d.pipeline_ms+' ms';
 const s=document.getElementById('summary');
 const kv=(k,v)=>`<div class=kv><b>${k}</b><span class=big>${v}</span></div>`;
 s.innerHTML=kv('fault',d.what.fault)+kv('subsystem',d.where.subsystem)
  +kv('risk',d.risk_score)+kv('RUL h',d.rul.hours??'—')
  +kv('action',d.action.action)+kv('safety',d.safety.decision);
 document.getElementById('out').textContent=JSON.stringify(d,null,1);}
health();
</script></body></html>"""


def serve(port: int = 8100, days: float = 10.0, seed: int = 7) -> None:
    threading.Thread(target=_bootstrap, args=(days, seed), daemon=True).start()
    httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"[serve] WT-PM platform API on :{port} (models fitting in background)")
    httpd.serve_forever()
