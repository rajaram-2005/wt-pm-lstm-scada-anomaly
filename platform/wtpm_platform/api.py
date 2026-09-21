"""HTTP API + operator dashboard for the unified platform (stdlib only).

  GET  /health /registry /plan /last /audit /alerts
  POST /analyse  /what-if  /alerts/ack  /feedback
  GET  /         dashboard
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict
from urllib.parse import urlparse

import numpy as np

_STATE: Dict[str, Any] = {"orchestrator": None, "batch": None, "last": None,
                          "fitted": False, "lock": threading.Lock(),
                          "feedback": []}


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
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        orch = _STATE["orchestrator"]
        if path == "/health":
            self._send(200, {
                "status": "ok" if _STATE["fitted"] else "starting",
                "models": None if orch is None else orch.registry.health_report(),
            })
        elif path == "/registry":
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
        elif path == "/plan":
            if orch is None:
                return self._send(503, {"error": "starting"})
            self._send(200, orch.edge.deployment_plan())
        elif path == "/last":
            self._send(200, _STATE["last"] or {"info": "POST /analyse first"})
        elif path == "/audit":
            if orch is None:
                return self._send(503, {"error": "starting"})
            self._send(200, orch.audit.all())
        elif path == "/alerts":
            if orch is None:
                return self._send(503, {"error": "starting"})
            self._send(200, orch.alerts.snapshot())
        elif path == "/scada/tags":
            last = _STATE.get("last")
            if not last:
                return self._send(200, {"info": "POST /scada or /analyse first"})
            from wtpm_platform.scada import emit_scada_tags
            self._send(200, emit_scada_tags(last, last.get("turbine_id", "WT")))
        elif path == "/":
            self._send(200, DASHBOARD.encode(), "text/html")
        else:
            self._send(404, {"error": "unknown path"})

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length) or b"{}")

    def do_POST(self):
        path = urlparse(self.path).path
        if not _STATE["fitted"]:
            return self._send(503, {"error": "models still fitting, retry shortly"})
        try:
            payload = self._read_json()
        except json.JSONDecodeError:
            return self._send(400, {"error": "invalid JSON"})

        with _STATE["lock"]:
            orch, ctx = _STATE["orchestrator"], _STATE["ctx"]
            batch = _STATE["batch"]

            if path == "/analyse":
                if payload.get("seed") is not None:
                    from wtpm_platform.cli import _make_batch
                    batch = orch.prepare(_make_batch(
                        float(payload.get("days", 10.0)), seed=int(payload["seed"]),
                        turbine_id=payload.get("turbine_id", "WT-live")))
                    _STATE["batch"] = batch
                if payload.get("fusion_method"):
                    orch.fusion.cfg.method = str(payload["fusion_method"])
                result = orch.analyse(batch, ctx)
                _STATE["last"] = result
                return self._send(200, result)

            if path == "/what-if":
                overrides = payload.get("overrides") or {}
                derate = float(payload.get("derate_pct", 0) or 0)
                if derate:
                    rpm = float(np.mean(batch.channel("rotor_speed_rpm")))
                    overrides["rotor_speed_rpm"] = rpm * (1 - derate / 100)
                res = orch.twin.what_if(batch, {k: float(v) for k, v in overrides.items()})
                orch.audit.record("what-if", {"overrides": overrides})
                return self._send(200, res)

            if path == "/alerts/ack":
                ok = orch.alerts.ack(str(payload.get("alert_id", "")))
                orch.audit.record("ack", {"alert_id": payload.get("alert_id"), "ok": ok})
                return self._send(200, {"ok": ok})

            if path == "/scada":
                from wtpm_platform.contracts import OperatingContext
                from wtpm_platform.scada import emit_scada_tags, ingest_scada_rows
                rows = payload.get("rows") or payload.get("samples")
                if not rows:
                    return self._send(400, {"error": "JSON body needs 'rows': [{plant tags...}]"})
                live = ingest_scada_rows(
                    rows,
                    turbine_id=str(payload.get("turbine_id", "WT-SCADA")),
                    tag_map=payload.get("map") or {},
                )
                live = orch.prepare(live)
                result = orch.analyse(live, OperatingContext(mode="production", has_labels=False,
                                                             has_vibration_waveform=True))
                _STATE["last"] = result
                tags = emit_scada_tags(result, live.turbine_id)
                orch.audit.record("scada", {"turbine_id": live.turbine_id,
                                            "n": live.n_steps})
                return self._send(200, {"analyse": result, "writeback": tags})

            if path == "/feedback":
                row = {"fault": payload.get("fault"),
                       "correct": bool(payload.get("correct", True)),
                       "note": payload.get("note", "")}
                _STATE["feedback"].append(row)
                orch.audit.record("operator_feedback", row)
                return self._send(200, {"stored": len(_STATE["feedback"])})

        self._send(404, {"error": "unknown path"})


DASHBOARD = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WT-PM Command Center</title>
<style>
:root{--bg:#070b16;--card:#10182c;--line:#24304f;--txt:#dce4f5;--mut:#8ea0c4;--acc:#4cc9f0;--ok:#3ddc97;--warn:#f4c95d;--bad:#ff6b6b}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--txt);font-family:ui-sans-serif,system-ui;padding:18px}
h1{font-size:20px;margin:0 0 4px} h2{font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);margin:0 0 10px}
.sub{color:var(--mut);font-size:12px;margin-bottom:14px}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px}
.span2{grid-column:span 2}.span3{grid-column:span 3}.span4{grid-column:span 4}.span6{grid-column:span 6}.span8{grid-column:span 8}.span12{grid-column:span 12}
@media(max-width:900px){.span2,.span3,.span4,.span6,.span8{grid-column:span 12}}
.kpi{font-size:28px;font-weight:800;letter-spacing:-.03em}
.kpi small{display:block;font-size:11px;font-weight:600;color:var(--mut);letter-spacing:.08em;text-transform:uppercase}
button,select,input{background:#1a2540;color:var(--txt);border:1px solid var(--line);border-radius:8px;padding:8px 12px;font:inherit}
button.pri{background:var(--acc);color:#071018;font-weight:800;border:0;cursor:pointer}
button.pri:disabled{opacity:.5} canvas{width:100%;height:88px;display:block}
.pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700}
.ok{background:#123;color:var(--ok)}.bad{background:#311;color:var(--bad)}.warn{background:#332;color:var(--warn)}
.models{display:flex;flex-wrap:wrap;gap:6px}
.models i{font-style:normal;font-size:10px;background:#0c1324;border:1px solid var(--line);border-radius:6px;padding:4px 6px}
table{width:100%;border-collapse:collapse;font-size:12px} td,th{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;color:var(--mut)}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:10px}
pre{white-space:pre-wrap;font-size:11px;color:var(--mut);max-height:220px;overflow:auto;margin:0}
</style></head>
<body>
<h1>WT-PM Command Center</h1>
<div class="sub">25 connected models · fusion · digital twin · safety · maintenance economics</div>
<div class="row">
  <button class="pri" id="go" onclick="run()">Run full analysis</button>
  <select id="fusion">
    <option value="confidence_weighted">fusion: confidence-weighted</option>
    <option value="weighted">fusion: weighted</option>
    <option value="median">fusion: median</option>
    <option value="max">fusion: max</option>
  </select>
  <button onclick="whatif()">What-if derate 20%</button>
  <span id="status" class="sub"></span>
</div>
<div class="grid">
  <div class="card span2"><small class="kpi"><small>fault</small><span id="k_fault">—</span></small></div>
  <div class="card span2"><small class="kpi"><small>subsystem</small><span id="k_sub">—</span></small></div>
  <div class="card span2"><small class="kpi"><small>risk</small><span id="k_risk">—</span></small></div>
  <div class="card span2"><small class="kpi"><small>health</small><span id="k_hi">—</span></small></div>
  <div class="card span2"><small class="kpi"><small>RUL h</small><span id="k_rul">—</span></small></div>
  <div class="card span2"><small class="kpi"><small>safety</small><span id="k_safe">—</span></small></div>

  <div class="card span8">
    <h2>Health / fused anomaly / vibration</h2>
    <canvas id="c1"></canvas>
    <canvas id="c2"></canvas>
  </div>
  <div class="card span4">
    <h2>Seven questions</h2>
    <table id="q7"></table>
  </div>

  <div class="card span4">
    <h2>Work order</h2>
    <div id="wo">Run analysis first.</div>
  </div>
  <div class="card span4">
    <h2>Cost-risk (48h heuristic)</h2>
    <div id="cost">—</div>
  </div>
  <div class="card span4">
    <h2>Sensor trust</h2>
    <div id="qual">—</div>
  </div>

  <div class="card span6">
    <h2>Alerts</h2>
    <div id="alerts">—</div>
  </div>
  <div class="card span6">
    <h2>Why (SHAP + physics)</h2>
    <div id="why">—</div>
  </div>
  <div class="card span12">
    <h2>Hermes agent (Thought → Action → Observation)</h2>
    <pre id="hermes">—</pre>
  </div>

  <div class="card span12">
    <h2>25-model engine</h2>
    <div class="models" id="models"></div>
  </div>
  <div class="card span12">
    <h2>Operator feedback</h2>
    <div class="row">
      <input id="fb_note" placeholder="note (optional)" style="flex:1">
      <button onclick="feedback(true)">Confirm diagnosis</button>
      <button onclick="feedback(false)">Reject diagnosis</button>
    </div>
    <pre id="audit">audit trail loads after analysis</pre>
  </div>
</div>
<script>
const $=id=>document.getElementById(id);
function spark(canvas, series, color){
  const ctx=canvas.getContext('2d'); const w=canvas.width=canvas.clientWidth*2; const h=canvas.height=88*2;
  ctx.clearRect(0,0,w,h); if(!series||!series.length) return;
  const mn=Math.min(...series), mx=Math.max(...series); const span=(mx-mn)||1;
  ctx.strokeStyle=color; ctx.lineWidth=2; ctx.beginPath();
  series.forEach((v,i)=>{const x=i/(series.length-1)*w; const y=h-8-((v-mn)/span)*(h-16);
    i?ctx.lineTo(x,y):ctx.moveTo(x,y);}); ctx.stroke();
}
function pill(ok,t){return `<span class="pill ${ok}">${t}</span>`}
async function health(){
  const r=await fetch('/health'); const d=await r.json();
  const mods=d.models||[];
  $('models').innerHTML=mods.map(m=>`<i class="${m.fitted?'ok':'bad'}">${m.model_id}${m.fitted?' ✓':''}</i>`).join('');
  $('status').textContent=d.status==='ok'?' models ready':' fitting models…';
}
async function run(){
  $('go').disabled=true; $('status').textContent=' running 25-model pipeline…';
  const r=await fetch('/analyse',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({fusion_method:$('fusion').value})});
  const d=await r.json(); render(d); $('go').disabled=false;
}
function render(d){
  $('status').textContent=` done in ${d.pipeline_ms} ms · ${ (d.model_health.ran||[]).length }/25 models`;
  $('k_fault').textContent=d.what.fault;
  $('k_sub').textContent=d.where.subsystem;
  $('k_risk').textContent=d.risk_score;
  $('k_hi').textContent=(d.health_index&&d.health_index.now)!=null?d.health_index.now:'—';
  $('k_rul').textContent=d.rul.hours??'—';
  $('k_safe').textContent=d.safety.decision;
  spark($('c1'), (d.series||{}).health_index||[], '#3ddc97');
  spark($('c2'), (d.series||{}).fused_score||[], '#4cc9f0');
  $('q7').innerHTML=`
    <tr><th>What</th><td>${d.what.fault} (${d.what.fault_confidence})</td></tr>
    <tr><th>Where</th><td>${d.where.subsystem}</td></tr>
    <tr><th>Why</th><td>${Object.keys((d.why||{}).contributing_features||{}).slice(0,3).join(', ')||'—'}</td></tr>
    <tr><th>Severity</th><td>state ${d.severity.degradation_state}</td></tr>
    <tr><th>How long</th><td>${d.rul.hours} h ± ${d.rul.uncertainty_hours} via ${d.rul.source}</td></tr>
    <tr><th>What next</th><td>${d.action.action} / cost-opt ${d.action.cost_optimal||''}</td></tr>
    <tr><th>Safe?</th><td>${d.safety.decision}</td></tr>`;
  const wo=d.work_order||{};
  $('wo').innerHTML=`<b>${wo.work_order_id||''}</b> ${pill(wo.priority==='P1'?'bad':'warn', wo.priority||'')}
    <div>${wo.fault} · ${wo.subsystem}</div>
    <div>window ${wo.window_hours} h · €${wo.estimated_cost_eur}</div>
    <div class="sub">${JSON.stringify(wo.parts||{})}</div>`;
  const c=d.cost_risk||{};
  $('cost').innerHTML=`recommend <b>${c.recommended||'—'}</b> · P(fail 48h)=${c.p_fail_48h}
    <pre>${JSON.stringify(c.expected_cost_eur||{},null,1)}</pre>`;
  const q=d.sensor_quality||{};
  $('qual').innerHTML=`trust ${q.trust} · flags ${q.n_flags}
    <pre>${(q.flags||[]).slice(0,6).map(f=>f.kind+' '+f.channel).join('\\n')||'none'}</pre>`;
  const al=(d.alerts&&d.alerts.active)||[];
  $('alerts').innerHTML=al.length?al.map(a=>`${pill(a.severity==='critical'?'bad':'warn',a.severity)} ${a.message}
     <button onclick="ack('${a.alert_id}')">ack</button>`).join('<br>'):'no active alerts';
  const shap=d.why&&d.why.contributing_features||{};
  $('why').innerHTML=`<div class="sub">${(d.why&&d.why.narrative)||''}</div>
     <pre>${JSON.stringify(shap,null,1)}</pre>
     physics residual ${((d.physics||{}).power_residual_kw_now)} kW
     counterfactual: ${JSON.stringify((d.why&&d.why.counterfactual)||{})}`;
  const ht=(d.hermes&&d.hermes.trace)||[];
  $('hermes').textContent=ht.map((s,i)=>`Thought ${i+1}: ${s.thought}\nAction  ${i+1}: ${s.action}\nObserve ${i+1}: ${JSON.stringify(s.observation).slice(0,280)}`).join('\n\n')
    + '\n\nFINAL '+JSON.stringify((d.hermes&&d.hermes.final)||{},null,1);
  const ran=new Set(d.model_health.ran||[]);
  $('models').innerHTML=(d.model_health.connected||[]).map(id=>
    `<i class="${ran.has(id)?'ok':'warn'}">${id}${ran.has(id)?' ✓':' · idle'}</i>`).join('');
  loadAudit();
}
async function whatif(){
  const r=await fetch('/what-if',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({derate_pct:20})});
  const d=await r.json(); alert('derate 20% delta %\\n'+JSON.stringify(d.delta_pct||d,null,2));
}
async function ack(id){ await fetch('/alerts/ack',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({alert_id:id})}); run(); }
async function feedback(ok){
  await fetch('/feedback',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({correct:ok, note:$('fb_note').value})});
  loadAudit();
}
async function loadAudit(){ const r=await fetch('/audit'); $('audit').textContent=JSON.stringify(await r.json(),null,1); }
health();
</script>
</body></html>
"""


def serve(port: int = 8100, days: float = 10.0, seed: int = 7) -> None:
    threading.Thread(target=_bootstrap, args=(days, seed), daemon=True).start()
    httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"[serve] WT-PM platform API on :{port} (models fitting in background)")
    httpd.serve_forever()
