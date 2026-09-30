"""CLI: research and production modes for the unified WT-PM system.

  wt-pm inspect                     # registry + availability report
  wt-pm research [--days N]         # train, evaluate, compare, track
  wt-pm production [--days N]       # inference/monitoring/alerts path
  wt-pm fleet [--turbines N]        # GNN cascade analysis (m20)
  wt-pm edge --out DIR              # export ESP32 header + INT8 tflite
  wt-pm serve [--port 8100]         # HTTP API + dashboard
  wt-pm fetch-models                # clone the 24 sibling repos into external/
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


__version__ = "0.1.0"


def _make_batch(days: float, seed: int = 7, turbine_id: str = "WT-01"):
    """Simulated SCADA (reference physics engine if installed, else built-in)."""
    from wtpm_platform.simulate import make_batch
    return make_batch(days, seed=seed, turbine_id=turbine_id)


def cmd_inspect(args) -> int:
    from wtpm_platform.orchestrator import build_default_registry
    reg = build_default_registry()
    rows = []
    for spec in reg.specs():
        m = reg.get(spec.model_id)
        rows.append({
            "model_id": spec.model_id, "repository": spec.repository,
            "task": spec.task.value,
            "available": m.available(),
            "deployment": [d.value for d in spec.deployment_targets],
            "fallback": spec.fallback, "notes": spec.notes,
        })
    print(json.dumps(rows, indent=2))
    n_av = sum(r["available"] for r in rows)
    print(f"\n{n_av}/{len(rows)} adapters available in this environment "
          f"(+ the reference package wt_pm_lstm itself = 25 models)", file=sys.stderr)
    return 0


def cmd_research(args) -> int:
    from wtpm_platform.contracts import OperatingContext
    from wtpm_platform.orchestrator import Orchestrator
    from wtpm_platform.evaluation import Evaluator, ExperimentTracker, select_weights
    from wtpm_platform.adapters.prognostics import rul_proxy_hours

    print(f"[research] simulating {args.days} days of labelled SCADA ...")
    from wtpm_platform.simulate import make_training_batch, without_labels
    batch = make_training_batch(args.days, seed=args.seed)
    ctx = OperatingContext(mode="research", has_labels=True,
                           has_vibration_waveform=True)
    orch = Orchestrator(max_workers=args.workers)
    batch = orch.prepare(batch)

    print("[research] fitting on permitted rows (healthy-only for anomaly models) ...")
    status = orch.fit(batch, ctx)

    print("[research] running full analysis ...")
    live = orch.prepare(without_labels(_make_batch(args.days, seed=args.seed + 100)))
    result = orch.analyse(live, OperatingContext(mode="production", has_labels=False,
                                                has_vibration_waveform=True),
                          parallel=not args.sequential)

    # ---- evaluation on a HELD-OUT record (different seed, unseen faults) ----
    print("[research] evaluating on a held-out record (seed+37) ...")
    eval_batch = orch.prepare(_make_batch(args.days, seed=args.seed + 37,
                                          turbine_id="WT-eval"))
    ev = Evaluator()
    y = np.asarray(eval_batch.meta["fault_label"]).astype(int)
    kinds = eval_batch.meta["fault_kind_per_step"]
    rul_true = rul_proxy_hours(eval_batch)
    eval_batch = without_labels(eval_batch)
    outs = []
    for mid in result["model_health"]["ran"]:
        m = orch.registry.get(mid)
        o = m.predict(eval_batch)
        if o.ok:
            outs.append(o)
    anomaly_eval = ev.evaluate_anomaly_models(
        [o for o in outs if o.anomaly_score is not None], y)
    clf_eval = ev.evaluate_classifiers(
        [o for o in outs if o.prediction is not None and o.probability], kinds)
    rul_eval = ev.evaluate_rul_models(
        [o for o in outs if o.rul_hours is not None], rul_true)

    weights = select_weights(anomaly_eval)
    tracker = ExperimentTracker(root=args.runs_dir)
    run_dir = tracker.save_run(
        "research",
        config={"days": args.days, "seed": args.seed, "mode": "research"},
        metrics={"anomaly": anomaly_eval, "classification": clf_eval,
                 "rul": rul_eval, "fusion_weights": weights,
                 "scope": "synthetic validation set; proposed weights require a separate final test set",
                 "fit_status": status,
                 "risk_score": result["risk_score"],
                 "diagnosis": result["what"]["fault"]},
    )
    print("\n=== cross-model comparison (comparable tasks only) ===")
    for name, table, cols in (
        ("ANOMALY", anomaly_eval, ["roc_auc", "f1", "false_alarm_rate", "missed_fault_rate", "latency_ms"]),
        ("CLASSIFICATION", clf_eval, ["accuracy", "macro_f1", "ece_healthy", "latency_ms"]),
        ("RUL", rul_eval, ["mae", "rmse", "late_fraction", "latency_ms"]),
    ):
        print(f"\n[{name}]")
        for mid, m in sorted(table.items()):
            vals = "  ".join(f"{c}={m.get(c, float('nan')):.3f}" for c in cols if c in m)
            print(f"  {mid:32s} {vals}")
    print(f"\n[research] learned fusion weights: "
          + ", ".join(f"{k}={v:.2f}" for k, v in weights.items()))
    print(f"[research] run record: {run_dir}")
    print(f"[research] final diagnosis: {result['what']['fault']} "
          f"(conf {result['what']['fault_confidence']}), risk {result['risk_score']}, "
          f"action {result['action']['action']}")
    return 0


def cmd_production(args) -> int:
    from wtpm_platform.contracts import OperatingContext
    from wtpm_platform.orchestrator import Orchestrator
    from wtpm_platform.evaluation import drift_report

    print(f"[production] simulating {args.days} days of live SCADA ...")
    # train on one record, score a *different* one (no labels used at inference)
    from wtpm_platform.simulate import make_training_batch, without_labels
    train_batch = make_training_batch(args.days, seed=args.seed)
    live_batch = without_labels(_make_batch(args.days, seed=args.seed + 100, turbine_id="WT-02"))

    ctx_fit = OperatingContext(mode="research", has_labels=True, has_vibration_waveform=True)
    ctx = OperatingContext(mode="production", has_labels=False, has_vibration_waveform=True)
    orch = Orchestrator(max_workers=args.workers)
    train_batch = orch.prepare(train_batch)
    live_batch = orch.prepare(live_batch)
    print("[production] fitting on the training record (labels allowed there) ...")
    orch.fit(train_batch, ctx_fit, verbose=False)

    print("[production] analysing live record (no labels) ...")
    result = orch.analyse(live_batch, ctx)

    n_train = int(train_batch.n_steps * 0.45)
    result["drift"] = drift_report(train_batch.features[:n_train],
                                   live_batch.features,
                                   train_batch.feature_names)
    # twin what-if for the operator
    result["what_if_derate_20pct"] = orch.twin.maintenance_scenario(
        live_batch, None, derate_pct=20.0)

    out = args.out or "production_result.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps({k: result[k] for k in
                      ("what", "where", "severity", "rul", "risk_score",
                       "action", "safety", "drift", "pipeline_ms")},
                     indent=2, default=str))
    print(f"[production] full record -> {out}")
    return 0


SIBLING_REPOS = (
    "wt-pm-1d-cnn-bearing-vibration", "wt-pm-convlstm-wear-prognostics",
    "wt-pm-tcn-power-curve", "wt-pm-gru-scada-telemetry",
    "wt-pm-informer-long-sequence", "wt-pm-snn-event-vibration",
    "wt-pm-contrastive-ssl-vibration", "wt-pm-dbn-feature-extraction",
    "wt-pm-random-forest-telemetry", "wt-pm-xgboost-tabular-faults",
    "wt-pm-svm-rbf-generator-stator", "wt-pm-deep-svdd-boundary",
    "wt-pm-isolation-forest-telemetry", "wt-pm-hmm-degradation-states",
    "wt-pm-particle-filter-rul", "wt-pm-mlp-rul-regression",
    "wt-pm-pg-bnn-wind-turbine", "wt-pm-digital-twin-surrogate",
    "wt-pm-gnn-turbines-cascade", "wt-pm-xai-shap-interpretable",
    "wt-pm-vae-reconstruction-loss", "wt-pm-aerozip-autoencoder-compressor",
    "wt-pm-quantized-mobilenet-edge", "wt-pm-tinyml-esp32-safety-relay",
)


_PARTICLE_FILTER_STUB = '''\
"""Stub for wt-pm-particle-filter-rul — replace with the real research module.

Minimal pure-numpy particle filter over RUL, implementing the interface the
platform adapter (wtpm_platform/adapters/prognostics.py, m16) calls:

    pf = ParticleFilterRUL(num_particles=..., init_rul=...)
    pf.predict(degradation_rate=..., noise=...)   # time update
    pf.update(observed_rul=..., obs_noise=...)    # measurement update
    pf.estimate()                                 # weighted-mean RUL
"""
import numpy as np


class ParticleFilterRUL:
    def __init__(self, num_particles=800, init_rul=400.0, rng_seed=0):
        self.num_particles = int(num_particles)
        self.particles = np.full(self.num_particles, float(init_rul))
        self.weights = np.full(self.num_particles, 1.0 / self.num_particles)
        self._rng = np.random.default_rng(rng_seed)

    def predict(self, degradation_rate=1.0, noise=0.5):
        self.particles = (self.particles - degradation_rate
                          + self._rng.normal(0.0, noise, size=self.num_particles))
        self.particles = np.maximum(self.particles, 0.0)
        self.weights = np.full(self.num_particles, 1.0 / self.num_particles)

    def update(self, observed_rul, obs_noise=25.0):
        w = np.exp(-0.5 * ((self.particles - float(observed_rul))
                           / float(obs_noise)) ** 2)
        s = float(w.sum())
        self.weights = (w / s if s > 0
                        else np.full(self.num_particles, 1.0 / self.num_particles))
        self._resample()

    def _resample(self):
        self.particles = self._rng.choice(
            self.particles, size=self.num_particles, replace=True, p=self.weights)
        self.weights = np.full(self.num_particles, 1.0 / self.num_particles)

    def estimate(self) -> float:
        return float(np.sum(self.particles * self.weights))
'''

#: repo name -> functional stub model.py source; everything else gets the
#: generic placeholder below.
_STUB_MODELS = {"wt-pm-particle-filter-rul": _PARTICLE_FILTER_STUB}


def _stub_sibling(path: str, name: str) -> None:
    """Write the local scaffold for one sibling repo (README + model.py).

    The particle-filter repo gets a *functional* pure-numpy stub: m16 is the
    only sibling whose adapter runs on numpy alone, so its stub must actually
    filter for the platform test suite to pass in lean environments (CI).
    The other 23 adapters are gated on torch/tensorflow/xgboost/... and stay
    honestly unavailable there, so a placeholder suffices.
    """
    os.makedirs(path, exist_ok=True)
    readme = os.path.join(path, "README.md")
    model = os.path.join(path, "model.py")
    if not os.path.exists(readme):
        with open(readme, "w", encoding="utf-8") as f:
            f.write(f"# {name}\n\nLocal scaffold (GitHub write not available from this session).\n")
    if not os.path.exists(model):
        src = _STUB_MODELS.get(name) or (
            f'"""Stub for {name} — replace with the real research module."""\n'
            "class Model:\n"
            "    def available(self) -> bool:\n"
            "        return False\n"
        )
        with open(model, "w", encoding="utf-8") as f:
            f.write(src)


def cmd_fetch_models(args) -> int:
    """Clone the 24 sibling research repos into ``external/`` (or local stubs)."""
    import subprocess
    dest = os.path.abspath(args.dir)
    os.makedirs(dest, exist_ok=True)
    ok = 0
    stub = bool(getattr(args, "stub", False))
    for r in SIBLING_REPOS:
        path = os.path.join(dest, r)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "model.py")):
            print(f"  [skip] {r}")
            ok += 1
            continue
        if stub:
            _stub_sibling(path, r)
            print(f"  [stub] {r}")
            ok += 1
            continue
        url = f"https://github.com/rajaram-2005/{r}.git"
        print(f"  [clone] {url}")
        rc = subprocess.call(["git", "clone", "--depth", "1", url, path])
        if rc == 0:
            ok += 1
        else:
            print(f"  [fail] {r} exit={rc}  (retry with --stub)")
    print(f"[fetch-models] {ok}/{len(SIBLING_REPOS)} repos in {dest}")
    print(f"               export WTPM_EXTERNAL_DIR={dest}")
    return 0 if ok == len(SIBLING_REPOS) else 1


def cmd_mock_bus(args) -> int:
    from wtpm_platform.drivers import JsonTcpServer, InMemoryBus
    bus = InMemoryBus()
    srv = JsonTcpServer(bus, host=args.host, port=args.port).start()
    print(f"[mock-bus] JSON/TCP on {args.host}:{args.port}  (op=read|write)")
    print("[mock-bus] optional: pip install pymodbus asyncua paho-mqtt")
    try:
        if args.once:
            from wtpm_platform.drivers import client_read
            print(json.dumps(client_read(args.host if args.host != "0.0.0.0" else "127.0.0.1",
                                         args.port), indent=2))
            return 0
        import time
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        srv.stop()
    return 0


def cmd_sil(args) -> int:
    from wtpm_platform.safety_gate import safety_gate_check
    r = safety_gate_check(pitch_angle=args.pitch, rpm=args.rpm, vibration_mm_s=args.vib)
    print(json.dumps(r, indent=2))
    print(r["message"])
    return 0 if r["ok"] else 1


def cmd_field(args) -> int:
    from wtpm_platform.field import hot_tag_map, write_hot_sample_csv
    from wtpm_platform.scada import probe_headers
    path = args.sample or "hot_sample.csv"
    write_hot_sample_csv(path, n=args.rows)
    import csv as _csv
    with open(path, newline="", encoding="utf-8") as f:
        headers = next(_csv.reader(f))
    report = probe_headers(headers, hot_tag_map())
    print(json.dumps({"csv": path, "probe": report, "map": hot_tag_map()}, indent=2, default=str))
    return 0 if report["ready"] else 1


def cmd_fleet(args) -> int:
    from wtpm_platform.contracts import OperatingContext
    from wtpm_platform.orchestrator import Orchestrator

    print(f"[fleet] simulating {args.turbines} wake-coupled turbines ...")
    try:
        from wt_pm_lstm.config import DataConfig
        from wt_pm_lstm.simulate import simulate_fleet
        from wtpm_platform.data import ingest_timeline
        cfg = DataConfig(); cfg.n_days = args.days; cfg.seed = args.seed
        tls = simulate_fleet(cfg, n_turbines=args.turbines)
        batches = [ingest_timeline(t) for t in tls]
    except Exception:
        batches = [_make_batch(args.days, seed=args.seed + i,
                               turbine_id=f"WT-{i+1:02d}")
                   for i in range(args.turbines)]
    orch = Orchestrator(max_workers=args.workers)
    ctx = OperatingContext(mode="research", has_labels=True)
    b0 = orch.prepare(batches[0])
    if "tls" in locals():
        b0.meta["fault_kind_per_step"] = tls[0].meta.get("fault_kind_per_step")
        b0.meta["fault_label"] = tls[0].fault_label
    from wtpm_platform.simulate import make_training_batch
    orch.fit(orch.prepare(make_training_batch(args.days, args.seed + 500)), ctx,
             model_ids=["m14-isolation-forest", "m20-gnn-cascade"], verbose=False)
    batches = [orch.prepare(b) for b in batches]
    # wake edges: chain by layout order (upstream -> downstream, both dirs)
    n = len(batches)
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j and abs(i - j) <= 3:
                src.append(i); dst.append(j)
    out = orch.analyse_fleet(batches, np.array([src, dst]))
    if out is None or not out.ok:
        print("[fleet] m20 unavailable:", "" if out is None else out.error)
        return 1
    for tid, risk in zip(out.extra["turbine_ids"], out.prediction):
        print(f"  {tid}: cascade risk = {risk}")
    return 0


def cmd_edge(args) -> int:
    from wtpm_platform.contracts import OperatingContext
    from wtpm_platform.orchestrator import Orchestrator

    from wtpm_platform.simulate import make_training_batch
    batch = make_training_batch(args.days, seed=args.seed)
    orch = Orchestrator()
    batch = orch.prepare(batch)
    ctx = OperatingContext(mode="research", has_labels=True)
    orch.fit(batch, ctx, model_ids=["m25-tinyml-safety"], verbose=False)
    os.makedirs(args.out, exist_ok=True)
    arts = orch.safety.export_mcu_artifacts(args.out)
    arts.update(orch.edge.export_edge_bundle(batch, args.out))
    print(json.dumps({"deployment_plan": orch.edge.deployment_plan(),
                      "artifacts": arts}, indent=2, default=str))
    return 0


def cmd_serve(args) -> int:
    from wtpm_platform.api import serve
    serve(port=args.port, days=args.days, seed=args.seed)
    return 0


def cmd_agent(args) -> int:
    """Run the Hermes Thought/Action/Observation loop and print the trace."""
    from wtpm_platform.contracts import OperatingContext
    from wtpm_platform.orchestrator import Orchestrator

    print(f"[agent] Hermes loop on {args.days} days of SCADA ...")
    from wtpm_platform.simulate import make_training_batch
    batch = make_training_batch(args.days, seed=args.seed)
    orch = Orchestrator(max_workers=args.workers)
    ctx = OperatingContext(mode="research", has_labels=True, has_vibration_waveform=True)
    batch = orch.prepare(batch)
    orch.fit(batch, ctx, verbose=not args.quiet if hasattr(args, "quiet") else False)
    from wtpm_platform.simulate import without_labels
    live = orch.prepare(without_labels(_make_batch(args.days, seed=args.seed + 100)))
    result = orch.analyse(live, OperatingContext(mode="production", has_labels=False,
                                                has_vibration_waveform=True))
    hermes = result.get("hermes") or {}
    print(f"\n[agent] goal: {hermes.get('goal')}")
    print("[agent] principles:", ", ".join(hermes.get("principles") or []))
    for i, step in enumerate(hermes.get("trace") or [], 1):
        print(f"\n  Thought {i}: {step['thought']}")
        print(f"  Action  {i}: {step['action']} {step.get('args') or ''}")
        obs = step.get("observation") or {}
        brief = {k: v for k, v in obs.items() if k != "contrast_vs_healthy"}
        print(f"  Observe {i}: {json.dumps(brief, default=str)[:400]}")
    fin = hermes.get("final") or {}
    print("\n[agent] FINAL")
    print(json.dumps(fin, indent=2, default=str))
    return 0


def cmd_scada(args) -> int:
    """Combine a plant SCADA export with the 25-model platform."""
    from wtpm_platform.contracts import OperatingContext
    from wtpm_platform.orchestrator import Orchestrator
    from wtpm_platform.scada import (
        default_map_document, emit_scada_tags, ingest_scada_csv,
        load_tag_map, write_tags_csv,
    )

    if getattr(args, "probe", ""):
        import csv as _csv
        with open(args.probe, newline="", encoding="utf-8-sig") as f:
            headers = next(_csv.reader(f))
        from wtpm_platform.scada import probe_headers
        extra = load_tag_map(args.map) if args.map else {}
        report = probe_headers(headers, extra)
        print(json.dumps(report, indent=2))
        print(f"[scada] coverage {report['coverage']:.0%}  "
              f"mapped {len(report['mapped'])}  unknown {len(report['unknown'])}  "
              f"missing {report['missing_canonical']}", file=sys.stderr)
        return 0 if report["ready"] else 1

    if args.write_map:
        path = args.write_map
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default_map_document(), f, indent=2)
        print(f"[scada] example tag map -> {path}")
        return 0

    tag_map = load_tag_map(args.map) if args.map else {}
    if args.demo and not args.inp:
        from wtpm_platform.data import canonical_channels
        batch = _make_batch(args.days, seed=args.seed, turbine_id=args.turbine)
        demo = args.out.replace(".json", "_in.csv") if args.out.endswith(".json") else "scada_demo_in.csv"
        chans = list(canonical_channels())
        import csv as _csv
        with open(demo, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["timestamp", "turbine_id"] + chans)
            for i in range(batch.n_steps):
                w.writerow([int(batch.timestamps[i]), batch.turbine_id]
                           + [f"{batch.values[i, j]:.6f}" for j in range(len(chans))])
        args.inp = demo
        print(f"[scada] demo historian CSV -> {demo}")

    if not args.inp:
        print("[scada] need --in historian.csv  (or --demo / --write-map)", file=sys.stderr)
        return 2

    print(f"[scada] ingest {args.inp} ...")
    live = ingest_scada_csv(args.inp, turbine_id=args.turbine, tag_map=tag_map)
    print(f"[scada] mapped {live.meta.get('scada_mapped_channels')}  "
          f"unmapped={live.meta.get('scada_unmapped_headers')}")

    orch = Orchestrator(max_workers=args.workers)
    ctx_fit = OperatingContext(mode="research", has_labels=True, has_vibration_waveform=True)
    ctx = OperatingContext(mode="production", has_labels=False, has_vibration_waveform=True)
    live = orch.prepare(live)
    print("[scada] fitting on the leading band of this export ...")
    orch.fit(live, ctx_fit, verbose=False)
    result = orch.analyse(live, ctx)
    tags = emit_scada_tags(result, live.turbine_id)
    out = args.out or "scada_writeback.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(tags, f, indent=2, default=str)
    csv_path = os.path.splitext(out)[0] + "_points.csv"
    write_tags_csv(tags, csv_path)
    print(json.dumps(tags["tags"], indent=2, default=str))
    print(f"[scada] writeback JSON -> {out}")
    print(f"[scada] SCADA point list CSV -> {csv_path}")
    return 0


def cmd_watch(args) -> int:
    """Historian hot-folder: drop CSV in, get WTPM writeback out."""
    import glob
    import shutil
    import time
    from wtpm_platform.config import load_config
    from wtpm_platform.contracts import OperatingContext
    from wtpm_platform.orchestrator import Orchestrator
    from wtpm_platform.scada import (
        emit_scada_tags, ingest_scada_csv, load_tag_map, split_by_turbine,
        write_tags_csv,
    )
    import csv as _csv

    cfg = load_config(args.config or None)
    drop = args.dir or cfg["drop_dir"]
    out_dir = cfg["out_dir"]
    done = cfg["processed_dir"]
    poll = args.poll or float(cfg["poll_seconds"])
    os.makedirs(drop, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(done, exist_ok=True)
    tag_map = load_tag_map(cfg["map"]) if os.path.isfile(str(cfg.get("map", ""))) else {}
    print(f"[watch] drop={drop} out={out_dir} poll={poll}s config={cfg.get('_path')}")

    orch = None
    ctx_fit = OperatingContext(mode="research", has_labels=True, has_vibration_waveform=True)
    ctx = OperatingContext(mode="production", has_labels=False, has_vibration_waveform=True)

    def handle(path: str) -> None:
        nonlocal orch
        print(f"[watch] {path}")
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(_csv.DictReader(f))
        batches = split_by_turbine(rows, tag_map=tag_map)
        if orch is None:
            orch = Orchestrator(max_workers=args.workers)
        for tid, batch in batches.items():
            batch = orch.prepare(batch)
            orch.fit(batch, ctx_fit, verbose=False)
            result = orch.analyse(batch, ctx)
            tags = emit_scada_tags(result, tid)
            stem = os.path.splitext(os.path.basename(path))[0] + "_" + tid
            jp = os.path.join(out_dir, stem + ".json")
            cp = os.path.join(out_dir, stem + "_points.csv")
            with open(jp, "w", encoding="utf-8") as f:
                json.dump(tags, f, indent=2, default=str)
            write_tags_csv(tags, cp)
            print(f"[watch] {tid} -> {jp}  risk={tags['tags'].get('WTPM.RISK')} "
                  f"fault={tags['tags'].get('WTPM.FAULT')}")
        shutil.move(path, os.path.join(done, os.path.basename(path)))

    seen = set()
    while True:
        for path in sorted(glob.glob(os.path.join(drop, "*.csv"))):
            if path in seen:
                continue
            try:
                handle(path)
                seen.add(path)
            except Exception as exc:  # noqa: BLE001
                print(f"[watch] FAIL {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
        if args.once:
            return 0
        time.sleep(max(poll, 1))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="wt-pm",
        description="WT-PM unified predictive-maintenance platform (25 models).",
    )
    p.add_argument("--version", action="version", version=f"wt-pm {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--days", type=float, default=20.0)
        sp.add_argument("--seed", type=int, default=7)
        sp.add_argument("--workers", type=int, default=4)

    sp = sub.add_parser("inspect"); sp.set_defaults(fn=cmd_inspect)
    sp = sub.add_parser("research"); common(sp)
    sp.add_argument("--sequential", action="store_true")
    sp.add_argument("--runs-dir", default="runs_platform")
    sp.set_defaults(fn=cmd_research)
    sp = sub.add_parser("production"); common(sp)
    sp.add_argument("--out", default="")
    sp.set_defaults(fn=cmd_production)
    sp = sub.add_parser("fleet"); common(sp)
    sp.add_argument("--turbines", type=int, default=6)
    sp.set_defaults(fn=cmd_fleet)
    sp = sub.add_parser("edge"); common(sp)
    sp.add_argument("--out", default="edge_artifacts")
    sp.set_defaults(fn=cmd_edge)
    sp = sub.add_parser("serve"); common(sp)
    sp.add_argument("--port", type=int, default=os.environ.get("PORT", "8100"))
    sp.set_defaults(fn=cmd_serve)
    sp = sub.add_parser("fetch-models",
                        help="clone the 24 sibling wt-pm-* repos into external/")
    sp.add_argument("--dir", default="external")
    sp.add_argument("--stub", action="store_true",
                    help="write local model.py stubs instead of git clone")
    sp.set_defaults(fn=cmd_fetch_models)
    sp = sub.add_parser("mock-bus", help="stdlib JSON/TCP SCADA register emulator")
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=15020)
    sp.add_argument("--once", action="store_true")
    sp.set_defaults(fn=cmd_mock_bus)
    sp = sub.add_parser("sil", help="software SIL gate (drop overspeed / negative pitch)")
    sp.add_argument("--rpm", type=float, default=12.0)
    sp.add_argument("--pitch", type=float, default=2.0)
    sp.add_argument("--vib", type=float, default=2.0)
    sp.set_defaults(fn=cmd_sil)
    sp = sub.add_parser("field", help="Hill-of-Towie-shaped sample CSV + tag map")
    sp.add_argument("--sample", default="hot_sample.csv")
    sp.add_argument("--rows", type=int, default=48)
    sp.set_defaults(fn=cmd_field)
    sp = sub.add_parser("agent", help="Hermes Thought/Action/Observation loop + XAI")
    common(sp)
    sp.add_argument("--quiet", action="store_true")
    sp.set_defaults(fn=cmd_agent)
    sp = sub.add_parser("scada", help="ingest plant SCADA CSV and write WTPM.* tags back")
    common(sp)
    sp.add_argument("--in", dest="inp", default="", help="historian CSV export")
    sp.add_argument("--turbine", default="WT-SCADA")
    sp.add_argument("--map", default="", help="JSON plant_tag → canonical channel")
    sp.add_argument("--out", default="scada_writeback.json")
    sp.add_argument("--demo", action="store_true", help="write a demo CSV then score it")
    sp.add_argument("--write-map", default="", help="write example tag_map.json and exit")
    sp.add_argument("--probe", default="", help="print suggested tag map from a CSV header and exit")
    sp.add_argument("--no-fit", dest="fit", action="store_false")
    sp.set_defaults(fn=cmd_scada, fit=True)
    sp = sub.add_parser("watch", help="poll a historian drop folder and write WTPM.* tags")
    sp.add_argument("--config", default="")
    sp.add_argument("--dir", default="", help="override drop_dir")
    sp.add_argument("--once", action="store_true")
    sp.add_argument("--poll", type=float, default=0)
    sp.add_argument("--workers", type=int, default=4)
    sp.set_defaults(fn=cmd_watch)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
