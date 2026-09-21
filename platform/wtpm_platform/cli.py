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
    batch = _make_batch(args.days, seed=args.seed)
    ctx = OperatingContext(mode="research", has_labels=True,
                           has_vibration_waveform=True)
    orch = Orchestrator(max_workers=args.workers)
    batch = orch.prepare(batch)

    print("[research] fitting routed models on the healthy band ...")
    status = orch.fit(batch, ctx)

    print("[research] running full analysis ...")
    result = orch.analyse(batch, ctx, parallel=not args.sequential)

    # ---- evaluation on a HELD-OUT record (different seed, unseen faults) ----
    print("[research] evaluating on a held-out record (seed+37) ...")
    eval_batch = orch.prepare(_make_batch(args.days, seed=args.seed + 37,
                                          turbine_id="WT-eval"))
    ev = Evaluator()
    y = np.asarray(eval_batch.meta["fault_label"]).astype(int)
    kinds = eval_batch.meta["fault_kind_per_step"]
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
    rul_true = rul_proxy_hours(eval_batch)
    rul_eval = ev.evaluate_rul_models(
        [o for o in outs if o.rul_hours is not None], rul_true)

    weights = select_weights(anomaly_eval)
    tracker = ExperimentTracker(root=args.runs_dir)
    run_dir = tracker.save_run(
        "research",
        config={"days": args.days, "seed": args.seed, "mode": "research"},
        metrics={"anomaly": anomaly_eval, "classification": clf_eval,
                 "rul": rul_eval, "fusion_weights": weights,
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
    train_batch = _make_batch(args.days, seed=args.seed)
    live_batch = _make_batch(args.days, seed=args.seed + 100, turbine_id="WT-02")

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


def cmd_fetch_models(args) -> int:
    """Clone the 24 sibling research repos into ``external/`` (read-only)."""
    import subprocess
    dest = os.path.abspath(args.dir)
    os.makedirs(dest, exist_ok=True)
    ok = 0
    for r in SIBLING_REPOS:
        path = os.path.join(dest, r)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "model.py")):
            print(f"  [skip] {r}")
            ok += 1
            continue
        url = f"https://github.com/rajaram-2005/{r}.git"
        print(f"  [clone] {url}")
        rc = subprocess.call(["git", "clone", "--depth", "1", url, path])
        if rc == 0:
            ok += 1
        else:
            print(f"  [fail] {r} exit={rc}")
    print(f"[fetch-models] {ok}/{len(SIBLING_REPOS)} repos in {dest}")
    print(f"               export WTPM_EXTERNAL_DIR={dest}")
    return 0 if ok == len(SIBLING_REPOS) else 1


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
    orch.fit(b0, ctx, model_ids=["m14-isolation-forest"], verbose=False)
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

    batch = _make_batch(args.days, seed=args.seed)
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
    sp.add_argument("--port", type=int, default=8100)
    sp.set_defaults(fn=cmd_serve)
    sp = sub.add_parser("fetch-models",
                        help="clone the 24 sibling wt-pm-* repos into external/")
    sp.add_argument("--dir", default="external")
    sp.set_defaults(fn=cmd_fetch_models)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
