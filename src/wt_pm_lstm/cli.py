"""Command-line interface: ``wtpm <command>``.

    wtpm demo      run the full protocol and write a report + run record
    wtpm train     fit a detector and save the bundle
    wtpm score     score a SCADA CSV with a saved detector
    wtpm fleet     run one detector per turbine over a wake-coupled farm
    wtpm schema    print the data contract
    wtpm serve     start the HTTP API

Deliberately dependency-free (argparse + stdlib): the reference engine must be
runnable anywhere the research pipeline is expected to reproduce a number.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from wt_pm_lstm import __version__
from wt_pm_lstm.config import RunConfig, config_fingerprint
from wt_pm_lstm.dataio import read_csv_timeline, records_to_json, write_json
from wt_pm_lstm.pipeline import ExperimentResult, fit_detector, run_experiment, run_fleet_experiment
from wt_pm_lstm.registry import list_runs, load_detector, model_id, save_detector, save_run
from wt_pm_lstm.schema import describe_schema


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="path to a RunConfig JSON file")
    parser.add_argument("--days", type=int, help="override data.n_days")
    parser.add_argument("--epochs", type=int, help="override model.epochs")
    parser.add_argument("--ensemble", type=int, help="override model.n_ensemble")
    parser.add_argument("--stride", type=int, help="override model.stride")
    parser.add_argument("--cell", choices=("lstm", "gru"), help="override model.cell")
    parser.add_argument("--seed", type=int, help="override data.seed")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")


def _config_from_args(args: argparse.Namespace) -> RunConfig:
    cfg = RunConfig.from_json(args.config) if getattr(args, "config", None) else RunConfig()
    if getattr(args, "days", None):
        cfg.data.n_days = int(args.days)
    if getattr(args, "epochs", None):
        cfg.model.epochs = int(args.epochs)
    if getattr(args, "ensemble", None):
        cfg.model.n_ensemble = int(args.ensemble)
    if getattr(args, "stride", None):
        cfg.model.stride = int(args.stride)
    if getattr(args, "cell", None):
        cfg.model.cell = args.cell
    if getattr(args, "seed", None):
        cfg.data.seed = int(args.seed)
    if getattr(args, "name", None):
        cfg.name = args.name
    return cfg


def _print_summary(summary: Dict[str, Any]) -> None:
    print(f"  model_id           {summary['model_id']}")
    print(f"  configuration      {summary['fingerprint']}  ({summary['cell']}, {summary['n_parameters']} params,"
          f" {summary['ensemble']} ensemble members)")
    print(f"  threshold          {summary['threshold']:.4f}")
    print(f"  point  P/R/F1      {summary['precision']:.3f} / {summary['recall']:.3f} / {summary['f1']:.3f}")
    print(f"  ranking PR/ROC     {summary['pr_auc']:.3f} / {summary['roc_auc']:.3f}")
    print(f"  event recall       {summary['event_recall']}")
    print(f"  median latency     {summary['median_latency_samples']} samples")
    print(f"  false alarms/day   {summary['false_alarms_per_day']:.2f}")
    print(f"  drift detected     {summary['drift_detected']}")
    print(f"  wall time          {summary['seconds']}s")


def cmd_demo(args: argparse.Namespace) -> int:
    """Run the complete protocol on simulated data and persist every artefact."""
    cfg = _config_from_args(args)
    if args.name:
        cfg.name = args.name
    print(f"wtpm demo — {cfg.name} (engine={cfg.engine})")
    result = run_experiment(cfg, verbose=not args.quiet)
    _print_summary(result.summary())

    out = args.out or os.path.join(cfg.output_dir, "demo")
    os.makedirs(out, exist_ok=True)
    from wt_pm_lstm.report import markdown_report

    report_path = os.path.join(out, "REPORT.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(markdown_report(result))
    write_json(os.path.join(out, "score_trace.json"), _trace_payload(result))
    save_run(
        cfg,
        metrics=result.metrics,
        history=[h.to_dict() for h in result.histories],
        records=records_to_json(result.detection.records),
        extras={"drift": result.drift.to_dict(), "split": result.plan.to_dict()},
    )
    prefix = os.path.join(out, "detector")
    save_detector(result.detector, prefix)
    print(f"\n  report      {report_path}")
    print(f"  detector    {prefix}.npz + {prefix}.json")
    print(f"  run record  runs/{cfg.name}-{result.fingerprint}/")
    return 0


def _trace_payload(result: ExperimentResult) -> Dict[str, Any]:
    test = result.timeline.slice(*result.plan.test)
    det = result.detection
    return {
        "turbine_id": test.turbine_id,
        "timestamps": test.timestamps.tolist(),
        "score": det.score.tolist(),
        "threshold": float(det.threshold),
        "alarm": det.alarm.astype(int).tolist(),
        "fault_label": (test.fault_label.astype(int).tolist() if test.fault_label is not None else []),
        "channels": list(det.channel_names),
        "attribution": det.attribution.tolist(),
        "records": records_to_json(det.records),
        "components": {k: v.tolist() for k, v in det.components.items() if isinstance(v, np.ndarray)},
    }


def cmd_train(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    from wt_pm_lstm.pipeline import generate_record
    from wt_pm_lstm.windows import split_indices

    n_steps = int(cfg.data.n_days * 24 * 60 / cfg.data.sample_minutes)
    plan = split_indices(n_steps, cfg.data.healthy_fraction, cfg.data.calibration_fraction, None, cfg.model.window)
    timeline = generate_record(cfg, plan)
    detector, _, histories = fit_detector(cfg, timeline, verbose=not args.quiet)
    prefix = args.out or os.path.join(cfg.output_dir, "detector")
    npz, js = save_detector(detector, prefix)
    save_run(
        cfg,
        metrics={"stage": "train", "threshold": detector.threshold.to_dict()},
        history=[h.to_dict() for h in histories],
    )
    print(f"trained {model_id(cfg)} -> {npz}")
    print(f"  threshold {detector.threshold.value:.4f} (95% CI [{detector.threshold.ci_low:.4f}, {detector.threshold.ci_high:.4f}])")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    detector = load_detector(args.detector)
    timeline = read_csv_timeline(
        args.csv, turbine_id=args.turbine_id, channel_names=detector.channel_names
    )
    problems = timeline.validate()
    if problems:
        print("  schema warnings:", "; ".join(problems))
    result = detector.score_timeline(timeline)
    print(f"scored {timeline.turbine_id}: {result.n_steps} samples, {len(result.records)} alarm events, "
          f"threshold={result.threshold:.4f}")
    for record in result.records[: args.max_records]:
        print(f"  {record.timestamp}  score={record.score:.2f}  {record.recommended_action}")
        print(f"      {record.explanation}")
    if args.json_out:
        write_json(args.json_out, {"summary": result.summary(), "records": records_to_json(result.records)})
        print(f"  wrote {args.json_out}")
    return 0


def cmd_fleet(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    payload = run_fleet_experiment(cfg, n_turbines=args.turbines, verbose=not args.quiet)
    print(f"fleet: {payload['n_turbines']} turbines in {payload['seconds']:.1f}s")
    for entry in payload["per_turbine"]:
        m = entry["metrics"]["point"]
        print(
            f"  {entry['turbine_id']:22s} F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} "
            f"wake_deficit={entry['wake_deficit_mean']:.3f}"
        )
    print(f"  cross-turbine alarm correlation: {payload['cross_turbine_alarm_correlation']:+.3f}")
    if args.out:
        write_json(args.out, {k: v for k, v in payload.items() if k != "score_matrix"})
        print(f"  wrote {args.out}")
    return 0


def cmd_schema(args: argparse.Namespace) -> int:
    payload = describe_schema()
    payload["engine_version"] = __version__
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    runs = list_runs(args.root)
    if not runs:
        print(f"no run records under {args.root!r}")
        return 0
    for r in runs:
        print(f"  {r['run']:48s} F1={r['f1'] if r['f1'] is None else round(r['f1'], 3)}  PR-AUC={r['pr_auc']}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from wt_pm_lstm.api import serve

    cfg = _config_from_args(args)
    return serve(host=args.host, port=args.port, cfg=cfg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wtpm", description="WT-PM model 13 — LSTM SCADA anomaly detection")
    parser.add_argument("--version", action="version", version=f"wt-pm-lstm-scada-anomaly {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("demo", help="run the full protocol on simulated data")
    _add_common(p)
    p.add_argument("--out", default=None, help="output directory (default artifacts/demo)")
    p.add_argument("--name", default=None, help="run name")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("train", help="fit a detector and save the bundle")
    _add_common(p)
    p.add_argument("--out", default=None, help="bundle path prefix")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("score", help="score a SCADA CSV with a saved detector")
    p.add_argument("--detector", required=True, help="bundle path prefix (without .npz/.json)")
    p.add_argument("--csv", required=True)
    p.add_argument("--turbine-id", default="WT-001")
    p.add_argument("--max-records", type=int, default=10)
    p.add_argument("--json-out", default=None)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("fleet", help="run a detector per turbine over a wind farm")
    _add_common(p)
    p.add_argument("--turbines", type=int, default=6)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_fleet)

    p = sub.add_parser("schema", help="print the data contract")
    p.set_defaults(func=cmd_schema)

    p = sub.add_parser("runs", help="list run records")
    p.add_argument("--root", default="runs")
    p.set_defaults(func=cmd_runs)

    p = sub.add_parser("serve", help="start the HTTP API")
    _add_common(p)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
