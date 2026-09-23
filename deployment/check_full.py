"""Fail unless the full research deployment actually runs every model.

Never uses stubs, fabricated success flags or fallbacks to satisfy the count.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import time

# Set limits before loading numerical libraries. These also match Dockerfile.full.
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
    os.environ.setdefault(key, '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('WTPM_EXTERNAL_DIR', str(Path(__file__).resolve().parents[1] / 'external'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imports-only', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    from wtpm_platform.orchestrator import Orchestrator
    start = time.monotonic()
    orch = Orchestrator(max_workers=1)
    health = orch.registry.health_report()
    missing = [row for row in health if not row['available']]
    if len(health) != 25 or missing:
        raise RuntimeError(f'Full profile dependencies incomplete: {missing}')
    if args.imports_only:
        print('25/25 adapters import successfully (not an inference test).')
        return
    from wtpm_platform.cli import _make_batch
    from wtpm_platform.contracts import OperatingContext
    batch = orch.prepare(_make_batch(6, seed=7))
    ctx = OperatingContext(mode='research', has_labels=True, has_vibration_waveform=True)
    fit = orch.fit(batch, ctx)
    result = orch.analyse(batch, ctx, parallel=False)
    actual = result['model_health']
    expected = set(orch.registry.ids())
    if set(actual['ran']) != expected or actual['errors'] or actual['fallbacks']:
        raise RuntimeError(f'Full pipeline failed: {actual}')
    shap = result['support']['m21-xai-shap']
    edge = result['support']['m24-quantized-edge']
    if not shap.get('shap_values') or edge.get('input_dtype') != 'int8' or edge.get('size_bytes', 0) <= 0:
        raise RuntimeError('SHAP / INT8 execution evidence is missing')
    report = {
        'scope': 'CPU smoke test, six simulated days; not field accuracy validation',
        'fit': fit, 'model_health': actual,
        'shap': shap, 'edge': edge,
        'duration_seconds': round(time.monotonic() - start, 2),
        'peak_rss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
