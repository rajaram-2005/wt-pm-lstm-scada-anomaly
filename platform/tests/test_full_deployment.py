"""Regression checks for honest model accounting and full-profile readiness."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wtpm_platform import api, base
from wtpm_platform.adapters.physics_graph_edge import XAIShapInterpretable
from wtpm_platform.contracts import OperatingContext, TaskType
from wtpm_platform.orchestrator import Orchestrator


def test_external_dir_is_resolved_at_call_time(tmp_path, monkeypatch):
    for name, value in [('a', 1), ('b', 2)]:
        root = tmp_path / name
        (root / 'example').mkdir(parents=True)
        (root / 'example' / 'model.py').write_text(f'value = {value}\n')
        monkeypatch.setenv('WTPM_EXTERNAL_DIR', str(root))
        assert base.load_repo_module('example').value == value


def test_failed_import_is_not_cached(tmp_path, monkeypatch):
    path = tmp_path / 'broken'
    path.mkdir()
    (path / 'model.py').write_text('raise ImportError("missing dependency")\n')
    monkeypatch.setenv('WTPM_EXTERNAL_DIR', str(tmp_path))
    for _ in range(2):
        with pytest.raises(ImportError, match='missing dependency'):
            base.load_repo_module('broken')


def test_fit_failure_is_visible_and_clears_previous_success():
    class Failing(base.BaseWTModel):
        spec = base.ModelSpec('bad', 'example', TaskType.ANOMALY_DETECTION,
                              [], [], [], 0, [])
        def fit(self, batch, mask):
            raise ValueError('invalid input')
        def _predict(self, batch):
            raise AssertionError('must not predict')
    model = Failing()
    model._fitted = True
    registry = base.ModelRegistry()
    registry.register(model)
    orch = Orchestrator(registry)
    batch = SimpleNamespace(features=np.zeros((10, 2)), n_steps=10, meta={})
    orch.fit(batch, OperatingContext(), verbose=False)
    row = model.health()
    assert row['available'] and not row['fitted']
    assert row['fit_status'] == 'fit failed'
    assert row['reason'] == 'ValueError: invalid input'


def test_shap_cannot_report_connected_as_success(monkeypatch):
    model = XAIShapInterpretable()
    monkeypatch.setattr(model, '_check_deps', lambda: None)
    model._fitted = True
    result = model.predict(SimpleNamespace(turbine_id='test', n_steps=1, values=np.zeros((1, 1)), timestamps=np.array([0])))
    assert not result.ok
    assert 'fitted upstream tree' in result.error


def test_full_readiness_rejects_missing_models_errors_and_fallbacks():
    names = [f'm{i:02}' for i in range(25)]
    orch = SimpleNamespace(registry=SimpleNamespace(ids=lambda: names))
    valid = {'ran': names, 'errors': {}, 'fallbacks': {}}
    api._require_full_result(orch, {'model_health': valid})
    for change in ({'ran': names[:-1]}, {'ran': names + ['fused-rul']},
                   {'errors': {'m01': 'failed'}},
                   {'fallbacks': {'m01': 'm02'}}):
        with pytest.raises(RuntimeError, match='Full profile verification failed'):
            api._require_full_result(orch, {'model_health': {**valid, **change}})


def test_bootstrap_reports_failure(monkeypatch):
    from wtpm_platform import cli
    monkeypatch.setattr(api, '_STATE', {})
    def fail(*args, **kwargs):
        raise ValueError('bad simulator input')
    monkeypatch.setattr(cli, '_make_batch', fail)
    api._bootstrap(6, 7)
    assert not api._STATE['fitted']
    assert 'bad simulator input' in api._STATE['error']


def test_source_manifest_is_complete():
    from wtpm_platform.cli import SIBLING_REPOS
    root = Path(__file__).resolve().parents[2]
    rows = json.loads((root / 'deployment/models.lock.json').read_text())['models']
    assert {r['repository'] for r in rows} == set(SIBLING_REPOS)
    assert all(len(r['revision']) == 40 and len(r['model_sha256']) == 64 for r in rows)


@pytest.mark.skipif(os.environ.get('WTPM_TEST_FULL') != '1',
                    reason='requires the full CPU dependencies and real sibling sources')
def test_full_cpu_pipeline():
    from wtpm_platform.simulate import make_batch, make_training_batch, without_labels
    orch = Orchestrator(max_workers=1)
    training = orch.prepare(make_training_batch(6, seed=7))
    batch = orch.prepare(without_labels(make_batch(6, seed=107)))
    ctx = OperatingContext(mode='research', has_labels=True, has_vibration_waveform=True)
    orch.fit(training, ctx, verbose=False)
    ctx = OperatingContext(mode="production", has_labels=False, has_vibration_waveform=True)
    result = orch.analyse(batch, ctx, parallel=False)
    api._require_full_result(orch, result)
    assert len(result['model_health']['fitted']) == 25
    assert result['support']['m21-xai-shap']['shap_values']
    assert result['support']['m24-quantized-edge']['input_dtype'] == 'int8'
    assert result['support']['m24-quantized-edge']['output_dtype'] == 'int8'
    assert result['support']['m24-quantized-edge']['size_bytes'] > 0
    # A second analysis must work too, without stale interpreter shapes or state.
    again = orch.analyse(batch, ctx, parallel=False)
    api._require_full_result(orch, again)
