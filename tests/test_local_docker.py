"""Validate the local runtime check without heavy optional ML dependencies."""
from copy import deepcopy
import io
import json
from pathlib import Path
import runpy

import pytest

module = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'deployment/check_local.py'))
validate_health = module['validate_health']
check = module['check']


def healthy():
    ids = [f'm{i:02d}' for i in range(1, 26)]
    return {
        'status': 'ok', 'profile': 'full',
        'counts': dict.fromkeys(('registered', 'available', 'fitted', 'ran'), 25),
        'models': [{'model_id': mid, 'available': True, 'fitted': True} for mid in ids],
        'last_execution': {'ran': ids, 'errors': {}, 'fallbacks': {}},
    }


def test_local_health_requires_actual_execution():
    h = healthy()
    assert len(validate_health(h)) == 25


@pytest.mark.parametrize('case', [
    'demo', 'starting', 'count', 'duplicate', 'unfitted', 'unavailable',
    'missing', 'error', 'fallback',
])
def test_local_health_rejects_partial_or_misleading_success(case):
    h = healthy()
    if case == 'demo':
        h['profile'] = 'demo'
    elif case == 'starting':
        h['status'] = 'starting'
    elif case == 'count':
        h['counts']['fitted'] = 6
    elif case == 'duplicate':
        h['models'][-1] = deepcopy(h['models'][0])
    elif case == 'unfitted':
        h['models'][0]['fitted'] = False
    elif case == 'unavailable':
        h['models'][0]['available'] = False
    elif case == 'missing':
        h['last_execution']['ran'].pop()
    elif case == 'error':
        h['last_execution']['errors'] = {'m01': 'inference failed'}
    else:
        h['last_execution']['fallbacks'] = {'m01': 'm05'}
    with pytest.raises(ValueError):
        validate_health(h)


def test_local_http_check_runs_analysis(monkeypatch):
    calls = []
    h = healthy()
    result = {
        'model_health': h['last_execution'],
        'support': {
            'm21-xai-shap': {'shap_values': {'vibration': 0.1}},
            'm24-quantized-edge': {'input_dtype': 'int8', 'output_dtype': 'int8', 'size_bytes': 100},
        },
    }
    def fake_urlopen(req, timeout):
        calls.append((req.full_url, req.get_method(), req.data))
        assert timeout == 120
        body = {'/ready': {'status': 'ready'}, '/health': h, '/analyse': result}[req.selector]
        return io.BytesIO(json.dumps(body).encode())
    monkeypatch.setitem(check.__globals__, 'urlopen', fake_urlopen)
    check('http://127.0.0.1:8100/', analyse=True)
    assert calls[-1] == ('http://127.0.0.1:8100/analyse', 'POST', b'{}')
    result['support']['m24-quantized-edge']['output_dtype'] = 'float32'
    with pytest.raises(ValueError, match='INT8'):
        check('http://127.0.0.1:8100', analyse=True)
