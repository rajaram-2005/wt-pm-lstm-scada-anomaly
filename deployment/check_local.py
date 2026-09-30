"""Verify a running local full-model service (stdlib only, no external calls).

Usage: python deployment/check_local.py [--analyse]
"""
import argparse
import json
from urllib.request import Request, urlopen


EXPECTED_MODELS = 25


def validate_health(health):
    if health.get('status') != 'ok' or health.get('profile') != 'full':
        raise ValueError('The full profile is not ready: ' + str(health.get('error')))
    counts = health.get('counts', {})
    if any(counts.get(key) != EXPECTED_MODELS for key in ('registered', 'available', 'fitted', 'ran')):
        raise ValueError(f'Expected 25 registered/available/fitted/ran models, got {counts}')
    models = health.get('models', [])
    ids = {row['model_id'] for row in models}
    if len(models) != EXPECTED_MODELS or len(ids) != EXPECTED_MODELS:
        raise ValueError('Model registry must contain 25 unique entries')
    if any(not row.get('available') or not row.get('fitted') for row in models):
        raise ValueError('Some models are unavailable or unfitted')
    last = health.get('last_execution', {})
    if set(last.get('ran', [])) != ids or last.get('errors') or last.get('fallbacks'):
        raise ValueError('Last execution was incomplete or used errors/fallbacks')
    return ids


def check(base_url, analyse=False):
    def request(path, payload=None):
        req = Request(base_url.rstrip('/') + path,
                      data=None if payload is None else json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json'})
        with urlopen(req, timeout=120) as response:
            return json.load(response)

    if request('/ready').get('status') != 'ready':
        raise ValueError('Service is not ready')
    health = request('/health')
    ids = validate_health(health)
    if analyse:
        result = request('/analyse', {})
        execution = result.get('model_health', {})
        if (set(execution.get('ran', [])) != ids or execution.get('errors')
                or execution.get('fallbacks')):
            raise ValueError('Analysis did not execute all 25 models successfully')
        support = result.get('support', {})
        edge = support.get('m24-quantized-edge', {})
        if not support.get('m21-xai-shap', {}).get('shap_values'):
            raise ValueError('SHAP execution evidence is missing')
        if (edge.get('input_dtype') != 'int8' or edge.get('output_dtype') != 'int8'
                or edge.get('size_bytes', 0) <= 0):
            raise ValueError('INT8 execution evidence is missing')
    print('Local full-model service verified: ' + json.dumps(health['counts']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8100')
    parser.add_argument('--analyse', action='store_true')
    args = parser.parse_args()
    check(args.base_url, args.analyse)
