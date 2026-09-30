"""Boundary/causality regressions, not claims of field accuracy."""
from copy import deepcopy
import os
from types import SimpleNamespace

import numpy as np
import pytest

from wtpm_platform.contracts import ModelOutput, TaskType, OperatingContext
from wtpm_platform.data import FeaturePipeline
from wtpm_platform.fusion import FusionEngine, FusionConfig
from wtpm_platform.orchestrator import build_default_registry, Orchestrator
from wtpm_platform.protocol import training_mask, window_training_mask, validate_output, hold_probabilities
from wtpm_platform.safety_gate import gate_command
from wtpm_platform.simulate import make_training_batch, make_batch, without_labels


@pytest.mark.parametrize('mid', build_default_registry().ids())
def test_all_25_reject_invalid_training_masks_before_loading_dependencies(mid):
    model = build_default_registry().get(mid)
    batch = SimpleNamespace(n_steps=5)
    for mask in (np.ones(4, bool), np.ones(5, int), np.zeros(5, bool)):
        with pytest.raises(ValueError, match='train_mask'):
            model.fit(batch, mask)
    assert not model.fitted


def test_feature_pipeline_prefix_is_independent_of_future():
    raw = make_batch(2, seed=17)
    raw.values[30:40, 0] = np.nan
    full = FeaturePipeline().transform(deepcopy(raw))
    short = deepcopy(raw)
    short.timestamps, short.values = raw.timestamps[:144], raw.values[:144]
    short.meta['mask'] = raw.meta['mask'][:144]
    prefix = FeaturePipeline().transform(short)
    np.testing.assert_allclose(prefix.values, full.values[:144])
    np.testing.assert_allclose(prefix.features, full.features[:144])
    np.testing.assert_array_equal(prefix.operating_state, full.operating_state[:144])
    np.testing.assert_allclose(prefix.vib_waveforms, full.vib_waveforms[:len(prefix.vib_index)])
    for win, end in zip(prefix.windows, prefix.window_index):
        np.testing.assert_array_equal(win[-1], prefix.values[end])
    mask = np.arange(full.n_steps) < 100
    mask[50] = False
    keep = window_training_mask(full, mask)
    for end in full.window_index[keep]:
        assert mask[end - 35:end + 1].all()


def test_probability_warmup_is_not_future_backfill():
    result = hold_probabilities(np.array([[0., 1.], [1., 0.]]), [3, 5], 7)
    np.testing.assert_allclose(result[:3], .5)
    np.testing.assert_array_equal(result[3:5], [[0, 1], [0, 1]])


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), 'broken', '', None, True, -1])
def test_software_safety_rejects_invalid_supplied_values(bad):
    assert not gate_command({'rpm': bad})['forwarded']


def test_safety_zero_and_alias_conflicts_cannot_hide_unsafe_values():
    assert gate_command({'rpm': 0, 'pitch_angle': 0})['forwarded']
    assert not gate_command({'rpm': 0, 'rotor_speed_rpm': 30})['forwarded']
    assert not gate_command({})['forwarded']


def output(mid='a', timestamps=None, **kwargs):
    return ModelOutput(model_id=mid, task=TaskType.ANOMALY_DETECTION,
                       turbine_id='test', timestamps=np.arange(3) if timestamps is None else timestamps,
                       **kwargs)


@pytest.mark.parametrize('field,value', [('anomaly_score', [0, np.nan, 1]),
    ('rul_hours', [1, -1, 0]), ('uncertainty', [1, 2]),
    ('probability', {'healthy': np.ones(3) * .9})])
def test_output_contract_rejects_invalid_success(field, value):
    with pytest.raises(ValueError):
        validate_output(output(**{field: value}))


@pytest.mark.parametrize('kind', ['anomaly', 'rul', 'probabilities'])
def test_fusion_rejects_shifted_timestamps_and_short_streams(kind):
    args = {'anomaly': {'anomaly_score': np.ones(3)},
            'rul': {'rul_hours': np.ones(3)},
            'probabilities': {'probability': {'healthy': np.ones(3)}}}[kind]
    method = getattr(FusionEngine(), 'fuse_' + kind)
    extra = [['healthy']] if kind == 'probabilities' else []
    with pytest.raises(ValueError, match='timestamps'):
        method([output(**args), output('b', np.arange(3) + 1, **args)], *extra)
    short = {k: ({'healthy': np.ones(2)} if k == 'probability' else np.ones(2)) for k in args}
    with pytest.raises(ValueError, match='timestamps'):
        method([output(**args), output('b', np.arange(2), **short)], *extra)


def test_missing_classifier_classes_abstain_instead_of_winning():
    o = output(probability={'healthy': np.full(3, .1), 'bearing_wear': np.full(3, .9)})
    p, labels, _ = FusionEngine().fuse_probabilities([o], ['healthy', 'bearing_wear', 'converter_fault'])
    assert np.all(labels == 'bearing_wear')
    assert np.all(p['converter_fault'] == 0)


def test_anomaly_consensus_does_not_dilute_startup_with_zeroes():
    fused = FusionEngine(FusionConfig(temporal_window=9)).fuse_anomaly([output(anomaly_score=np.full(3, 6.))])
    np.testing.assert_allclose(fused.score, 6.)
    assert fused.alarm.all()


def test_simulator_does_not_hide_runtime_errors(monkeypatch):
    import wt_pm_lstm.simulate as sim
    def fail(*args, **kwargs):
        raise ValueError('bad physical input')
    monkeypatch.setattr(sim, 'simulate_timeline', fail)
    with pytest.raises(ValueError, match='bad physical input'):
        make_batch(6)


def test_rul_proxy_uses_elapsed_time_and_excludes_future_labels():
    from wtpm_platform.adapters.prognostics import rul_proxy_hours
    b = SimpleNamespace(n_steps=4, timestamps=np.array([0, 1800, 7200, 10800]),
                        meta={'fault_kind_per_step': ['', '', 'bearing_wear', 'bearing_wear']})
    np.testing.assert_allclose(rul_proxy_hours(b), [2, 1.5, 0, 0])
    np.testing.assert_allclose(rul_proxy_hours(b, train_mask=np.array([True, True, False, False]))[:2], 400)


@pytest.mark.parametrize('mid', ['m10-random-forest', 'm11-xgboost-tabular', 'm14-isolation-forest', 'm17-mlp-rul'])
def test_excluded_future_rows_cannot_change_fitted_model(mid):
    batch = FeaturePipeline().transform(make_training_batch())
    mask = np.arange(batch.n_steps) < 640
    changed = deepcopy(batch)
    changed.values[~mask] += 100000
    changed.features[~mask] += 100000
    changed.meta['fault_kind_per_step'] = list(changed.meta['fault_kind_per_step'])
    for i in np.flatnonzero(~mask):
        changed.meta['fault_kind_per_step'][i] = 'converter_fault'
    models = [build_default_registry().get(mid) for _ in range(2)]
    if not models[0].available():
        pytest.skip(models[0]._unavailable_reason)
    for model, data in zip(models, (batch, changed)):
        model.fit(data, mask)
    test = FeaturePipeline().transform(without_labels(make_batch(2, 107)))
    a, b = [model.predict(test) for model in models]
    assert a.ok and b.ok, (a.error, b.error)
    for name in ('anomaly_score', 'rul_hours'):
        if getattr(a, name) is not None:
            np.testing.assert_allclose(getattr(a, name), getattr(b, name))
    for c in a.probability or {}:
        np.testing.assert_allclose(a.probability[c], b.probability[c])


@pytest.fixture(scope='module')
def full_models():
    if os.environ.get('WTPM_TEST_FULL') != '1':
        pytest.skip('requires pinned full model sources and dependencies')
    orch = Orchestrator(max_workers=1)
    batch = orch.prepare(make_training_batch())
    orch.fit(batch, OperatingContext(mode='research', has_labels=True), verbose=False)
    assert all(m['fitted'] for m in orch.registry.health_report())
    return orch, orch.prepare(without_labels(make_batch(2, seed=107)))


def test_informer_forecast_does_not_read_target_row(full_models):
    orch, batch = full_models
    model = orch.registry.get('m06-informer-forecast')
    a = model.predict(batch)
    changed = deepcopy(batch)
    changed.windows[:, -1, :] += 1000
    b = model.predict(changed)
    assert a.ok and b.ok
    np.testing.assert_allclose(a.extra['forecast_standardized'], b.extra['forecast_standardized'])
    assert not np.allclose(a.anomaly_score, b.anomaly_score)


def test_gnn_inference_uses_frozen_weights_and_consistent_node_units(full_models, monkeypatch):
    orch, batch = full_models
    model = orch.registry.get('m20-gnn-cascade')
    before = {k: v.clone() for k, v in model._net.state_dict().items()}
    monkeypatch.setattr(model, 'fit', lambda *args: pytest.fail('inference refitted GNN'))
    a, b = model.predict(batch), model.predict(batch)
    assert a.ok and b.ok
    for k in a.probability:
        np.testing.assert_array_equal(a.probability[k], b.probability[k])
    other = deepcopy(batch)
    other.turbine_id = 'other'
    fleet = model.predict_fleet([batch, other], {}, np.array([[0, 1], [1, 0]]))
    assert fleet.ok
    for k, v in model._net.state_dict().items():
        assert (v == before[k]).all()
    nodes = model._node_features([batch], {'WT-01': np.full(batch.n_steps, 9999)})
    assert nodes[0, 0] == pytest.approx(batch.channel('bearing_vib_rms_mm_s')[-144:].mean())


def test_particle_filter_prefix_is_causal_and_repeatable(full_models):
    orch, batch = full_models
    model = orch.registry.get('m16-particle-filter-rul')
    batch = deepcopy(batch)
    batch.meta['rul_observations'] = np.linspace(200, 30, batch.n_steps)
    a = model.predict(batch)
    batch.meta['rul_observations'][100:] = 9999
    b = model.predict(batch)
    assert a.ok and b.ok
    np.testing.assert_array_equal(a.rul_hours[:100], b.rul_hours[:100])
    np.testing.assert_array_equal(b.rul_hours, model.predict(batch).rul_hours)


def test_physics_penalty_has_gradient_and_consistent_kw_units(full_models):
    import torch
    from wtpm_platform.base import load_repo_module
    loss = load_repo_module('wt-pm-pg-bnn-wind-turbine').PhysicsGuidedLoss(.1)
    pred = torch.tensor([0.], requires_grad=True)
    # 10 kNm at 60 RPM -> 59.0619 kW electrical. mu=20 kW, sd=10 kW.
    torque_kw_scaled = torch.tensor([.94 * 10000 / 1000 / 10])
    value = loss(pred, pred.detach(), pred + 20 / 10, torque_kw_scaled, torch.tensor([60.]))
    value.backward()
    expected_kw = .94 * 10 * 2 * 3.14159
    assert pred.grad.item() == pytest.approx(.2 * (20 - expected_kw) / 10, rel=1e-5)


def test_nt_xent_uses_negative_pairs(full_models):
    import torch
    from wtpm_platform.adapters.classification import nt_xent
    vectors = torch.eye(8, requires_grad=True)
    aligned = nt_xent(vectors, vectors)
    scrambled = nt_xent(vectors, vectors.roll(1, dims=0))
    assert aligned < scrambled
    aligned.backward()
    assert torch.isfinite(vectors.grad).all()


def test_rul_metric_extreme_errors_do_not_overflow():
    from wtpm_platform.evaluation import rul_metrics
    assert np.isfinite(rul_metrics(np.array([0., 100000.]), np.array([100000., 0.]))['phm_score_mean'])


def test_safety_manager_does_not_authorize_missing_telemetry():
    from wtpm_platform.engines import SafetyManager
    batch = FeaturePipeline().transform(make_batch(2))
    idx = list(batch.channel_names).index('rotor_speed_rpm')
    batch.meta['mask'][-1, idx] = False
    diagnosis = SimpleNamespace(fault='healthy', confidence=1.)
    result = SafetyManager(build_default_registry()).evaluate(batch, diagnosis, None)
    assert result['decision'] != 'CONTINUE'
    assert any('unavailable' in reason for reason in result['reasons'])
