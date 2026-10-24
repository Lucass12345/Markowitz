"""Pinned, frame-level financial decisions and their worker adapter."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.optimization import inputs_builder
from src.optimization.errors import OptimizationInputError
from src.platform.optimization import OptimizationService, optimize_frame


@pytest.fixture
def portfolio():
    frame = pd.DataFrame(np.random.default_rng(81).normal(.0006, .011, (48, 4)),
                         index=pd.bdate_range('2024-01-01', periods=48), columns=list('ABCD'))
    frame.attrs.update(dataset_version='usd-v1', base_currency='USD')
    metadata = pd.DataFrame({'ticker': list('ABCD'), 'block': ['equity', 'equity', 'bonds', 'bonds']})
    config = {'dataset_version': 'usd-v1', 'targets': {'equity': .6, 'bonds': .4},
              'max_weight': .4, 'min_history': 12, 'min_overlap': 10, 'calculate_frontiers': False}
    return frame, metadata, config


def forecasts(frame, family='lstm', **changes):
    risk = family == 'garch'
    return pd.DataFrame([{'ticker': ticker, 'dataset_version': frame.attrs['dataset_version'],
        'base_currency': 'USD', 'as_of': frame.index[-1].isoformat(),
        'prediction_units': 'daily_log_volatility' if risk else 'daily_log_return',
        'vol_pred' if risk else 'mu_pred': .023 if risk else .002,
        'dir_acc_val': .63, 'amp_ratio': .4, 'use_for_optimizer': True, **changes} for ticker in frame])


def assert_constraints(result):
    weights = result['blocks']['weights_series']
    assert weights.sum() == pytest.approx(1, abs=1e-7)
    assert weights.min() >= -1e-8
    assert weights.max() <= .4 + 1e-7
    assert weights[['A', 'B']].sum() == pytest.approx(.6, abs=1e-7)
    assert weights[['C', 'D']].sum() == pytest.approx(.4, abs=1e-7)


@pytest.mark.parametrize('mode', ['classic', 'lstm', 'mlp', 'garch', 'combined'])
def test_pure_decision_never_loads_current_production_signals(portfolio, monkeypatch, mode):
    frame, metadata, config = portfolio
    original = frame.copy(deep=True)
    def forbidden(*args, **kwargs):
        raise AssertionError('A pinned frame decision read a current model source')
    monkeypatch.setattr(inputs_builder, '_pinned_signals', forbidden)
    result = optimize_frame(frame, metadata, {**config, 'model_type': mode})
    assert_constraints(result)
    pd.testing.assert_frame_equal(frame, original)
    assert result['inputs']['base_currency'] == 'USD'
    assert result['portfolio_risk']['return_type'] == 'daily_log'
    for item in result['diagnostics']['risk']['signals']:
        assert item['source'] == 'historical'
    assert np.linalg.eigvalsh(result['estimates']['covariance']).min() > 0


def test_combined_uses_daily_signals_and_validation_only(portfolio):
    frame, metadata, config = portfolio
    returns = forecasts(frame)
    returns.loc[returns.ticker.eq('A'), ['dir_acc_val', 'mu_pred']] = [.52, .5]
    returns['dir_acc_test'] = 1.
    result = optimize_frame(frame, metadata, {**config, 'model_type': 'combined'}, returns, forecasts(frame, 'garch'))
    assert_constraints(result)
    mu, covariance = result['estimates']['mu'], result['estimates']['covariance']
    assert mu.A == pytest.approx(frame.A.mean())
    assert mu.B == pytest.approx(.002)
    assert np.sqrt(np.diag(covariance)) == pytest.approx(np.repeat(.023, 4))
    signals = result['diagnostics']['risk']['signals']
    assert signals[0]['reason'] == 'validation_direction_below_threshold'
    assert all(item['source'] == 'ai' for item in result['diagnostics']['risk']['risk_signals'])


@pytest.mark.parametrize('changes,reason', [({'as_of': '2025-01-01'}, 'future'),
    ({'as_of': '2020-01-01'}, 'stale'), ({'use_for_optimizer': False}, 'producer_validation')])
def test_signal_policy_records_rejection_and_required_policy_fails(portfolio, changes, reason):
    frame, metadata, config = portfolio
    predictions = forecasts(frame, **changes)
    result = optimize_frame(frame, metadata, {**config, 'model_type': 'lstm'}, predictions)
    assert all(reason in item['reason'] for item in result['diagnostics']['risk']['signals'])
    pd.testing.assert_series_equal(result['estimates']['mu'], frame.mean())
    with pytest.raises(OptimizationInputError, match=reason):
        optimize_frame(frame, metadata, {**config, 'model_type': 'lstm', 'prediction_fallback': 'required'}, predictions)


def test_forecast_contract_and_combined_dates_are_explicit(portfolio):
    frame, metadata, config = portfolio
    for changes, message in [({'dataset_version': 'other'}, 'dataset_version'),
                             ({'base_currency': 'ARS'}, 'USD'),
                             ({'prediction_units': 'five_day_log_return'}, 'units')]:
        with pytest.raises(OptimizationInputError, match=message):
            optimize_frame(frame, metadata, {**config, 'model_type': 'lstm'}, forecasts(frame, **changes))
    with pytest.raises(OptimizationInputError, match='prediction_units'):
        optimize_frame(frame, metadata, {**config, 'model_type': 'lstm'}, forecasts(frame).drop(columns='prediction_units'))
    with pytest.raises(OptimizationInputError, match='different as_of'):
        optimize_frame(frame, metadata, {**config, 'model_type': 'combined'},
                       forecasts(frame), forecasts(frame, 'garch', as_of=frame.index[-2].isoformat()))


def test_cutoff_covariance_and_constraints_are_shared_engine_results(portfolio):
    frame, metadata, config = portfolio
    cutoff = frame.index[29]
    reference = optimize_frame(frame.iloc[:30], metadata, config)
    changed = frame.copy()
    changed.iloc[30:] = 1.
    actual = optimize_frame(changed, metadata, {**config, 'as_of': cutoff.isoformat(), 'covariance_estimator': 'historical'})
    pd.testing.assert_series_equal(actual['blocks']['weights_series'], reference['blocks']['weights_series'])
    pd.testing.assert_frame_equal(actual['estimates']['covariance'], reference['estimates']['covariance'])
    assert actual['inputs']['as_of'] == cutoff.isoformat()
    shrinkage = optimize_frame(frame, metadata, {**config, 'covariance_estimator': 'ledoit_wolf'})
    assert shrinkage['diagnostics']['risk']['estimator'] == 'ledoit_wolf'
    assert_constraints(shrinkage)
    with pytest.raises(OptimizationInputError, match='capacity|infeasible|feasib', ):
        optimize_frame(frame, metadata, {**config, 'max_weight': .2})


def test_quality_filter_can_make_block_allocation_infeasible(portfolio):
    frame, metadata, config = portfolio
    frame.loc[frame.index[5:], 'A'] = np.nan
    with pytest.raises(OptimizationInputError, match='capacity|infeasible|feasib'):
        optimize_frame(frame, metadata, config)
    frame.attrs['base_currency'] = 'ARS'
    with pytest.raises(OptimizationInputError, match='USD'):
        optimize_frame(frame, metadata, config)


class Snapshot:
    id = 'dataset-id'
    version = 'usd-v1'
    def __init__(self, frame, metadata):
        self.frame, self.metadata = frame, metadata
    def load_returns(self):
        return self.frame.copy(deep=True)
    def load_metadata(self):
        return self.metadata.copy(deep=True)


class Datasets:
    def __init__(self, snapshot):
        self.snapshot = snapshot
    def resolve(self, version):
        assert version in {'usd-v1', 'dataset-id'}
        return {'id': self.snapshot.id, 'version': self.snapshot.version}
    def materialize(self, version, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        return self.snapshot


def context(tmp_path, store=None):
    updates = []
    return SimpleNamespace(job_id='job-id', dataset_version='usd-v1', temp_dir=tmp_path, actor='test',
        database=None, store=store, progress=lambda *values: updates.append(values),
        check_cancelled=lambda: None, updates=updates)


def test_worker_uses_pinned_registry_artifacts_and_keeps_publication_for_caller(portfolio, tmp_path):
    frame, metadata, config = portfolio
    bundle = tmp_path / 'source-bundle'
    bundle.mkdir()
    forecasts(frame).to_csv(bundle / 'predictions.csv', index=False)
    models = [{'id': f'model-{ticker}', 'family': 'lstm', 'ticker': ticker, 'state': 'validated',
               'dataset_version': 'usd-v1', 'training_cutoff': str(frame.index[-1]),
               'horizon': 5, 'artifact_uri': 'verified://models/lstm'} for ticker in frame]
    class Registry:
        def resolve_active(self, *args, **kwargs):
            raise AssertionError('Pinned job must never resolve current active models')
        def get(self, *args):
            raise AssertionError('Full pinned records should be sufficient')
    calls = []
    store = SimpleNamespace(materialize=lambda uri, destination: calls.append(uri) or bundle)
    ctx = context(tmp_path / 'job', store)
    service = OptimizationService(Datasets(Snapshot(frame, metadata)), Registry())
    result = service.run({**config, 'model_type': 'lstm', 'pinned_return_models': models}, ctx)
    assert result['model_ids'] == [model['id'] for model in models]
    assert calls == ['verified://models/lstm']
    artifacts = Path(result['artifacts_dir'])
    assert artifacts.is_relative_to(ctx.temp_dir)
    assert {'result.json', 'weights.csv', 'covariance.csv', 'block_exposures.csv'} <= {p.name for p in artifacts.iterdir()}
    report = json.loads((artifacts / 'result.json').read_text())
    assert report['model_references']['returns'] == models
    assert all(signal['source'] == 'ai' for signal in report['diagnostics']['risk']['signals'])


def test_worker_rejects_dataset_mismatch_and_future_training(portfolio, tmp_path):
    frame, metadata, config = portfolio
    service = OptimizationService(Datasets(Snapshot(frame, metadata)), object())
    with pytest.raises(OptimizationInputError, match='different dataset'):
        service.run({**config, 'dataset_version': 'other'}, context(tmp_path))
    model = {'id': 'future-model', 'family': 'lstm', 'ticker': 'A', 'state': 'validated',
             'dataset_version': 'usd-v1', 'training_cutoff': '2025-01-01', 'artifact_uri': 'verified://future'}
    with pytest.raises(OptimizationInputError, match='trained after'):
        service.run({**config, 'model_type': 'lstm', 'pinned_return_models': [model]}, context(tmp_path))


def test_worker_honors_frozen_universe_without_reading_current_definition(portfolio, tmp_path):
    frame, metadata, config = portfolio
    frozen = {'assets': [{'ticker': ticker, 'block': block, 'enabled': True}
                        for ticker, block in zip('ABCD', ['bonds', 'bonds', 'equity', 'equity'])] +
                        [{'ticker': 'MISSING', 'enabled': False}]}
    class Universes:
        def get(self, *args):
            raise AssertionError('Queued job read an edited current universe')
    service = OptimizationService(Datasets(Snapshot(frame, metadata)), object(), Universes())
    result = service.run({**config, 'universe_id': 'mutable-id', 'universe_snapshot': frozen}, context(tmp_path))
    weights = result['result']['blocks']['weights_series']
    assert weights['A'] + weights['B'] == pytest.approx(.4, abs=1e-7)
    assert weights['C'] + weights['D'] == pytest.approx(.6, abs=1e-7)
    assert result['result']['inputs']['universe_snapshot'] == frozen
