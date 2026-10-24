"""Walk-forward financial evidence uses the serving engine with past-only fits."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.backtesting import BacktestConfig, walk_forward
from src.backtesting.strategies import ConstrainedModelStrategy, matched_benchmark_weights
from src.optimization import inputs_builder
from src.platform.backtesting import BacktestService, STRATEGIES


@pytest.fixture
def research():
    data = pd.DataFrame(np.random.default_rng(18).normal(.0003, .009, (336, 4)), columns=list('ABCD'),
                        index=pd.bdate_range('2024-02-01', periods=336))
    data.attrs.update(dataset_version='usd-v1', base_currency='USD')
    metadata = pd.DataFrame({'ticker': list('ABCD'), 'block': ['equity', 'equity', 'bonds', 'bonds']})
    optimization = {'targets': {'equity': .6, 'bonds': .4}, 'max_weight': .4,
                    'min_history': 12, 'min_overlap': 10, 'prediction_fallback': 'required'}
    config = {'dataset_version': 'usd-v1', 'train_window': 318, 'min_train': 300, 'rebalance_every': 5,
              'oos_sessions': 10, 'commission_bps': 2., 'slippage_bps': 3., 'transaction_cost_bps': 1.,
              'optimization': optimization, 'strategies': list(STRATEGIES)}
    return data, metadata, config


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
    def __init__(self, frame, metadata):
        self.snapshot = Snapshot(frame, metadata)
    def resolve(self, version):
        assert version == self.snapshot.version
        return {'id': self.snapshot.id, 'version': self.snapshot.version}
    def materialize(self, version, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        return self.snapshot


def context(directory):
    updates = []
    return SimpleNamespace(job_id='backtest-job', dataset_version='usd-v1', temp_dir=directory,
        store=object(), database=None, actor='test', check_cancelled=lambda: None,
        progress=lambda *args: updates.append(args), updates=updates)


class PastOnlyModels:
    """Explicit injected trainer; the portfolio engine and simulation remain real."""
    def __init__(self):
        self.calls = []
    def train_on_frame(self, family, frame, version, config, child):
        self.calls.append((family, frame.copy(deep=True), child.temp_dir, child.job_id))
        child.check_cancelled()
        child.progress(50, 'training', 'synthetic deterministic fit')
        artifact_dir = Path(child.temp_dir) / 'artifacts'
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / 'checkpoint.json').write_text(json.dumps({'training_cutoff': str(frame.index.max())}))
        rows = [{'ticker': ticker, 'dataset_version': version, 'base_currency': 'USD',
                 'as_of': frame[ticker].dropna().index[-1].isoformat(),
                 'prediction_units': 'daily_log_volatility' if family == 'garch' else 'daily_log_return',
                 'vol_pred' if family == 'garch' else 'mu_pred': float(frame[ticker].std()) if family == 'garch' else float(frame[ticker].mean()),
                 'dir_acc_val': .65, 'amp_ratio': .4, 'use_for_optimizer': True} for ticker in frame]
        pd.DataFrame(rows).to_csv(artifact_dir / 'predictions.csv', index=False)
        return {'predictions': rows, 'artifacts_dir': str(artifact_dir), 'metrics': {'tickers': len(frame)},
                'models': [{'id': f'{child.job_id}:{ticker}', 'ticker': ticker, 'family': family} for ticker in frame]}


def assert_target(weights):
    assert sum(weights.values()) == pytest.approx(1, abs=1e-7)
    assert min(weights.values()) >= -1e-8
    assert max(weights.values()) <= .4 + 1e-7
    assert weights['A'] + weights['B'] == pytest.approx(.6, abs=1e-7)
    assert weights['C'] + weights['D'] == pytest.approx(.4, abs=1e-7)


def test_all_strategies_publish_comparable_evidence_with_isolated_past_only_models(research, tmp_path, monkeypatch):
    data, metadata, config = research
    # A production return-model setting must not turn every comparison into MLP.
    config['optimization']['return_model'] = 'mlp'
    models = PastOnlyModels()
    def forbidden(*args, **kwargs):
        raise AssertionError('Historical decision attempted a current production signal read')
    monkeypatch.setattr(inputs_builder, '_pinned_signals', forbidden)
    ctx = context(tmp_path)
    result = BacktestService(Datasets(data, metadata), models).run(config, ctx)
    output = Path(result['artifacts_dir'])
    assert output.is_relative_to(tmp_path)
    assert set(result['metrics']) == set(STRATEGIES)
    assert result['dataset_version'] == 'usd-v1'
    assert result['result']['metrics'] == result['metrics']
    curves = pd.read_csv(output / 'wealth_curves.csv', index_col='date', parse_dates=True)
    assert curves.index.equals(data.index[-10:].rename('date'))
    assert not curves.isna().any().any()
    for name in STRATEGIES:
        decisions = json.loads((output / f'{name}_decisions.json').read_text())
        assert_target(decisions[0]['weights'])
        for decision in decisions:
            assert pd.Timestamp(decision['train_end']) < pd.Timestamp(decision['holding_start'])
            if name != 'buy_and_hold':
                assert_target(decision['weights'])
            if name not in {'equal_weight', 'buy_and_hold'}:
                assert decision['diagnostics']['model_type'] == ConstrainedModelStrategy.MODES[name]
        assert result['metrics'][name]['n_observations'] == 10
        assert result['metrics'][name]['currency'] == 'USD'
    assert len(models.calls) == 10
    assert len({str(call[2]) for call in models.calls}) == len(models.calls)
    assert len({call[3] for call in models.calls}) == len(models.calls)
    for family, history, directory, job_id in models.calls:
        assert history.index.max() in {data.index[-11], data.index[-6]}
        assert len(history) == 318
        assert history.attrs['dataset_version'] == 'usd-v1'
        assert directory.is_relative_to(output)
        assert (directory / 'artifacts' / 'checkpoint.json').exists()
    assert set(result['differences_vs_equal_weight']) == set(STRATEGIES) - {'equal_weight'}
    assert (output / 'report.json').exists() and (output / 'configuration.json').exists()
    percentages = [float(values[0]) for values in ctx.updates]
    assert all(left <= right for left, right in zip(percentages, percentages[1:]))


def test_future_perturbations_do_not_change_earlier_combined_decision_or_path(research, tmp_path):
    data, metadata, config = research
    config['strategies'] = ['combined', 'equal_weight']
    first = BacktestService(Datasets(data, metadata), PastOnlyModels()).run(config, context(tmp_path / 'first'))
    changed = data.copy(deep=True)
    changed.iloc[-5:] = .09
    second = BacktestService(Datasets(changed, metadata), PastOnlyModels()).run(config, context(tmp_path / 'second'))
    before = json.loads((Path(first['artifacts_dir']) / 'combined_decisions.json').read_text())
    after = json.loads((Path(second['artifacts_dir']) / 'combined_decisions.json').read_text())
    assert before[0] == after[0]
    first_path = pd.read_csv(Path(first['artifacts_dir']) / 'combined_path.csv')
    second_path = pd.read_csv(Path(second['artifacts_dir']) / 'combined_path.csv')
    pd.testing.assert_frame_equal(first_path.iloc[:5], second_path.iloc[:5])


def test_costs_and_slippage_use_actual_trades_and_buyhold_never_rebalances():
    data = pd.DataFrame({'A': [0.] * 4 + [np.log(1.1)] * 6, 'B': [0.] * 10},
                        index=pd.bdate_range('2024-01-01', periods=10))
    config = BacktestConfig(4, 4, 2, commission_bps=2, slippage_bps=3, transaction_cost_bps=5)
    result = walk_forward(data, config=config, strategy='buy_and_hold')
    expected = (.5 * 1.1 ** 6 + .5) * .999
    assert result['path'].wealth.iloc[-1] == pytest.approx(expected)
    assert result['decisions'][0]['transaction_cost_fraction'] == pytest.approx(.001)
    assert all(decision['traded_notional'] == pytest.approx(0) for decision in result['decisions'][1:])
    assert result['metrics']['turnover_total'] == pytest.approx(1.)
    assert result['decisions'][1]['weights']['A'] > .5


def test_oos_window_cuts_input_before_model_training(research, tmp_path):
    data, metadata, config = research
    config.update(strategies=['lstm', 'equal_weight'], oos_end=str(data.index[-4]), oos_sessions=6)
    models = PastOnlyModels()
    result = BacktestService(Datasets(data, metadata), models).run(config, context(tmp_path))
    assert pd.Timestamp(result['oos_end']) == data.index[-4]
    assert pd.Timestamp(result['oos_start']) == data.index[-9]
    assert all(history.index.max() < data.index[-4] for _, history, _, _ in models.calls)
    assert all(metrics['n_observations'] == 6 for metrics in result['metrics'].values())


@pytest.mark.parametrize('request_form', ['model', 'complete_dump', 'api_dump'])
def test_public_request_model_parameters_and_date_aliases_reach_every_fold(research, tmp_path, request_form):
    from api.schemas.platform import BacktestRequest
    data, metadata, config = research
    parameters = {'lstm': {'epochs': 2, 'seed': 19}, 'garch': {'validation_days': 3}}
    request = BacktestRequest(dataset_version='usd-v1', train_window=318, min_train=300,
        optimization=config['optimization'], strategies=['combined'],
        start=str(data.index[-8].date()), end=str(data.index[-2].date()), model_parameters=parameters)
    payload = {'model': request, 'complete_dump': request.model_dump(),
               'api_dump': request.model_dump(exclude_none=True)}[request_form]
    class ParameterAwareModels(PastOnlyModels):
        def train_on_frame(self, family, frame, version, settings, child):
            assert settings == {'start_date': 'full', **parameters[family], 'dataset_version': version}
            return super().train_on_frame(family, frame, version, settings, child)
    models = ParameterAwareModels()
    result = BacktestService(Datasets(data, metadata), models).run(payload, context(tmp_path))
    assert {call[0] for call in models.calls} == {'lstm', 'garch'}
    assert result['configuration']['models'] == parameters
    assert pd.Timestamp(result['oos_start']) == data.index[-8]
    assert pd.Timestamp(result['oos_end']) == data.index[-2]
    assert result['metrics']['combined']['n_observations'] == 7


def test_explicit_models_remain_compatible_and_empty_strategy_selection_is_rejected(research, tmp_path):
    data, metadata, config = research
    config.update(strategies=['mlp'], models={'mlp': {'seed': 13}}, model_parameters={'mlp': {'seed': 99}})
    class LegacyModels(PastOnlyModels):
        def train_on_frame(self, family, frame, version, settings, child):
            assert settings['seed'] == 13
            return super().train_on_frame(family, frame, version, settings, child)
    BacktestService(Datasets(data, metadata), LegacyModels()).run(config, context(tmp_path / 'legacy'))
    config['strategies'] = []
    with pytest.raises(ValueError, match='unique backtest strategies'):
        BacktestService(Datasets(data, metadata), PastOnlyModels()).run(config, context(tmp_path / 'empty'))


@pytest.mark.parametrize('frozen_universe', [False, True])
def test_nested_tickers_select_one_common_universe_for_models_and_benchmarks(research, tmp_path, frozen_universe):
    data, metadata, config = research
    config['strategies'] = ['mlp', 'equal_weight', 'buy_and_hold']
    config['optimization'].update(tickers=['A', 'C'], max_weight=.6)
    expected = {'A', 'C'}
    if frozen_universe:
        config['universe_snapshot'] = {'assets': [{'ticker': ticker, 'enabled': True} for ticker in 'ABCD']}
        expected = set('ABCD')
    models = PastOnlyModels()
    result = BacktestService(Datasets(data, metadata), models).run(config, context(tmp_path))
    assert all(set(history.columns) == expected for _, history, _, _ in models.calls)
    assert set(result['configuration']['optimization']['tickers']) == expected
    for strategy in config['strategies']:
        decisions = json.loads((Path(result['artifacts_dir']) / f'{strategy}_decisions.json').read_text())
        assert all(set(decision['weights']) == expected for decision in decisions)


def test_garch_calendar_reserves_train_validation_and_test_before_first_decision(research, tmp_path):
    data, metadata, config = research
    config.update(strategies=['gjr_garch', 'equal_weight'], min_train=252, train_window=300, oos_sessions=None)
    models = PastOnlyModels()
    result = BacktestService(Datasets(data.iloc[:296], metadata), models).run(config, context(tmp_path))
    assert result['training_requirements'] == {'requested_min_train': 252, 'effective_min_train': 292,
                                               'model_minimum_observations': {'garch': 292}}
    assert pd.Timestamp(result['oos_start']) == data.index[292]
    assert result['metrics']['gjr_garch']['n_observations'] == result['metrics']['equal_weight']['n_observations'] == 4
    assert len(models.calls[0][1]) == 292
    config['train_window'] = 280
    models = PastOnlyModels()
    with pytest.raises(ValueError, match='292.*train_window'):
        BacktestService(Datasets(data, metadata), models).run(config, context(tmp_path / 'short'))
    assert not models.calls


def test_custom_garch_holdout_lengths_control_effective_history(research, tmp_path):
    data, metadata, config = research
    config.update(strategies=['gjr_garch'], min_train=252, train_window=280, oos_sessions=None,
        model_parameters={'garch': {'parameters': {'min_obs': 260, 'validation_days': 4, 'test_days': 3}}})
    result = BacktestService(Datasets(data.iloc[:270], metadata), PastOnlyModels()).run(config, context(tmp_path))
    assert result['training_requirements']['effective_min_train'] == 267
    assert pd.Timestamp(result['oos_start']) == data.index[267]


def test_missing_realized_marks_are_never_silently_imputed(research, tmp_path):
    data, metadata, config = research
    config['strategies'] = ['equal_weight']
    data.loc[data.index[-8], 'A'] = np.nan
    with pytest.raises(ValueError, match='Missing realized return.*A'):
        BacktestService(Datasets(data, metadata)).run(config, context(tmp_path))
    assert not (tmp_path / 'backtest' / 'report.json').exists()


def test_enabled_universe_and_frozen_blocks_define_all_benchmark_weights(research, tmp_path):
    data, metadata, config = research
    config['strategies'] = ['equal_weight', 'buy_and_hold']
    config['universe_snapshot'] = {'assets': [{'ticker': ticker, 'block': block, 'enabled': 'true'}
        for ticker, block in zip('ABCD', ['bonds', 'bonds', 'equity', 'equity'])] +
        [{'ticker': 'UNAVAILABLE', 'enabled': 'false', 'block': 'equity'}]}
    result = BacktestService(Datasets(data, metadata)).run(config, context(tmp_path))
    decisions = json.loads((Path(result['artifacts_dir']) / 'equal_weight_decisions.json').read_text())
    assert decisions[0]['weights'] == pytest.approx({'A': .2, 'B': .2, 'C': .3, 'D': .3})
    assert result['configuration']['universe_snapshot'] == config['universe_snapshot']['assets']


def test_benchmark_eligibility_uses_nested_risk_config_and_never_future_observations(research):
    data, metadata, config = research
    controls = {**config['optimization'], 'risk_config': {'min_history': 12, 'min_overlap': 10}}
    controls.pop('min_history')
    controls.pop('min_overlap')
    weights = matched_benchmark_weights(data.iloc[:18], metadata, controls)
    assert_target(weights.to_dict())
    early = data.iloc[:18].copy()
    early.loc[early.index[4:], 'A'] = np.nan
    with pytest.raises(ValueError, match='capacity|infeasible|feasib'):
        matched_benchmark_weights(early, metadata, controls)


def test_fold_models_cannot_write_outside_backtest_artifacts(research, tmp_path):
    data, metadata, config = research
    class EscapingTrainer(PastOnlyModels):
        def train_on_frame(self, *args):
            trained = super().train_on_frame(*args)
            return {**trained, 'artifacts_dir': str(tmp_path / 'outside')}
    config['strategies'] = ['lstm']
    with pytest.raises(ValueError, match='isolated backtest workspace'):
        BacktestService(Datasets(data, metadata), EscapingTrainer()).run(config, context(tmp_path / 'job'))


def test_models_fit_only_assets_eligible_at_the_historical_decision(research, tmp_path):
    data, metadata, config = research
    history = data.iloc[:18].copy()
    history['NEW'] = np.nan
    history.loc[history.index[-3:], 'NEW'] = [.01, -.02, .005]
    metadata = pd.concat([metadata, pd.DataFrame({'ticker': ['NEW'], 'block': ['equity']})], ignore_index=True)
    models = PastOnlyModels()
    strategy = ConstrainedModelStrategy('mlp', metadata, config['optimization'], 'usd-v1', context(tmp_path), model_service=models)
    weights = strategy(history)
    assert list(models.calls[0][1].columns) == list('ABCD')
    assert 'NEW' not in weights
    assert_target(weights.to_dict())
    assert 'history' in weights.attrs['decision_diagnostics']['selection']['risk']['excluded_assets']['NEW']


def test_invalid_and_empty_oos_windows_fail_before_training(research, tmp_path):
    data, metadata, config = research
    config.update(oos_start='2030-01-01', strategies=['lstm'])
    models = PastOnlyModels()
    with pytest.raises(ValueError, match='two OOS'):
        BacktestService(Datasets(data, metadata), models).run(config, context(tmp_path))
    assert not models.calls
    with pytest.raises(ValueError, match='integers'):
        BacktestConfig(train_window=12.5)
    # The optimizer and realized excess return use the same annual-rate domain.
    assert BacktestConfig(rf_annual_simple=-.01).rf_annual_simple == -.01


def test_real_training_flows_into_shared_optimizer_for_every_model_family(tmp_path, monkeypatch):
    import torch
    from src.platform.models import ModelService
    data = pd.DataFrame({'AAA': np.random.default_rng(42).normal(.0005, .009, 340)},
                        index=pd.bdate_range('2023-01-02', periods=340))
    data.attrs.update(dataset_version='usd-v1', base_currency='USD')
    metadata = pd.DataFrame({'ticker': ['AAA'], 'block': ['equity']})
    config = {'dataset_version': 'usd-v1', 'train_window': 338, 'min_train': 252,
              'oos_sessions': 2, 'strategies': list(STRATEGIES),
              'optimization': {'targets': {'equity': 1.}, 'max_weight': 1., 'prediction_fallback': 'historical'},
              'models': {'lstm': {'device': 'cpu', 'seq_len': 8, 'epochs': 1, 'hidden_size': 4,
                                  'num_layers': 1, 'search': False, 'dropout': 0, 'batch_size': 64,
                                  'selected_features': ['return', 'lag_1'], 'start_date': 'full'},
                         'mlp': {'hidden_layers': [5], 'max_iter': 10, 'early_stopping': False,
                                 'lags': 3, 'short_window': 3, 'long_window': 8, 'start_date': 'full'},
                         'garch': {'validation_days': 2, 'test_days': 2}}}
    def forbidden(*args, **kwargs):
        raise AssertionError('Real fold training attempted a current signal read')
    monkeypatch.setattr(inputs_builder, '_pinned_signals', forbidden)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = BacktestService(Datasets(data, metadata), ModelService()).run(config, context(tmp_path))
    finally:
        torch.set_num_threads(previous)
    output = Path(result['artifacts_dir'])
    assert set(result['metrics']) == set(STRATEGIES)
    for strategy in ('lstm', 'mlp', 'gjr_garch', 'combined'):
        decisions = json.loads((output / f'{strategy}_decisions.json').read_text())
        training = decisions[0]['diagnostics']['training']
        assert {record['family'] for record in training} == set(ConstrainedModelStrategy.FAMILIES[strategy])
        for record in training:
            assert pd.Timestamp(record['train_cutoff']) == data.index[-3]
            checkpoint = output / record['artifacts']
            assert (checkpoint / 'models.json').is_file()
            assert all((checkpoint / model['prediction_file']).is_file() for model in record['models'])
        assert result['metrics'][strategy]['n_observations'] == 2


def test_real_garch_default_holdouts_fit_at_the_effective_292_session_boundary(tmp_path):
    from src.platform.models import ModelService
    data = pd.DataFrame({'AAA': np.random.default_rng(42).normal(.0005, .009, 294)},
                        index=pd.bdate_range('2020-01-02', periods=294))
    data.attrs.update(dataset_version='usd-v1', base_currency='USD')
    metadata = pd.DataFrame({'ticker': ['AAA'], 'block': ['equity']})
    config = {'dataset_version': 'usd-v1', 'train_window': 300, 'min_train': 252,
              'strategies': ['gjr_garch', 'equal_weight'],
              'optimization': {'targets': {'equity': 1.}, 'max_weight': 1., 'prediction_fallback': 'historical'}}
    result = BacktestService(Datasets(data, metadata), ModelService()).run(config, context(tmp_path))
    assert result['training_requirements']['effective_min_train'] == 292
    assert pd.Timestamp(result['oos_start']) == data.index[292]
    decisions = json.loads((Path(result['artifacts_dir']) / 'gjr_garch_decisions.json').read_text())
    model = decisions[0]['diagnostics']['training'][0]['models'][0]
    assert model['metrics']['val']['n'] == model['metrics']['test']['n'] == 20
    assert result['metrics']['gjr_garch']['n_observations'] == 2
