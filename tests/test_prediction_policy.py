import numpy as np
import pandas as pd
import pytest
from src.optimization import inputs_builder as builder
from src.optimization.covariance import RiskConfig
from src.optimization.errors import OptimizationInputError


@pytest.fixture
def data():
    returns = pd.DataFrame({'A': [.01,-.01,.02,-.005,.003,.001], 'B': [-.02,.01,.005,-.01,.002,.01]},
                           index=pd.bdate_range('2024-01-01', periods=6))
    returns.attrs['dataset_version'] = 'pinned-usd'
    return returns


def signal(**changes):
    return {'ticker':'A','mu_pred':.004,'vol_pred':.03,'dataset_version':'pinned-usd','base_currency':'USD',
            'as_of':'2024-01-08','dir_acc_val':.6,'dir_acc_test':0.,'amp_ratio':.4, **changes}


def build(monkeypatch, data, rows, mode='ai_returns_lstm', **kwargs):
    monkeypatch.setattr(builder, '_load_signals', lambda mode: pd.DataFrame(rows))
    return builder.build_model_inputs(mode, data, RiskConfig(4,2), **kwargs)


def test_missing_keeps_universe_and_validation_selects_signal(monkeypatch, data):
    mu, cov, tickers = build(monkeypatch, data, [signal()])
    assert tickers == ['A','B']
    assert mu.A == .004 and mu.B == data.B.mean()
    assert cov.shape == (2,2)
    signals = mu.attrs['input_diagnostics']['signals']
    assert [s['source'] for s in signals] == ['ai','historical']
    assert signals[1]['reason'] == 'missing_prediction'
    # Test direction cannot rescue a rejected validation signal.
    mu, _, _ = build(monkeypatch, data, [signal(dir_acc_val=.1,dir_acc_test=1.)])
    assert mu.A == data.A.mean()


@pytest.mark.parametrize('changes,reason', [({'mu_pred':np.inf},'non_finite'), ({'mu_pred':'invalid'},'invalid_numeric'),
    ({'as_of':'2023-12-01'},'stale'), ({'as_of':'2024-01-09'},'future'), ({'as_of':'bad'},'invalid_as_of'),
    ({'dir_acc_val':'bad'},'validation_direction'), ({'amp_ratio':np.inf},'validation_amplitude'),
    ({'use_for_optimizer':False},'producer_validation')])
def test_invalid_row_uses_historical_without_removing_asset(monkeypatch, data, changes, reason):
    mu, _, tickers = build(monkeypatch, data, [signal(**changes)])
    assert tickers == ['A','B'] and mu.A == data.A.mean()
    assert reason in mu.attrs['input_diagnostics']['signals'][0]['reason']


@pytest.mark.parametrize('changes,match', [({'dataset_version':'other'},'dataset_version'),
    ({'base_currency':'ARS'},'USD'), ({'prediction_units':'five_day_log_return'},'units')])
def test_incompatible_predictions_are_rejected(monkeypatch, data, changes, match):
    with pytest.raises(OptimizationInputError,match=match):
        build(monkeypatch, data, [signal(**changes)])


def test_strict_policy_and_duplicate_predictions(monkeypatch, data):
    with pytest.raises(OptimizationInputError,match='B: missing_prediction'):
        build(monkeypatch, data, [signal()],prediction_fallback='required')
    with pytest.raises(OptimizationInputError,match='unique'):
        build(monkeypatch, data, [signal(),signal()])


def test_garch_daily_volatility_and_missing_fallback(monkeypatch, data):
    _, cov, tickers = build(monkeypatch, data, [signal(prediction_units='daily_log_volatility')],mode='ai_volatility')
    assert tickers == ['A','B']
    assert np.sqrt(cov.loc['A','A']) == pytest.approx(.03)
    assert np.sqrt(cov.loc['B','B']) == pytest.approx(data.B.std())
    assert np.linalg.eigvalsh(cov).min() >= 0


def test_missing_file_falls_back_and_classic_ignores_strict_policy(monkeypatch, data):
    def missing():
        raise FileNotFoundError('missing')
    monkeypatch.setattr(builder,'load_active_lstm_predictions_df',missing)
    mu, _, tickers = builder.build_model_inputs('ai_returns_lstm',data,RiskConfig(4,2))
    assert tickers == ['A','B']
    pd.testing.assert_series_equal(mu,data.mean())
    builder.build_model_inputs('classic',data,RiskConfig(4,2),prediction_fallback='required')
