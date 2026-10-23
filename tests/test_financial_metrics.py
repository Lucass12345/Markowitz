import numpy as np
import pandas as pd
import pytest
from src.utils_portfolio import portfolio_statistics, annualized_sharpe
from src.optimization.frontier_engine import _stats_dict
from src.optimization.optimize_blocks import run_blocks_optimizer


def test_daily_and_annual_sharpe_are_distinct_with_nonzero_rf():
    mu = np.array([.0006, .0009]); cov = np.diag([.0001, .0002]); w = np.array([.4, .6])
    stats = portfolio_statistics(w, mu, cov, rf_log_daily=.0001)
    expected_daily = (w @ mu - .0001) / np.sqrt(w @ cov @ w)
    assert stats['sharpe_daily'] == pytest.approx(expected_daily)
    assert stats['sharpe_annual'] == pytest.approx(expected_daily * np.sqrt(252))
    assert stats['return_log_annual'] == stats['return_annual'] == pytest.approx(w @ mu * 252)
    assert stats['return_simple_annual'] == pytest.approx(np.expm1(w @ mu * 252))
    assert _stats_dict(w, mu, cov, .0001)['sharpe_annual'] == stats['sharpe_annual']


def test_blocks_metrics_use_the_same_annualization():
    mu = pd.Series([.001, .0008], index=['A', 'B'])
    cov = pd.DataFrame(np.diag([.0001, .0002]), index=mu.index, columns=mu.index)
    meta = pd.DataFrame({'ticker':mu.index, 'block':['X','X']})
    result = run_blocks_optimizer(mu, cov, list(mu.index), meta, {'X':1.}, max_weight=1)
    stats = result['stats']
    assert stats['sharpe_annual'] == pytest.approx(stats['sharpe_daily'] * np.sqrt(252))


@pytest.mark.parametrize('periods', [0, -1, np.inf])
def test_invalid_annualization_periods(periods):
    with pytest.raises(ValueError): annualized_sharpe(.1, periods)
