import numpy as np
import pandas as pd
import pytest
from src.backtesting import BacktestConfig, walk_forward, compare_strategies


def sample():
    return pd.DataFrame(np.random.default_rng(4).normal(0,.01,size=(30,2)),columns=['A','B'],index=pd.bdate_range('2024-01-01',periods=30))


def test_training_never_observes_future_and_holdings_do_not_overlap():
    data=sample(); seen=[]
    def estimate(history):
        seen.append(history.index[-1])
        return pd.Series([.6,.4],index=history.columns)
    cfg=BacktestConfig(train_window=12,min_train=10,rebalance_every=5)
    result=walk_forward(data,estimate,cfg)
    assert result['path'].index.equals(data.index[10:])
    for end,decision in zip(seen,result['decisions']):
        assert end==pd.Timestamp(decision['train_end']) < pd.Timestamp(decision['holding_start'])
    changed=data.copy(); changed.iloc[15:]=.2
    altered=walk_forward(changed,estimate,cfg)
    pd.testing.assert_frame_equal(result['path'].iloc[:5],altered['path'].iloc[:5])


def test_future_perturbation_cannot_change_earlier_model_weights():
    data=sample()
    def estimate(history):
        raw=np.exp(history.mean()*10)
        return raw/raw.sum()
    cfg=BacktestConfig(train_window=12,min_train=10,rebalance_every=5)
    before=walk_forward(data,estimate,cfg)
    data.iloc[20:]=.1
    after=walk_forward(data,estimate,cfg)
    assert before['decisions'][:3]==after['decisions'][:3]


def test_buy_hold_drifts_and_costs_charge_traded_notional():
    data=pd.DataFrame({'A':[0.]*3+[np.log(1.1)]*6,'B':[0.]*9},index=pd.bdate_range('2024-01-01',periods=9))
    cfg=BacktestConfig(train_window=3,min_train=3,rebalance_every=2,commission_bps=10)
    result=walk_forward(data,config=cfg,strategy='buy_and_hold')
    assert result['metrics']['cumulative_simple_return']==pytest.approx((.5*1.1**6+.5)*.999-1)
    assert result['decisions'][1]['weights']['A']>.5
    assert result['decisions'][1]['turnover']==pytest.approx(0.)
    assert result['metrics']['turnover_total']==pytest.approx(1.)


def test_missing_held_asset_is_explicit_error():
    data=sample(); data.iloc[12,0]=np.nan
    with pytest.raises(ValueError,match='Missing realized return'):
        walk_forward(data,config=BacktestConfig(10,10,5),strategy='equal_weight')


def test_comparison_shares_oos_dates_and_counts_initial_drawdown():
    data=sample(); data.iloc[10:]=-.01
    results=compare_strategies(data,{'classic':lambda history:pd.Series([.5,.5],index=history.columns)},BacktestConfig(10,10,5))
    assert set(results)=={'classic','equal_weight','buy_and_hold'}
    assert results['classic']['metrics']['max_drawdown']==pytest.approx(np.exp(-.2)-1)


def test_single_oos_observation_rejected_before_estimation():
    data=pd.DataFrame({'A':[0.,0.,.01]},index=pd.bdate_range('2024-01-01',periods=3))
    with pytest.raises(ValueError,match='two OOS'):
        walk_forward(data,config=BacktestConfig(2,2),strategy='equal_weight')
