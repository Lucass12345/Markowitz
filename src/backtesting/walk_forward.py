"""Close-to-close simulation. Decisions after t's close trade for t+1.

Inputs are USD daily log returns. Realized portfolio returns use expm1(asset log
returns) and drifting holdings, never a weighted sum of log returns. Missing held
asset marks are an error: this engine cannot infer a price for a market holiday.
"""
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from src.utils_portfolio import annualized_sharpe


@dataclass(frozen=True)
class BacktestConfig:
    train_window: int = 504
    min_train: int = 252
    rebalance_every: int = 5
    periods: int = 252
    commission_bps: float = 0.
    transaction_cost_bps: float = 0.
    rf_annual_simple: float = 0.

    def __post_init__(self):
        if self.min_train < 2 or self.train_window < self.min_train or self.rebalance_every < 1 or self.periods < 1:
            raise ValueError('Invalid walk-forward windows/periods')
        if any(not np.isfinite(v) or v < 0 for v in (self.commission_bps,self.transaction_cost_bps,self.rf_annual_simple)):
            raise ValueError('Costs/risk-free rate must be finite and nonnegative')
        if self.commission_bps + self.transaction_cost_bps >= 5000:
            raise ValueError('Total transaction costs must be below 50% of traded notional')


def realized_metrics(log_returns, turnover, config):
    simple = np.expm1(log_returns)
    wealth = np.r_[1., np.cumprod(1 + simple.to_numpy())]
    drawdown = wealth / np.maximum.accumulate(wealth) - 1
    vol = float(log_returns.std(ddof=1))
    rf = np.log1p(config.rf_annual_simple) / config.periods
    sharpe = (float(log_returns.mean())-rf)/vol if vol > 0 else np.nan
    from src.portfolio_risk import path_risk
    return {**path_risk(log_returns), 'measurement':'ex_post', 'return_type':'daily_log', 'currency':'USD', 'periods':config.periods,
            'n_observations':len(log_returns), 'cumulative_simple_return':float(wealth[-1]-1),
            'annualized_log_return':float(log_returns.mean()*config.periods),
            'realized_vol_annual':vol*np.sqrt(config.periods), 'realized_sharpe_annual':annualized_sharpe(sharpe,config.periods),
            'max_drawdown':float(drawdown.min()), 'turnover_total':float(sum(turnover)),
            'turnover_mean_rebalance':float(np.mean(turnover))}


def walk_forward(returns, estimator=None, config=None, strategy='model'):
    config = config or BacktestConfig()
    data=returns.copy().astype(float)
    if not isinstance(data.index,pd.DatetimeIndex) or not data.index.is_monotonic_increasing or data.index.has_duplicates or data.columns.has_duplicates:
        raise ValueError('Backtest requires sorted unique dates and tickers')
    if len(data)<=config.min_train or data.empty or np.isinf(data.to_numpy()).any():
        raise ValueError('Insufficient or invalid backtest returns')
    if strategy not in {'model','equal_weight','buy_and_hold'} or (strategy=='model' and estimator is None):
        raise ValueError('Invalid backtest strategy/estimator')
    current=pd.Series(0.,index=data.columns)
    records, decisions, turnovers = [],[],[]
    for i in range(config.min_train-1,len(data)-1,config.rebalance_every):
        train=data.iloc[max(0,i-config.train_window+1):i+1].copy()
        # The callback receives no reference to future returns or live artifacts.
        if strategy=='buy_and_hold' and decisions:
            target=current.copy()
        elif strategy in {'equal_weight','buy_and_hold'}:
            eligible=train.columns[train.notna().sum()>=config.min_train]
            if len(eligible)==0:
                raise ValueError('No eligible assets for benchmark')
            target=pd.Series(1/len(eligible),index=eligible).reindex(data.columns,fill_value=0.)
        else:
            proposed=estimator(train.copy(deep=True))
            if not isinstance(proposed,pd.Series) or proposed.index.has_duplicates or not proposed.index.isin(data.columns).all():
                raise ValueError('Estimator must return unique labeled weights in training universe')
            target=proposed.reindex(data.columns,fill_value=0.).astype(float)
        if not np.isfinite(target).all() or target.min() < -1e-9 or abs(target.sum()-1)>1e-7:
            raise ValueError('Invalid portfolio weights')
        traded=float(np.abs(target-current).sum())
        turnover=1. if not decisions else traded/2
        turnovers.append(turnover)
        fee=traded*(config.commission_bps+config.transaction_cost_bps)/10000
        current=target.copy()
        stop=min(i+config.rebalance_every+1,len(data))
        decisions.append({'decision_date':data.index[i].isoformat(),'train_start':train.index[0].isoformat(),
                          'train_end':train.index[-1].isoformat(),'holding_start':data.index[i+1].isoformat(),
                          'holding_end':data.index[stop-1].isoformat(),'weights':target.to_dict(),
                          'turnover':turnover,'traded_notional':traded,'transaction_cost_fraction':fee})
        for j in range(i+1,stop):
            values=data.iloc[j]
            missing=values.isna() & (current.abs()>1e-12)
            if missing.any():
                raise ValueError(f'Missing realized return for held assets on {data.index[j]}: {list(values.index[missing])}')
            # Missing unheld assets contribute exactly zero by their zero weight.
            asset_simple=np.expm1(values.fillna(0.))
            gross=float(current @ asset_simple)
            net=(1+gross)*(1-fee if j==i+1 else 1)-1
            if not np.isfinite(net) or net<=-1:
                raise ValueError('Non-finite or bankrupt portfolio path')
            records.append({'date':data.index[j],'log_return':float(np.log1p(net)),
                            'simple_return':net,'gross_simple_return':gross,'cost_fraction':fee if j==i+1 else 0.})
            current=current*(1+asset_simple)/(1+gross)
    path=pd.DataFrame(records).set_index('date')
    return {'config':asdict(config),'strategy':strategy,'path':path,'decisions':decisions,
            'metrics':realized_metrics(path.log_return,turnovers,config),
            'methodology':{'decision':'after close t; first holding return t+1','valuation':'strict observed held-asset returns',
                           'turnover':'one-way 0.5 L1; initial entry 1','costs':'commission plus transaction bps times traded L1 notional'}}


def compare_strategies(returns, estimators, config=None):
    config=config or BacktestConfig()
    result={name:walk_forward(returns,estimator,config) for name,estimator in estimators.items()}
    result['equal_weight']=walk_forward(returns,config=config,strategy='equal_weight')
    result['buy_and_hold']=walk_forward(returns,config=config,strategy='buy_and_hold')
    return result
