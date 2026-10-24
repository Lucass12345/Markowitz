"""Causal close-to-close portfolio simulation with explicit trading costs.

A decision made after the close of t first earns the return at t+1. USD daily log
asset returns are converted to simple returns before aggregation. Held-asset marks
must exist: this engine never invents valuations for another market's holiday.
"""
from dataclasses import asdict, dataclass
from copy import deepcopy

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
    # Appended fields preserve the positional interface of previous experiments.
    slippage_bps: float = 0.
    oos_start: str | None = None
    oos_end: str | None = None

    def __post_init__(self):
        windows = (self.train_window, self.min_train, self.rebalance_every, self.periods)
        if any(not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) for value in windows):
            raise ValueError('Walk-forward windows/periods must be integers')
        if self.min_train < 2 or self.train_window < self.min_train or self.rebalance_every < 1 or self.periods < 1:
            raise ValueError('Invalid walk-forward windows/periods')
        costs = (self.commission_bps, self.transaction_cost_bps, self.slippage_bps)
        if any(not np.isfinite(value) or value < 0 for value in costs):
            raise ValueError('Costs must be finite and nonnegative')
        if not np.isfinite(self.rf_annual_simple) or self.rf_annual_simple <= -1:
            raise ValueError('Risk-free rate must be finite and greater than -1')
        if sum(costs) >= 5000:
            raise ValueError('Total transaction costs must be below 50% of traded notional')
        if self.oos_start is not None and pd.isna(pd.to_datetime(self.oos_start, errors='coerce')):
            raise ValueError('Invalid OOS start date')
        if self.oos_end is not None and pd.isna(pd.to_datetime(self.oos_end, errors='coerce')):
            raise ValueError('Invalid OOS end date')
        if self.oos_start is not None and self.oos_end is not None and pd.Timestamp(self.oos_start) > pd.Timestamp(self.oos_end):
            raise ValueError('OOS start must not be after OOS end')


def realized_metrics(log_returns, turnover, config):
    simple = np.expm1(log_returns)
    wealth = np.r_[1., np.cumprod(1 + simple.to_numpy())]
    drawdown = wealth / np.maximum.accumulate(wealth) - 1
    vol = float(log_returns.std(ddof=1))
    rf = np.log1p(config.rf_annual_simple) / config.periods
    sharpe = (float(log_returns.mean()) - rf) / vol if vol > 0 else np.nan
    from src.portfolio_risk import path_risk
    return {
        **path_risk(log_returns),
        'measurement': 'ex_post', 'return_type': 'daily_log', 'currency': 'USD',
        'periods': config.periods, 'n_observations': len(log_returns),
        'cumulative_simple_return': float(wealth[-1] - 1),
        'annualized_log_return': float(log_returns.mean() * config.periods),
        'annualized_simple_return': float(np.expm1(log_returns.mean() * config.periods)),
        'realized_vol_annual': vol * np.sqrt(config.periods),
        'realized_sharpe_annual': annualized_sharpe(sharpe, config.periods),
        'max_drawdown': float(drawdown.min()), 'turnover_total': float(sum(turnover)),
        'turnover_mean_rebalance': float(np.mean(turnover)),
    }


def _date_bound(value, index):
    bound = pd.Timestamp(value)
    if index.tz is None:
        return bound.tz_localize(None) if bound.tzinfo is not None else bound
    return bound.tz_localize(index.tz) if bound.tzinfo is None else bound.tz_convert(index.tz)


def walk_forward(returns, estimator=None, config=None, strategy='model', *, on_decision=None, check_cancelled=None):
    """Simulate a fixed past-only estimator or a benchmark on one OOS calendar.

    ``estimator`` receives only a copied training frame. Its Series may attach
    ``decision_diagnostics`` in attrs to preserve model selection and constraints.
    Supplying an estimator for a benchmark controls its initial/rebalance weights;
    buy-and-hold calls it once, then keeps drifting holdings without rebalancing.
    """
    config = config or BacktestConfig()
    data = returns.copy().astype(float)
    if not isinstance(data.index, pd.DatetimeIndex) or data.index.isna().any() or not data.index.is_monotonic_increasing or data.index.has_duplicates or data.columns.has_duplicates:
        raise ValueError('Backtest requires sorted unique dates and tickers')
    if config.oos_end is not None:
        data = data.loc[data.index <= _date_bound(config.oos_end, data.index)]
    first_oos = config.min_train
    if config.oos_start is not None:
        first_oos = max(first_oos, int(data.index.searchsorted(_date_bound(config.oos_start, data.index))))
    if len(data) < first_oos + 2 or data.empty or np.isinf(data.to_numpy()).any():
        raise ValueError('Backtest requires at least two OOS returns and valid training data')
    if strategy not in {'model', 'equal_weight', 'buy_and_hold'} or (strategy == 'model' and estimator is None):
        raise ValueError('Invalid backtest strategy/estimator')

    current = pd.Series(0., index=data.columns)
    records, decisions, turnovers = [], [], []
    origins = range(first_oos - 1, len(data) - 1, config.rebalance_every)
    for number, i in enumerate(origins):
        if check_cancelled is not None:
            check_cancelled()
        train = data.iloc[max(0, i - config.train_window + 1):i + 1].copy(deep=True)
        if on_decision is not None:
            on_decision(number, len(origins), data.index[i])
        diagnostics = None
        if strategy == 'buy_and_hold' and decisions:
            target = current.copy()
        elif estimator is not None:
            proposed = estimator(train)
            if not isinstance(proposed, pd.Series) or proposed.index.has_duplicates or not proposed.index.isin(data.columns).all():
                raise ValueError('Estimator must return unique labeled weights in training universe')
            diagnostics = deepcopy(proposed.attrs.get('decision_diagnostics'))
            target = proposed.reindex(data.columns, fill_value=0.).astype(float)
        else:
            eligible = train.columns[train.notna().sum() >= config.min_train]
            if len(eligible) == 0:
                raise ValueError('No eligible assets for benchmark')
            target = pd.Series(1 / len(eligible), index=eligible).reindex(data.columns, fill_value=0.)
        if not np.isfinite(target).all() or target.min() < -1e-9 or abs(target.sum() - 1) > 1e-7:
            raise ValueError('Invalid portfolio weights')
        traded = float(np.abs(target - current).sum())
        turnover = 1. if not decisions else traded / 2
        turnovers.append(turnover)
        fee = traded * (config.commission_bps + config.transaction_cost_bps + config.slippage_bps) / 10000
        current = target.copy()
        stop = min(i + config.rebalance_every + 1, len(data))
        decision = {
            'decision_date': data.index[i].isoformat(), 'train_start': train.index[0].isoformat(),
            'train_end': train.index[-1].isoformat(), 'holding_start': data.index[i + 1].isoformat(),
            'holding_end': data.index[stop - 1].isoformat(), 'weights': target.to_dict(),
            'turnover': turnover, 'traded_notional': traded, 'transaction_cost_fraction': fee,
        }
        if diagnostics is not None:
            decision['diagnostics'] = diagnostics
        decisions.append(decision)
        for j in range(i + 1, stop):
            if check_cancelled is not None:
                check_cancelled()
            values = data.iloc[j]
            missing = values.isna() & (current.abs() > 1e-12)
            if missing.any():
                raise ValueError(f'Missing realized return for held assets on {data.index[j]}: {list(values.index[missing])}')
            # Only unheld assets may contribute zero without an observed return.
            asset_simple = np.expm1(values.fillna(0.))
            gross = float(current @ asset_simple)
            net = (1 + gross) * (1 - fee if j == i + 1 else 1) - 1
            if not np.isfinite(net) or net <= -1:
                raise ValueError('Non-finite or bankrupt portfolio path')
            records.append({
                'date': data.index[j], 'log_return': float(np.log1p(net)), 'simple_return': net,
                'gross_simple_return': gross, 'cost_fraction': fee if j == i + 1 else 0.,
            })
            current = current * (1 + asset_simple) / (1 + gross)
    path = pd.DataFrame(records).set_index('date')
    path['wealth'] = np.exp(path.log_return.cumsum())
    return {
        'config': asdict(config), 'strategy': strategy, 'path': path, 'decisions': decisions,
        'metrics': realized_metrics(path.log_return, turnovers, config),
        'methodology': {
            'decision': 'after close t; first holding return t+1',
            'valuation': 'strict observed held-asset returns',
            'turnover': 'one-way 0.5 L1; initial entry 1',
            'costs': 'commission plus slippage plus transaction bps times traded L1 notional',
            'constraints_timing': 'target weights at rebalance; weights drift between decisions',
        },
    }


def compare_strategies(returns, estimators, config=None):
    config = config or BacktestConfig()
    result = {name: walk_forward(returns, estimator, config) for name, estimator in estimators.items()}
    result['equal_weight'] = walk_forward(returns, config=config, strategy='equal_weight')
    result['buy_and_hold'] = walk_forward(returns, config=config, strategy='buy_and_hold')
    return result
