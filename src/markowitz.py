import numpy as np
import pandas as pd
from scipy.optimize import minimize


def portfolio_performance(weights, mean_returns, cov_matrix):
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_vol


def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(ret - risk_free_rate) / vol


def min_variance(mean_returns, cov_matrix):
    n = len(mean_returns)
    args = (mean_returns, cov_matrix)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    initial = n * [1. / n]

    result = minimize(
        lambda w: portfolio_performance(w, mean_returns, cov_matrix)[1],
        initial,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return result.x


def max_sharpe(mean_returns, cov_matrix, risk_free_rate=0.0):
    n = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    initial = n * [1. / n]

    result = minimize(
        negative_sharpe,
        initial,
        args=args,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return result.x
