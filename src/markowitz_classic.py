import numpy as np
from scipy.optimize import minimize


# =========================================================
# CORE UTILITIES
# =========================================================

def portfolio_return(weights, mean_returns):
    """
    Retorno esperado (daily).
    """
    return float(np.dot(weights, mean_returns))


def portfolio_volatility(weights, cov_matrix):
    """
    Volatilidad (daily).
    """
    return float(np.sqrt(weights.T @ cov_matrix @ weights))


def portfolio_performance(weights, mean_returns, cov_matrix):
    """
    Retorno y volatilidad (daily).
    """
    ret = portfolio_return(weights, mean_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    return ret, vol


def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    """
    Sharpe negativo (para minimizar).
    """
    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    if vol <= 0:
        return 1e9
    return -((ret - risk_free_rate) / vol)


# =========================================================
# CLASSIC MARKOWITZ (LONG-ONLY)
# =========================================================

def min_variance(mean_returns, cov_matrix):
    """
    Min variance long-only:
        sum(w) = 1
        0 <= w <= 1
    """
    mu = np.asarray(mean_returns)
    Sigma = np.asarray(cov_matrix)
    n = len(mu)

    def objective(w):
        return float(w.T @ Sigma @ w)

    w0 = np.ones(n) / n
    bounds = tuple((0.0, 1.0) for _ in range(n))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 5000, "ftol": 1e-12},
    )

    if not res.success:
        raise RuntimeError("Min variance did not converge: " + res.message)

    return res.x


def max_sharpe(mean_returns, cov_matrix, risk_free_rate=0.0):
    """
    Max Sharpe long-only:
        sum(w) = 1
        0 <= w <= 1
    """
    mu = np.asarray(mean_returns)
    Sigma = np.asarray(cov_matrix)
    n = len(mu)

    w0 = np.ones(n) / n
    bounds = tuple((0.0, 1.0) for _ in range(n))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    res = minimize(
        negative_sharpe,
        w0,
        args=(mu, Sigma, risk_free_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 5000, "ftol": 1e-12},
    )

    if not res.success:
        raise RuntimeError("Max Sharpe did not converge: " + res.message)

    return res.x


# =========================================================
# EXTREME RETURNS (for frontier limits)
# =========================================================

def extreme_return(mean_returns, maximize=True):
    """
    Encuentra retorno mínimo o máximo factible:
        sum(w) = 1
        0 <= w <= 1
    """
    mu = np.asarray(mean_returns)
    n = len(mu)

    sign = -1.0 if maximize else 1.0

    def objective(w):
        return float(sign * (w @ mu))

    w0 = np.ones(n) / n
    bounds = tuple((0.0, 1.0) for _ in range(n))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 5000, "ftol": 1e-12},
    )

    if not res.success:
        raise RuntimeError("Extreme return optimization failed: " + res.message)

    return res.x


# =========================================================
# CLASSIC EFFICIENT FRONTIER
# =========================================================

def min_variance_return_target(mean_returns, cov_matrix, target_return):
    """
    Min variance sujeto a:
        sum(w) = 1
        mu'w = target_return
        0 <= w <= 1
    """
    mu = np.asarray(mean_returns)
    Sigma = np.asarray(cov_matrix)
    n = len(mu)

    def objective(w):
        return float(w.T @ Sigma @ w)

    w0 = np.ones(n) / n
    bounds = tuple((0.0, 1.0) for _ in range(n))

    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w, tr=float(target_return): float(w @ mu) - tr},
    )

    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 7000, "ftol": 1e-12},
    )

    if not res.success:
        raise RuntimeError("Frontier optimization failed: " + res.message)

    return res.x


def efficient_frontier(mean_returns, cov_matrix, n_points=80):
    """
    Devuelve puntos (vol_daily, ret_daily)
    """
    mu = np.asarray(mean_returns)
    Sigma = np.asarray(cov_matrix)

    w_min = extreme_return(mu, maximize=False)
    w_max = extreme_return(mu, maximize=True)

    r_min = portfolio_return(w_min, mu)
    r_max = portfolio_return(w_max, mu)

    grid = np.linspace(r_min, r_max, n_points)

    pts = []
    for tr in grid:
        w = min_variance_return_target(mu, Sigma, tr)
        ret = portfolio_return(w, mu)
        vol = portfolio_volatility(w, Sigma)
        pts.append((vol, ret))

    pts = np.array(pts)
    pts = pts[np.argsort(pts[:, 0])]
    return pts