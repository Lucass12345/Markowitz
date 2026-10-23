# src/markowitz.py
# =========================================================
# Markowitz core + bloques (convención: mu y Sigma en la MISMA escala)
# Este repo usa log returns diarios (returns.csv), así que:
#   mu = E[r_log_daily]
#   Sigma = Cov[r_log_daily]
#   risk_free_rate (si se usa) debe ser log daily también.
# =========================================================

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.optimize import minimize

from src.optimization.errors import (
    PortfolioValidationError, ConstraintFeasibilityError, validate_targets,
)

TRADING_DAYS = 252


# =========================================================
# Utilidades de alineación / validación
# =========================================================
def _as_numpy_1d(x):
    try:
        values = np.asarray(x.values if hasattr(x, "values") else x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise PortfolioValidationError("Expected a numeric one-dimensional portfolio vector.") from exc
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise PortfolioValidationError("Portfolio vectors must be nonempty, one-dimensional and finite.")
    index = getattr(x, "index", None)
    if index is not None and not callable(index) and not index.is_unique:
        raise PortfolioValidationError("Portfolio vector labels must be unique.")
    return values


def _as_numpy_2d(x):
    if hasattr(x, "columns"):
        if not x.index.is_unique or not x.columns.is_unique:
            raise PortfolioValidationError("Covariance row and column labels must be unique.")
        if set(x.index) != set(x.columns):
            raise PortfolioValidationError("Covariance row and column labels must match.")
        x = x.loc[x.index, x.index]
    try:
        values = np.asarray(x.values if hasattr(x, "values") else x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise PortfolioValidationError("Covariance must be a numeric square matrix.") from exc
    if values.ndim != 2 or not len(values) or values.shape[0] != values.shape[1] or not np.isfinite(values).all():
        raise PortfolioValidationError("Covariance must be nonempty, square and finite.")
    if not np.allclose(values, values.T, rtol=1e-8, atol=1e-12):
        raise PortfolioValidationError("Covariance must be symmetric.")
    return values


def _align_mu_sigma(mean_returns, cov_matrix):
    mu = _as_numpy_1d(mean_returns)
    sigma = _as_numpy_2d(cov_matrix)
    if sigma.shape != (len(mu), len(mu)):
        raise PortfolioValidationError("Mean returns and covariance dimensions must match.")
    index = getattr(mean_returns, "index", None)
    if index is not None and not callable(index) and hasattr(cov_matrix, "loc"):
        if set(index) != set(cov_matrix.index):
            raise PortfolioValidationError("Mean-return and covariance labels must match.")
        cov_matrix = cov_matrix.loc[index, index]
    return mean_returns, cov_matrix


def _check_block_targets(block_targets, blocks, tol=1e-9):
    validate_targets(block_targets, tol=tol)
    if not isinstance(blocks, Mapping):
        raise ConstraintFeasibilityError("Blocks must be a mapping of names to asset indices.")
    missing = set(block_targets) - set(blocks)
    if missing:
        raise ConstraintFeasibilityError(f"Missing block definitions: {sorted(missing)}.")


def validate_block_feasibility(n, blocks, block_targets, max_weight=1., tol=1e-9):
    """Sufficient capacity checks for disjoint, long-only block allocations."""
    _check_block_targets(block_targets, blocks, tol)
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n <= 0:
        raise ConstraintFeasibilityError("The optimizer requires at least one asset.")
    try:
        cap = float(max_weight)
    except (TypeError, ValueError) as exc:
        raise ConstraintFeasibilityError("max_weight must be finite and in (0, 1].") from exc
    if not np.isfinite(cap) or not 0 < cap <= 1:
        raise ConstraintFeasibilityError("max_weight must be finite and in (0, 1].")
    seen = set()
    for name, indices in blocks.items():
        if isinstance(indices, (str, bytes)) or not hasattr(indices, "__len__"):
            raise ConstraintFeasibilityError(f"Indices for block {name} must be an integer sequence.")
        try:
            indices = list(indices)
        except TypeError as exc:
            raise ConstraintFeasibilityError(f"Indices for block {name} must be an integer sequence.") from exc
        current = set()
        for index in indices:
            if not isinstance(index, (int, np.integer)) or isinstance(index, bool) or not 0 <= index < n:
                raise ConstraintFeasibilityError(f"Block {name} has an invalid asset index: {index}.")
            if index in current or index in seen:
                raise ConstraintFeasibilityError("Block indices must be unique and blocks must not overlap.")
            current.add(index)
        seen.update(current)
    if n * cap < 1 - tol:
        raise ConstraintFeasibilityError(f"Global capacity is {n * cap:g}; {n} assets at max_weight={cap:g} cannot sum to 1.")
    for name, target in block_targets.items():
        count = len(blocks[name])
        if target > 0 and count == 0:
            raise ConstraintFeasibilityError(f"Block {name} has target {target:g} but no available assets.")
        if target > count * cap + tol:
            raise ConstraintFeasibilityError(f"Block {name} target {target:g} exceeds capacity {count * cap:g} ({count} assets × {cap:g}).")


def rf_simple_annual_to_log_daily(rf_simple_annual: float, periods: int = TRADING_DAYS) -> float:
    """
    Convierte rf simple anual -> log daily (consistente con log returns diarios).
    """
    return float(np.log1p(rf_simple_annual) / periods)


# =========================================================
# Core (sin bloques)
# =========================================================
def portfolio_performance(weights, mean_returns, cov_matrix):
    """
    Retorno y volatilidad (diarios) en la escala de mean_returns/Sigma.
    En tu caso: log daily.
    """
    mean_returns, cov_matrix = _align_mu_sigma(mean_returns, cov_matrix)

    w = _as_numpy_1d(weights)
    mu = _as_numpy_1d(mean_returns)
    Sigma = _as_numpy_2d(cov_matrix)

    ret = float(w @ mu)
    var = float(w.T @ Sigma @ w)
    vol = float(np.sqrt(max(var, 0.0)))
    return ret, vol


def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    if not np.isfinite(vol) or vol <= 1e-18:
        return 1e9
    return -((ret - float(risk_free_rate)) / vol)


def min_variance(mu, Sigma, bounds=None, constraints=()):
    """
    Min var: solve directo. (Matemáticamente equivalente a min vol, pero
    numéricamente más estable para SLSQP.)
    """
    mu, Sigma = _align_mu_sigma(mu, Sigma)

    n = len(mu)
    Sigma_np = _as_numpy_2d(Sigma)

    base_constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    all_constraints = tuple(base_constraints) + tuple(constraints)

    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(n))

    w0 = np.ones(n) / n

    res = minimize(
        lambda w: float(w.T @ Sigma_np @ w),  # varianza
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=all_constraints,
        options={"maxiter": 5000, "ftol": 1e-12}
    )
    if not res.success:
        raise RuntimeError(f"min_variance failed: {res.message}")
    return res.x


def max_sharpe(mean_returns, cov_matrix, risk_free_rate=0.0):
    """
    Max Sharpe classic long-only: 0<=w<=1, sum(w)=1
    """
    mean_returns, cov_matrix = _align_mu_sigma(mean_returns, cov_matrix)

    n = len(mean_returns)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    w0 = np.ones(n) / n

    res = minimize(
        negative_sharpe,
        w0,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 5000, "ftol": 1e-12}
    )
    if not res.success:
        raise RuntimeError(f"max_sharpe failed: {res.message}")
    return res.x


# =========================================================
# Helpers (bloques)
# =========================================================
def _init_w0_from_block_targets(n, blocks, block_targets):
    """
    Inicial uniforme dentro de cada bloque según targets.
    Asume targets suman 1.
    """
    w0 = np.zeros(n, dtype=float)
    for bname, idx in blocks.items():
        t = float(block_targets.get(bname, 0.0))
        if len(idx) > 0:
            w0[list(idx)] = t / len(idx)
    return w0


def _block_constraints(blocks, block_targets, n=None):
    """Keep independent equalities and skip empty zero-target blocks."""
    active = [(name, list(blocks[name]), float(target))
              for name, target in block_targets.items() if len(blocks[name])]
    covered = [index for _, indices, _ in active for index in indices]
    full_partition = (n is not None and len(covered) == n and len(set(covered)) == n)
    # With a full partition, block equalities already imply the budget. Retain
    # the budget when OTHER/unassigned assets exist so they cannot gain weight.
    cons = [] if full_partition else [{"type": "eq", "fun": lambda w: np.sum(w) - 1.}]
    for _, indices, target in active:
        cons.append({"type": "eq", "fun": lambda w, idx=indices, t=target: np.sum(w[idx]) - t})
    return cons


# =========================================================
# Con bloques: Min Var
# =========================================================
def min_variance_with_blocks(cov_matrix, blocks, block_targets, max_weight=1.0):
    """
    Min var long-only con:
      - sum(w)=1
      - 0 <= w <= max_weight
      - sum(w en bloque)=target para cada bloque
    Devuelve OptimizeResult
    """
    _check_block_targets(block_targets, blocks)

    Sigma = _as_numpy_2d(cov_matrix)
    n = Sigma.shape[0]
    validate_block_feasibility(n, blocks, block_targets, max_weight)

    def port_var(w):
        return float(w.T @ Sigma @ w)

    bounds = tuple((0.0, float(max_weight)) for _ in range(n))
    w0 = _init_w0_from_block_targets(n, blocks, block_targets)
    constraints = _block_constraints(blocks, block_targets, n=n)

    res = minimize(
        port_var,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 4000, "ftol": 1e-12}
    )
    return res


# =========================================================
# Extremos de retorno (factible) con bloques
# =========================================================
def extreme_return_with_blocks(mean_returns, blocks, block_targets, maximize=True, max_weight=1.0):
    """
    Encuentra retorno mínimo o máximo factible bajo constraints (sin mirar varianza).
    maximize=True => max mu'w
    maximize=False => min mu'w
    Devuelve OptimizeResult
    """
    _check_block_targets(block_targets, blocks)

    mu = _as_numpy_1d(mean_returns)
    n = len(mu)
    validate_block_feasibility(n, blocks, block_targets, max_weight)

    sign = -1.0 if maximize else 1.0

    def obj(w):
        return float(sign * (w @ mu))

    bounds = tuple((0.0, float(max_weight)) for _ in range(n))
    w0 = _init_w0_from_block_targets(n, blocks, block_targets)
    constraints = _block_constraints(blocks, block_targets, n=n)

    res = minimize(
        obj,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 3000, "ftol": 1e-12}
    )
    return res


# =========================================================
# Frontera: Min Var dado retorno mínimo (>=) con bloques
# =========================================================
def min_variance_with_blocks_min_return(
    cov_matrix,
    mean_returns,
    blocks,
    block_targets,
    target_return,
    max_weight=1.0
):
    """
    Min var long-only con:
      - sum(w)=1
      - 0<=w<=max_weight
      - sum(w en bloque)=target_bloque
      - mu'w >= target_return   (INEQ, más estable que igualdad)
    Devuelve OptimizeResult
    """
    _check_block_targets(block_targets, blocks)

    mean_returns, cov_matrix = _align_mu_sigma(mean_returns, cov_matrix)
    mu = _as_numpy_1d(mean_returns)
    Sigma = _as_numpy_2d(cov_matrix)
    n = len(mu)
    validate_block_feasibility(n, blocks, block_targets, max_weight)

    def portfolio_var(w):
        return float(w.T @ Sigma @ w)

    w0 = _init_w0_from_block_targets(n, blocks, block_targets)
    bounds = tuple((0.0, float(max_weight)) for _ in range(n))

    constraints = _block_constraints(blocks, block_targets, n=n)
    constraints.append({
        "type": "ineq",
        "fun": lambda w, tr=float(target_return): float(w @ mu) - tr
    })

    res = minimize(
        portfolio_var,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 4000, "ftol": 1e-12}
    )
    return res


def min_variance_with_blocks_return_target(
    cov_matrix,
    mean_returns,
    blocks,
    block_targets,
    target_return,
    max_weight=1.0
):
    """
    Versión con igualdad mu'w = target_return (menos estable).
    Devuelve OptimizeResult
    """
    _check_block_targets(block_targets, blocks)

    mean_returns, cov_matrix = _align_mu_sigma(mean_returns, cov_matrix)
    mu = _as_numpy_1d(mean_returns)
    Sigma = _as_numpy_2d(cov_matrix)
    n = len(mu)
    validate_block_feasibility(n, blocks, block_targets, max_weight)

    def portfolio_var(w):
        return float(w.T @ Sigma @ w)

    w0 = _init_w0_from_block_targets(n, blocks, block_targets)
    bounds = tuple((0.0, float(max_weight)) for _ in range(n))

    constraints = _block_constraints(blocks, block_targets, n=n)
    constraints.append({
        "type": "eq",
        "fun": lambda w, tr=float(target_return): float(w @ mu) - tr
    })

    res = minimize(
        portfolio_var,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 4000, "ftol": 1e-12}
    )
    return res


# =========================================================
# Con bloques: Max Sharpe (robusto)
# =========================================================
def max_sharpe_with_blocks(
    mean_returns,
    cov_matrix,
    blocks,
    block_targets,
    risk_free_rate=0.0,
    max_weight=1.0,
    n_grid=120,
    tol=1e-6
):
    """
    Max Sharpe bajo constraints de bloques, de forma robusta:

    1) Resuelve MinVar (factible) como lower bound.
    2) Resuelve extremos de retorno factible (min/max).
    3) Recorre una grilla de retornos y resuelve: MinVar s.t. mu'w >= tr
       (convexo, estable). Calcula Sharpe y elige el mejor.
    4) Si la búsqueda numérica no supera a MinVar por tol, devuelve MinVar.

    Nota: risk_free_rate debe estar en la misma escala que mean_returns
    (log daily si mean_returns es log daily).
    """
    mean_returns, cov_matrix = _align_mu_sigma(mean_returns, cov_matrix)
    _check_block_targets(block_targets, blocks)

    mu = _as_numpy_1d(mean_returns)
    Sigma = _as_numpy_2d(cov_matrix)
    validate_block_feasibility(len(mu), blocks, block_targets, max_weight)

    def sharpe_of(w):
        w = np.asarray(w, dtype=float)
        var = float(w.T @ Sigma @ w)
        if not np.isfinite(var) or var <= 1e-18:
            return -np.inf
        vol = float(np.sqrt(var))
        ret = float(w @ mu)
        return (ret - float(risk_free_rate)) / vol  # Sharpe diario consistente

    # 0) Lower bound: MinVar
    res_mv = min_variance_with_blocks(cov_matrix, blocks, block_targets, max_weight=max_weight)
    if not getattr(res_mv, "success", False):
        return res_mv

    w_minvar = np.array(res_mv.x, dtype=float)
    sh_minvar = sharpe_of(w_minvar)

    # 1) extremos de retorno factible
    res_min_ret = extreme_return_with_blocks(mean_returns, blocks, block_targets,
                                             maximize=False, max_weight=max_weight)
    res_max_ret = extreme_return_with_blocks(mean_returns, blocks, block_targets,
                                             maximize=True, max_weight=max_weight)

    if (not res_min_ret.success) or (not res_max_ret.success):
        res_mv.message = f"Fallback to MinVar. Sharpe(minvar)={sh_minvar:.6f}"
        return res_mv

    r_min = float(mu @ np.asarray(res_min_ret.x, dtype=float))
    r_max = float(mu @ np.asarray(res_max_ret.x, dtype=float))

    if (r_max - r_min) < 1e-12:
        res_mv.message = f"Degenerate return range. Using MinVar. Sharpe(minvar)={sh_minvar:.6f}"
        return res_mv

    # 2) grilla de retornos (evito extremos)
    grid = np.linspace(
        r_min + 0.02 * (r_max - r_min),
        r_max - 0.02 * (r_max - r_min),
        int(n_grid)
    )

    best_res = None
    best_sh = -np.inf

    # 3) recorro frontera: min var con retorno mínimo (>=)
    for tr in grid:
        res = min_variance_with_blocks_min_return(
            cov_matrix=cov_matrix,
            mean_returns=mean_returns,
            blocks=blocks,
            block_targets=block_targets,
            target_return=float(tr),
            max_weight=max_weight
        )
        if not res.success:
            continue

        sh = sharpe_of(res.x)
        if sh > best_sh:
            best_sh = sh
            best_res = res

    # 4) Si la búsqueda numérica no mejora a MinVar, devolvé MinVar
    if best_res is None or (best_sh < sh_minvar - tol):
        res_mv.message = (
            f"MaxSharpe search did not beat MinVar within tol. "
            f"Sharpe(minvar)={sh_minvar:.6f}, best_found={best_sh:.6f}. Using MinVar."
        )
        return res_mv

    # 5) Si mejora o empata
    best_res.message = (
        f"Selected MaxSharpe on efficient frontier. "
        f"Sharpe(best)={best_sh:.6f}, Sharpe(minvar)={sh_minvar:.6f}"
    )
    return best_res


def max_sharpe_with_blocks_direct(
    mean_returns,
    cov_matrix,
    blocks,
    block_targets,
    risk_free_rate=0.0,
    max_weight=1.0
):
    """
    Opción directa (no convexa) por si querés probar/diagnosticar.
    Normalmente la versión por frontera (convexa) es más estable.
    """
    mean_returns, cov_matrix = _align_mu_sigma(mean_returns, cov_matrix)
    _check_block_targets(block_targets, blocks)

    mu = _as_numpy_1d(mean_returns)
    Sigma = _as_numpy_2d(cov_matrix)
    n = len(mu)
    validate_block_feasibility(n, blocks, block_targets, max_weight)

    def obj(w):
        var = float(w.T @ Sigma @ w)
        if var <= 1e-18 or not np.isfinite(var):
            return 1e9
        vol = float(np.sqrt(var))
        ret = float(w @ mu)
        return -((ret - float(risk_free_rate)) / vol)

    bounds = tuple((0.0, float(max_weight)) for _ in range(n))
    w0 = _init_w0_from_block_targets(n, blocks, block_targets)
    constraints = _block_constraints(blocks, block_targets, n=n)

    res = minimize(
        obj,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 4000, "ftol": 1e-12}
    )
    return res