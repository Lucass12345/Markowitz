import numpy as np
from scipy.optimize import minimize


# =========================================================
# Core (sin bloques)
# =========================================================
def portfolio_performance(weights, mean_returns, cov_matrix):
    """
    Retorno y volatilidad (diarios).
    mean_returns: pd.Series o np.array (n,)
    cov_matrix: pd.DataFrame o np.array (n,n)
    """
    portfolio_return = float(np.dot(weights, mean_returns))
    portfolio_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
    return portfolio_return, portfolio_vol


def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    if vol <= 0:
        return 1e9
    return -((ret - risk_free_rate) / vol)


def min_variance(mean_returns, cov_matrix):
    """
    Min variance long-only (0<=w<=1), sum(w)=1
    Devuelve weights (np.array)
    """
    n = len(mean_returns)

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    w0 = np.array([1.0 / n] * n)

    # Minimizar volatilidad (equivalente a minimizar varianza)
    res = minimize(
        lambda w: portfolio_performance(w, mean_returns, cov_matrix)[1],
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return res.x


def max_sharpe(mean_returns, cov_matrix, risk_free_rate=0.0):
    """
    Max Sharpe long-only (0<=w<=1), sum(w)=1
    Devuelve weights (np.array)
    """
    n = len(mean_returns)

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    w0 = np.array([1.0 / n] * n)

    res = minimize(
        negative_sharpe,
        w0,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return res.x


# =========================================================
# Helpers (bloques)
# =========================================================
def _init_w0_from_block_targets(n, blocks, block_targets):
    """
    Inicial uniforme dentro de cada bloque según targets.
    """
    w0 = np.zeros(n)
    for bname, idx in blocks.items():
        t = float(block_targets.get(bname, 0.0))
        if len(idx) > 0:
            w0[idx] = t / len(idx)
    return w0


def _block_constraints(blocks, block_targets):
    """
    Lista de constraints de igualdad:
    - sum(w)=1
    - sum(w[idx_block]) = target_block
    """
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    for bname, target in block_targets.items():
        idx = blocks[bname]
        cons.append({
            "type": "eq",
            "fun": lambda w, idx=idx, t=float(target): np.sum(w[idx]) - t
        })
    return cons


def _cap_and_redistribute_block(weights_block, target_sum, max_weight, tol=1e-12):
    """
    Ajusta un vector de pesos dentro de un bloque para que:
    - sum = target_sum
    - 0 <= w_i <= max_weight

    Water-filling con cap.
    Devuelve np.array o None si es infeasible/numéricamente inestable.
    """
    w = np.array(weights_block, dtype=float)
    n = len(w)

    if n == 0:
        return np.array([], dtype=float)

    # si viene todo cero, arranca uniforme
    if np.all(w <= tol):
        w = np.ones(n) / n

    # normalizar a target_sum
    s = w.sum()
    if s <= tol:
        w = np.ones(n) / n
        s = 1.0
    w = w / s * target_sum

    active = np.ones(n, dtype=bool)

    for _ in range(10 * n):
        over = (w > max_weight + tol) & active
        if not np.any(over):
            break

        w[over] = max_weight
        active[over] = False

        remaining = target_sum - w[~active].sum()
        if remaining < -1e-9:
            return None

        if active.sum() == 0:
            break

        s_active = w[active].sum()
        if s_active <= tol:
            w[active] = remaining / active.sum()
        else:
            w[active] = w[active] / s_active * remaining

    # clip final
    w = np.clip(w, 0.0, max_weight)

    # ajuste final de suma (suave)
    s = w.sum()
    if abs(s - target_sum) > 1e-6:
        free = w < (max_weight - 1e-10)
        if free.sum() > 0:
            remaining = target_sum - w[~free].sum()
            if remaining < -1e-9:
                return None
            sf = w[free].sum()
            if sf <= tol:
                w[free] = remaining / free.sum()
            else:
                w[free] = w[free] / sf * remaining
        w = np.clip(w, 0.0, max_weight)

    if abs(w.sum() - target_sum) > 1e-4:
        return None
    if np.any(w < -1e-9) or np.any(w > max_weight + 1e-9):
        return None

    return w


def _random_feasible_w0(n, blocks, block_targets, max_weight, rng):
    """
    Genera un w0 factible:
    - por cada bloque: Dirichlet -> suma target -> cap por activo con redistribución
    - suma total 1 (si targets suman 1)
    """
    w0 = np.zeros(n, dtype=float)

    for bname, idx in blocks.items():
        t = float(block_targets.get(bname, 0.0))
        k = len(idx)
        if k == 0:
            continue

        raw = rng.dirichlet(alpha=np.ones(k))
        wb = _cap_and_redistribute_block(raw, t, max_weight)
        if wb is None:
            return None
        w0[idx] = wb

    if abs(w0.sum() - 1.0) > 1e-6:
        return None
    if np.any(w0 < -1e-9) or np.any(w0 > max_weight + 1e-9):
        return None

    return w0


# =========================================================
# Con bloques: Max Sharpe / Min Var
# =========================================================
def max_sharpe_with_blocks(mean_returns, cov_matrix, blocks, block_targets,
                           risk_free_rate=0.0, max_weight=1.0,
                           n_grid=120, tol=1e-6):
    """
    Max Sharpe *matemáticamente consistente* bajo constraints:

    - Computa w_minvar (factible) => Sharpe_minvar (lower bound).
    - Construye frontera eficiente con constraints (convexo) usando:
        min var s.t. mu'w >= target_return
      para una grilla de retornos entre min/max factible.
    - Elige el mayor Sharpe encontrado.
    - Si por aproximación numérica el mejor Sharpe < Sharpe_minvar - tol,
      devuelve w_minvar (porque el verdadero máximo no puede ser peor que un factible).

    risk_free_rate debe estar en la misma unidad que mean_returns
    (si mean_returns es log diario, rf log diario).
    """
    mu = mean_returns.values
    Sigma = cov_matrix.values

    def sharpe_of(w):
        w = np.asarray(w, dtype=float)
        var = float(w.T @ Sigma @ w)
        if not np.isfinite(var) or var <= 1e-18:
            return -np.inf
        vol = float(np.sqrt(var))
        ret = float(w @ mu)
        return (ret - float(risk_free_rate)) / vol  # Sharpe diario (consistente)

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

    r_min = float(mu @ res_min_ret.x)
    r_max = float(mu @ res_max_ret.x)

    if (r_max - r_min) < 1e-10:
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
            target_return=float(tr),     # ✅ nombre correcto
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


def _max_sharpe_with_blocks_direct(mean_returns, cov_matrix, blocks, block_targets,
                                   risk_free_rate=0.0, max_weight=1.0):
    """
    Fallback: versión directa (no convexa) por si la grilla falla.
    """
    mu = mean_returns.values
    Sigma = cov_matrix.values
    n = len(mu)

    def obj(w):
        var = float(w.T @ Sigma @ w)
        if var <= 1e-18 or not np.isfinite(var):
            return 1e9
        vol = float(np.sqrt(var))
        ret = float(w @ mu)
        return -((ret - float(risk_free_rate)) / vol)

    bounds = tuple((0.0, float(max_weight)) for _ in range(n))
    w0 = _init_w0_from_block_targets(n, blocks, block_targets)
    constraints = _block_constraints(blocks, block_targets)

    return minimize(
        obj,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 4000, "ftol": 1e-12}
    )


def min_variance_with_blocks(cov_matrix, blocks, block_targets, max_weight=1.0):
    """
    Min var long-only con:
      - sum(w)=1
      - 0 <= w <= max_weight
      - sum(w en bloque)=target para cada bloque
    Devuelve res (OptimizeResult)
    """
    Sigma = cov_matrix.values
    n = Sigma.shape[0]

    def port_var(w):
        return float(np.dot(w.T, np.dot(Sigma, w)))

    bounds = tuple((0.0, float(max_weight)) for _ in range(n))
    w0 = _init_w0_from_block_targets(n, blocks, block_targets)
    constraints = _block_constraints(blocks, block_targets)

    res = minimize(
        port_var,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 3000, "ftol": 1e-12}
    )
    return res


# =========================================================
# Frontera exacta: Min Var dado retorno objetivo (con bloques)
# =========================================================
def min_variance_with_blocks_return_target(cov_matrix, mean_returns, blocks, block_targets,
                                           target_return, max_weight=1.0):
    """
    Min var long-only con:
      - sum(w)=1
      - 0 <= w <= max_weight
      - sum(w en bloque)=target_bloque
      - mu'w = target_return
    Devuelve res (OptimizeResult)
    """
    mu = mean_returns.values
    Sigma = cov_matrix.values
    n = len(mu)

    def port_var(w):
        return float(np.dot(w.T, np.dot(Sigma, w)))

    bounds = tuple((0.0, float(max_weight)) for _ in range(n))
    w0 = _init_w0_from_block_targets(n, blocks, block_targets)

    constraints = _block_constraints(blocks, block_targets)
    constraints.append({
        "type": "eq",
        "fun": lambda w, tr=float(target_return): float(np.dot(w, mu)) - tr
    })

    res = minimize(
        port_var,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 4000, "ftol": 1e-12}
    )
    return res


def extreme_return_with_blocks(mean_returns, blocks, block_targets, maximize=True, max_weight=1.0):
    """
    Encuentra retorno mínimo o máximo factible bajo constraints (sin mirar varianza).
    maximize=True => max mu'w
    maximize=False => min mu'w
    Devuelve res (OptimizeResult)
    """
    mu = mean_returns.values
    n = len(mu)

    sign = -1.0 if maximize else 1.0

    def obj(w):
        return float(sign * np.dot(w, mu))

    bounds = tuple((0.0, float(max_weight)) for _ in range(n))
    w0 = _init_w0_from_block_targets(n, blocks, block_targets)
    constraints = _block_constraints(blocks, block_targets)

    res = minimize(
        obj,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 3000, "ftol": 1e-12}
    )
    return res


def min_variance_with_blocks_min_return(cov_matrix, mean_returns, blocks, block_targets,
                                        target_return, max_weight=1.0):
    """
    Min var long-only con:
      - sum(w)=1
      - 0<=w<=max_weight
      - sum(w en bloque)=target_bloque
      - mu'w >= target_return   (INEQ, más estable que igualdad)
    Devuelve res (OptimizeResult)
    """
    mu = mean_returns.values
    Sigma = cov_matrix.values
    n = len(mu)

    def portfolio_var(w):
        return float(np.dot(w.T, np.dot(Sigma, w)))

    w0 = _init_w0_from_block_targets(n, blocks, block_targets)
    bounds = tuple((0.0, float(max_weight)) for _ in range(n))

    constraints = _block_constraints(blocks, block_targets)
    constraints.append({
        "type": "ineq",
        "fun": lambda w, tr=float(target_return): float(np.dot(w, mu)) - tr
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