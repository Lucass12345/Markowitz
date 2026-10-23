import numpy as np
import pandas as pd

from src.markowitz_classic import (
    min_variance,
    max_sharpe,
    efficient_frontier,
)
from src.utils_portfolio import annualize_log, portfolio_statistics


def _stats_dict(
    weights,
    mu,
    cov,
    rf_log_daily: float = 0.0,
    periods: int = 252,
) -> dict:
    return portfolio_statistics(weights, mu, cov, rf_log_daily, periods)


def _clean_frontier(frontier_pts, periods: int = 252) -> pd.DataFrame:
    """
    Limpia la frontera para quedarse solo con la rama eficiente:
    - anualiza
    - ordena por volatilidad
    - corta desde el punto de mínima volatilidad
    - elimina retrocesos numéricos raros
    """
    df = pd.DataFrame(frontier_pts, columns=["vol_daily", "return_daily"]).dropna().copy()

    # anualizar
    annualized = [
        annualize_log(r, v, periods=periods)
        for v, r in df[["vol_daily", "return_daily"]].itertuples(index=False, name=None)
    ]
    df["return_annual"] = [x[0] for x in annualized]
    df["vol_annual"] = [x[1] for x in annualized]

    df["return_log_annual"] = df["return_annual"]
    if df.empty:
        return df
    # ordenar por volatilidad
    df = df.sort_values("vol_annual").reset_index(drop=True)

    # quedarse desde el punto de mínima volatilidad hacia la derecha
    idx_min_vol = df["vol_annual"].idxmin()
    df = df.loc[idx_min_vol:].copy().reset_index(drop=True)

    # eliminar la rama ineficiente / serruchos:
    # nos quedamos solo con puntos cuyo retorno sea no decreciente
    cleaned_rows = []
    best_ret_so_far = -np.inf

    for _, row in df.iterrows():
        if row["return_annual"] >= best_ret_so_far - 1e-10:
            cleaned_rows.append(row)
            best_ret_so_far = max(best_ret_so_far, row["return_annual"])

    df_clean = pd.DataFrame(cleaned_rows).reset_index(drop=True)

    return df_clean


def run_frontier(
    mu,
    cov,
    rf_log_daily: float = 0.0,
    periods: int = 252,
    n_points: int = 80,
) -> dict:
    """
    Corre frontera, min var y max sharpe para el modo seleccionado.
    """
    tickers = list(mu.index) if hasattr(mu, "index") else list(range(len(mu)))

    w_min = min_variance(mu, cov)
    w_max = max_sharpe(mu, cov, risk_free_rate=rf_log_daily)

    stats_min = _stats_dict(w_min, mu, cov, rf_log_daily=rf_log_daily, periods=periods)
    stats_max = _stats_dict(w_max, mu, cov, rf_log_daily=rf_log_daily, periods=periods)

    frontier_pts = efficient_frontier(mu, cov, n_points=n_points)
    frontier_df = _clean_frontier(frontier_pts, periods=periods)

    w_min_df = pd.DataFrame({"ticker": tickers, "weight": w_min}).sort_values(
        "weight", ascending=False
    ).reset_index(drop=True)

    w_max_df = pd.DataFrame({"ticker": tickers, "weight": w_max}).sort_values(
        "weight", ascending=False
    ).reset_index(drop=True)

    return {
        "frontier": frontier_df,
        "minvar": {
            "weights_array": w_min,
            "weights_df": w_min_df,
            "stats": stats_min,
        },
        "maxsharpe": {
            "weights_array": w_max,
            "weights_df": w_max_df,
            "stats": stats_max,
        },
    }

def check_frontier_weights(weights, mu, blocks, targets, max_weight, min_return=None, tolerance=1e-7):
    w = np.asarray(weights, dtype=float)
    if not np.isfinite(w).all() or abs(w.sum() - 1) > tolerance or w.min() < -tolerance or w.max() > max_weight + tolerance:
        return False
    if any(abs(w[blocks[name]].sum() - target) > tolerance for name, target in targets.items()):
        return False
    return min_return is None or float(w @ np.asarray(mu)) >= min_return - tolerance


def run_constrained_frontier(mu, cov, blocks, targets, max_weight=1., periods=252, n_points=80):
    """Upper efficient branch with the exact allocation constraints of the portfolio."""
    from src.markowitz import min_variance_with_blocks, extreme_return_with_blocks, min_variance_with_blocks_min_return
    minimum = min_variance_with_blocks(cov, blocks, targets, max_weight)
    maximum = extreme_return_with_blocks(mu, blocks, targets, max_weight=max_weight)
    for result in (minimum, maximum):
        if not result.success or not check_frontier_weights(result.x, mu, blocks, targets, max_weight):
            raise RuntimeError(f"Constrained frontier endpoint did not converge: {result.message}")
    lower, upper = float(np.asarray(mu) @ minimum.x), float(np.asarray(mu) @ maximum.x)
    grid = [lower] if upper - lower < 1e-12 else np.linspace(lower, upper, n_points)
    rows, failed = [], []
    for target in grid:
        result = minimum if abs(target - lower) < 1e-14 else min_variance_with_blocks_min_return(cov, mu, blocks, targets, target, max_weight)
        if not result.success or not check_frontier_weights(result.x, mu, blocks, targets, max_weight, target):
            failed.append({"target_return_daily": float(target), "message": str(result.message)})
            continue
        ret = float(np.asarray(mu) @ result.x)
        vol = float(np.sqrt(result.x @ np.asarray(cov) @ result.x))
        rows.append({"return_daily": ret, "return_annual": ret * periods, "return_log_annual": ret * periods,
                     "vol_daily": vol, "vol_annual": vol * np.sqrt(periods), "weights": result.x.tolist()})
    if not rows:
        raise RuntimeError("Constrained frontier has no verified feasible points")
    return {"points": pd.DataFrame(rows), "diagnostics": {"requested_points": len(grid), "accepted_points": len(rows), "failed_points": failed,
            "constraints": {"targets": targets, "max_weight": max_weight, "long_only": True, "budget": 1.}, "tickers": list(mu.index)}}
