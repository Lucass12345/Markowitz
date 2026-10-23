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
        for v, r in frontier_pts
    ]
    df["return_annual"] = [x[0] for x in annualized]
    df["vol_annual"] = [x[1] for x in annualized]

    df["return_log_annual"] = df["return_annual"]
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