from src.config import TRADING_DAYS
from src.markowitz import rf_simple_annual_to_log_daily
from src.data.load_data import load_returns, load_metadata
from src.optimization.inputs_builder import build_model_inputs
from src.optimization.frontier_engine import run_frontier
from src.optimization.optimize_blocks import run_blocks_optimizer

def optimize_portfolio_mode(
    model_type: str,
    targets: dict,
    max_weight: float = 0.15,
    rf_annual_simple: float = 0.0,
    periods: int = TRADING_DAYS,
    n_frontier_points: int = 80,
) -> dict:
    """
    Orquestador principal para los 3 modos:
    - classic
    - ai_returns
    - ai_volatility
    """
    returns = load_returns().dropna(how="any")
    meta = load_metadata()

    rf_log_daily = rf_simple_annual_to_log_daily(rf_annual_simple, periods=periods)

    mu, cov, tickers = build_model_inputs(
        model_type=model_type,
        returns=returns,
    )

    if hasattr(mu, "index") and hasattr(cov, "loc"):
        cov = cov.loc[mu.index, mu.index]
        tickers = list(mu.index)

    frontier_result = run_frontier(
        mu=mu,
        cov=cov,
        rf_log_daily=rf_log_daily,
        periods=periods,
        n_points=n_frontier_points,
    )

    blocks_result = run_blocks_optimizer(
        mu=mu,
        cov=cov,
        tickers=tickers,
        meta=meta,
        targets=targets,
        max_weight=max_weight,
        rf_log_daily=rf_log_daily,
        periods=periods,
    )

    return {
        "model_type": model_type,
        "inputs": {
            "targets": targets,
            "max_weight": max_weight,
            "rf_annual_simple": rf_annual_simple,
            "rf_log_daily": rf_log_daily,
            "periods": periods,
        },
        "frontier": frontier_result["frontier"],
        "minvar": frontier_result["minvar"],
        "maxsharpe": frontier_result["maxsharpe"],
        "blocks": blocks_result,
    }
