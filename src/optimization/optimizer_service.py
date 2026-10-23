from dataclasses import asdict
from src.config import TRADING_DAYS
from src.markowitz import rf_simple_annual_to_log_daily
from src.data.load_data import load_returns, load_metadata, active_data_dir
from src.optimization.inputs_builder import build_model_inputs
from src.optimization.covariance import RiskConfig
from src.optimization.frontier_engine import run_frontier
from src.optimization.optimize_blocks import run_blocks_optimizer, validate_optimizer_inputs

def optimize_portfolio_mode(
    model_type: str,
    targets: dict,
    max_weight: float = 0.15,
    rf_annual_simple: float = 0.0,
    periods: int = TRADING_DAYS,
    n_frontier_points: int = 80,
    min_history: int = 252,
    min_overlap: int = 126,
    eigenvalue_floor: float = 1e-8,
    max_psd_adjustment: float = 1.0,
    prediction_fallback: str = "historical",
    max_prediction_age_days: int = 7,
) -> dict:
    """
    Orquestador principal para los 3 modos:
    - classic
    - ai_returns
    - ai_volatility
    """
    dataset_dir = active_data_dir()
    returns = load_returns(data_dir=dataset_dir)
    meta = load_metadata(data_dir=dataset_dir)
    risk_config = RiskConfig(min_history, min_overlap, eigenvalue_floor, max_psd_adjustment)

    rf_log_daily = rf_simple_annual_to_log_daily(rf_annual_simple, periods=periods)

    mu, cov, tickers = build_model_inputs(
        model_type=model_type,
        returns=returns,
        risk_config=risk_config,
        prediction_fallback=prediction_fallback,
        max_prediction_age_days=max_prediction_age_days,
    )

    validate_optimizer_inputs(mu, cov, tickers, meta, targets, max_weight)
    if hasattr(cov, "loc"):
        cov = cov.loc[tickers, tickers]

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
            "risk_config": asdict(risk_config),
            "prediction_fallback": prediction_fallback,
            "max_prediction_age_days": max_prediction_age_days,
            "dataset_version": dataset_dir.name,
            "base_currency": "USD",
        },
        "diagnostics": {"risk": mu.attrs.get("input_diagnostics", {})},
        "frontier": frontier_result["frontier"],
        "minvar": frontier_result["minvar"],
        "maxsharpe": frontier_result["maxsharpe"],
        "blocks": blocks_result,
    }
