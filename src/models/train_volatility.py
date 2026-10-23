from __future__ import annotations

from pathlib import Path
import pandas as pd
import json

from src.data.load_data import load_returns, active_data_dir
from src.models.vol_model_garch import forecast_garch_vol, diagnose_garch
from src.models.baselines.vol_model_ewma import next_day_variance_forecast

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "predictions"


def forecast_all_volatility(
    start_date: str = "2022-01-01",
    min_obs: int = 252,
    validation_days: int = 20,
):
    """
    Generates one-step-ahead volatility forecasts for all assets.

    Main model:
    - GJR-GARCH(1,1,1) with Student-t innovations

    Fallback:
    - EWMA one-step variance forecast

    Output:
    - outputs/predictions/volatility_predictions.csv
    """
    dataset_dir = active_data_dir()
    returns = load_returns(data_dir=dataset_dir).copy()
    returns.index = pd.to_datetime(returns.index)

    if start_date:
        returns = returns.loc[returns.index >= pd.Timestamp(start_date)].copy()

    rows = []
    diagnostics = {}

    for ticker in returns.columns:
        s = returns[ticker].dropna().astype(float)

        if len(s) < min_obs:
            diagnostics[ticker] = {"error": "insufficient_history", "n_obs": len(s)}
            continue

        source = "gjr_garch_t"
        details = {}
        error = None

        try:
            details = diagnose_garch(s, horizon=5, test_size=validation_days)
            diagnostics[ticker] = details
            var_pred = details["variance_by_day"][0]
            vol_pred = float(var_pred ** .5)
        except Exception as exc:
            error = str(exc)
            diagnostics[ticker] = {"error": error, "converged": False}
            var_pred = float(next_day_variance_forecast(s))
            vol_pred = float(var_pred ** 0.5)
            source = "ewma_fallback"

        rows.append({
            "ticker": ticker,
            "var_pred": var_pred,
            "vol_pred": vol_pred,
            "model_source": source,
            "dataset_version": dataset_dir.name,
            "base_currency": "USD",
            "fallback_reason": error,
            "forecast_horizon": 5,
            "var_5d": details.get("variance_sum_5d", var_pred * 5),
            "vol_5d": details.get("vol_5d", float((var_pred * 5) ** .5)),
            "persistence": details.get("persistence"),
            "residual_tests_pass": details.get("residual_tests_pass"),
            "ljung_box_squared_resid_p": details.get("ljung_box_squared_resid_p"),
            "arch_lm_p": details.get("arch_lm_p"),
            "backtest_qlike": (details.get("backtest", {}).get("garch") or {}).get("qlike"),
            "ewma_qlike": (details.get("backtest", {}).get("ewma") or {}).get("qlike"),
            "n_obs": int(len(s)),
            "train_start": str(s.index.min().date()),
            "train_end": str(s.index.max().date()),
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True) if rows else pd.DataFrame(columns=["ticker", "var_pred", "vol_pred"])
    (OUTPUT_DIR / "garch_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    df.to_csv(OUTPUT_DIR / "volatility_predictions.csv", index=False)

    print("\nVolatility forecast file generated:")
    print(OUTPUT_DIR / "volatility_predictions.csv")
    print("\nModel source counts:")
    if not df.empty:
        print(df["model_source"].value_counts())
        print("\nPreview:")
        print(df.head())
    else:
        print("No forecasts generated.")

    return df
