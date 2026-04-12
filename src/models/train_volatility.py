from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.data.load_data import load_returns
from src.models.vol_model_garch import forecast_garch_vol
from src.models.baselines.vol_model_ewma import next_day_variance_forecast

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "predictions"


def forecast_all_volatility(
    start_date: str = "2022-01-01",
    min_obs: int = 252,
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
    returns = load_returns().copy()
    returns.index = pd.to_datetime(returns.index)

    if start_date:
        returns = returns.loc[returns.index >= pd.Timestamp(start_date)].copy()

    rows = []

    for ticker in returns.columns:
        s = returns[ticker].dropna().astype(float)

        if len(s) < min_obs:
            continue

        source = "gjr_garch_t"

        try:
            var_pred, vol_pred = forecast_garch_vol(s)
        except Exception:
            var_pred = float(next_day_variance_forecast(s))
            vol_pred = float(var_pred ** 0.5)
            source = "ewma_fallback"

        rows.append({
            "ticker": ticker,
            "var_pred": var_pred,
            "vol_pred": vol_pred,
            "model_source": source,
            "n_obs": int(len(s)),
            "train_start": str(s.index.min().date()),
            "train_end": str(s.index.max().date()),
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
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
