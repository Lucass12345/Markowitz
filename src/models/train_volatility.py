from pathlib import Path
import pandas as pd

from src.data.load_data import load_returns
from src.models.vol_model_garch import forecast_garch_vol


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT / "outputs" / "predictions"


def forecast_all_volatility():

    returns = load_returns()

    rows = []

    for ticker in returns.columns:

        s = returns[ticker].dropna()

        if len(s) < 60:
            continue

        # GARCH forecast
        var_pred, vol_pred = forecast_garch_vol(s)

        rows.append({
            "ticker": ticker,
            "var_pred": var_pred,
            "vol_pred": vol_pred
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)

    df.to_csv(OUTPUT_DIR / "volatility_predictions.csv", index=False)

    print(df.head())
