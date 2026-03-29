import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.models.train_volatility import forecast_all_volatility


if __name__ == "__main__":
    df = forecast_all_volatility(
        start_date="2022-01-01",
        min_obs=252,
    )

    print("\nVolatility predictions generated successfully.")
    print(f"Rows: {len(df)}")
