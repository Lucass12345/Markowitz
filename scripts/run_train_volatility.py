import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.models.train_volatility import forecast_all_volatility


if __name__ == "__main__":

    forecast_all_volatility()

    print("\nVolatility predictions generated")