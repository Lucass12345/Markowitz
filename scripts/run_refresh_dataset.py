"""Refresh a Yahoo universe; optionally retrain both model families."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.yahoo_dataset import refresh_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--ccl-file", help="CSV histórico CCL ARS/USD: fecha,cierre o date,rate")
    parser.add_argument("--max-fx-age-days", type=int, default=7)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    if not args.tickers:
        from src.data.load_data import load_metadata
        args.tickers = load_metadata()["ticker"].tolist()
    print(refresh_dataset(args.tickers, start=args.start, ccl_file=args.ccl_file, max_fx_age_days=args.max_fx_age_days))
    if args.retrain:
        from src.models.train_returns_lstm import train_all_return_models_lstm
        from src.models.train_volatility import forecast_all_volatility
        train_all_return_models_lstm(epochs=args.epochs)
        forecast_all_volatility()
