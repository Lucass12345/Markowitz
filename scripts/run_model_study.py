"""Repeatable feature / architecture / Adam study on a fixed chronological split."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import torch
import numpy as np
import arch
import yfinance
from src.models.train_returns_lstm_lab import run_lstm_lab
from src.models.lstm_features import AVAILABLE_FEATURES
from src.models.vol_model_garch import diagnose_garch
from src.data.load_data import load_returns, active_data_dir

def refit_winners(winners, snapshot, dataset, epochs, start, output):
    forecasts, models = [], {}
    for _, winner in winners.iterrows():
        ticker = winner["ticker"]
        print(f"Refitting selected model on latest history: {ticker}", flush=True)
        features = ["return"] if winner["features"] == "returns_only" else AVAILABLE_FEATURES
        result = run_lstm_lab(features, epochs=epochs, start_date=start,
                              tickers=[ticker], candidates=[json.loads(winner["selected_config"])],
                              persist_db=False, publish_latest=False, returns_df=snapshot,
                              dataset_version=Path(dataset).name, refit_latest=True)
        if result["predictions"].empty:
            raise RuntimeError(f"Selected model refit failed: {ticker}")
        forecasts.append(result["predictions"])
        models[ticker] = {"model_dir": result["model_dir"], "run_id": result["run_id"],
                          "features": winner["features"], "selected_config": json.loads(winner["selected_config"])}
    pd.concat(forecasts, ignore_index=True).to_csv(Path(output) / "selected_forecasts.csv", index=False)
    (Path(output) / "selected_checkpoints.json").write_text(json.dumps(models, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "TLT", "GGAL.BA"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--output", default="outputs/model_search")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = str(active_data_dir())
    snapshot = load_returns(data_dir=dataset)
    rows = []
    for name, features in [("returns_only", ["return"]), ("return_statistics", AVAILABLE_FEATURES)]:
        print(f"Training {name}: {args.tickers}", flush=True)
        result = run_lstm_lab(features, epochs=args.epochs, start_date=args.start,
                              tickers=args.tickers, persist_db=False, publish_latest=False, returns_df=snapshot, dataset_version=Path(dataset).name)
        metrics = result["metrics"].copy()
        metrics["features"] = name
        metrics["run_id"] = result["run_id"]
        rows.append(metrics)
    comparison = pd.concat(rows, ignore_index=True)
    comparison.to_csv(output / "feature_comparison.csv", index=False)
    eligible = comparison.dropna(subset=["rmse_val"])
    winners = eligible.loc[eligible.groupby("ticker").rmse_val.idxmin()]
    winners.to_csv(output / "selected_models.csv", index=False)
    refit_winners(winners, snapshot, dataset, args.epochs, args.start, output)
    returns = snapshot.loc[args.start:]
    diagnostics = {}
    for ticker in args.tickers:
        print(f"Validating GARCH: {ticker}", flush=True)
        diagnostics[ticker] = diagnose_garch(returns[ticker], test_size=20)
    (output / "garch_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    (output / "study.json").write_text(json.dumps(dict(tickers=args.tickers, epochs=args.epochs,
        start=args.start, dataset=dataset, selection="lowest validation RMSE, including feature set",
        test_used_for_selection=False, seed=42,
        versions={"torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__, "arch": arch.__version__, "yfinance": yfinance.__version__}, runs=comparison.run_id.unique().tolist()), indent=2))
    print(winners[["ticker", "features", "selected_config", "rmse_val", "rmse_test", "baseline_rmse_test"]].to_string(index=False))
