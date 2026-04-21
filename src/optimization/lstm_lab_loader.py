from __future__ import annotations

from pathlib import Path
import pandas as pd

from app.lstm_lab_storage import get_active_prediction_run, load_active_predictions_df


def _compute_amp_ratio_from_run_dir(run_output_dir: str | None) -> pd.DataFrame:
    if not run_output_dir:
        return pd.DataFrame(columns=["ticker", "std_y_true_test", "std_y_pred_test", "amp_ratio"])

    per_ticker_dir = Path(run_output_dir) / "per_ticker"
    if not per_ticker_dir.exists():
        return pd.DataFrame(columns=["ticker", "std_y_true_test", "std_y_pred_test", "amp_ratio"])

    rows: list[dict] = []
    for csv_path in sorted(per_ticker_dir.glob("*_predictions.csv")):
        ticker = csv_path.name.replace("_predictions.csv", "")
        try:
            df = pd.read_csv(csv_path)
            test_df = df.loc[df["split"].astype(str).str.lower() == "test", ["y_true", "y_pred"]].copy()
            if test_df.empty:
                continue
            std_y_true = float(pd.to_numeric(test_df["y_true"], errors="coerce").std())
            std_y_pred = float(pd.to_numeric(test_df["y_pred"], errors="coerce").std())
            amp_ratio = (std_y_pred / std_y_true) if pd.notna(std_y_true) and std_y_true > 0 else float("nan")
            rows.append({
                "ticker": ticker,
                "std_y_true_test": std_y_true,
                "std_y_pred_test": std_y_pred,
                "amp_ratio": amp_ratio,
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


def load_active_lstm_predictions_df() -> pd.DataFrame:
    preds = load_active_predictions_df().copy()
    active_run = get_active_prediction_run()
    run_output_dir = active_run.get("local_run_output_dir") if active_run else None
    amp_df = _compute_amp_ratio_from_run_dir(run_output_dir)
    if amp_df.empty:
        preds["amp_ratio"] = float("nan")
        return preds
    return preds.merge(amp_df, on="ticker", how="left")
