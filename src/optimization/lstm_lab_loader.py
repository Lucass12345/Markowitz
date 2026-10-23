from __future__ import annotations

from pathlib import Path
import pandas as pd
import json

from sqlalchemy.exc import SQLAlchemyError

ROOT = Path(__file__).resolve().parents[2]


def _compute_amp_ratio_from_run_dir(run_output_dir: str | None) -> pd.DataFrame:
    if not run_output_dir:
        return pd.DataFrame(columns=["ticker", "std_y_true_val", "std_y_pred_val", "amp_ratio"])

    per_ticker_dir = Path(run_output_dir) / "per_ticker"
    if not per_ticker_dir.exists():
        return pd.DataFrame(columns=["ticker", "std_y_true_val", "std_y_pred_val", "amp_ratio"])

    rows: list[dict] = []
    for csv_path in sorted(per_ticker_dir.glob("*_predictions.csv")):
        ticker = csv_path.name.replace("_predictions.csv", "")
        try:
            df = pd.read_csv(csv_path)
            validation_df = df.loc[df["split"].astype(str).str.lower() == "val", ["y_true", "y_pred"]].copy()
            if validation_df.empty:
                continue
            std_y_true = float(pd.to_numeric(validation_df["y_true"], errors="coerce").std())
            std_y_pred = float(pd.to_numeric(validation_df["y_pred"], errors="coerce").std())
            amp_ratio = (std_y_pred / std_y_true) if pd.notna(std_y_true) and std_y_true > 0 else float("nan")
            rows.append({
                "ticker": ticker,
                "std_y_true_val": std_y_true,
                "std_y_pred_val": std_y_pred,
                "amp_ratio": amp_ratio,
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


def load_active_lstm_predictions_df() -> pd.DataFrame:
    def provenance(preds, directory):
        forecast_file = Path(directory) / "predictions.csv" if directory else None
        if forecast_file and forecast_file.exists():
            extra = pd.read_csv(forecast_file)
            columns = [c for c in ["dataset_version", "use_for_optimizer", "as_of", "base_currency", "forecast_horizon", "prediction_units"] if c in extra and c not in preds]
            if columns:
                preds = preds.merge(extra[["ticker"] + columns], on="ticker", how="left", validate="one_to_one")
        return preds

    # Honor an explicitly active, current laboratory run; otherwise use automation.
    database_error = None
    try:
        from app.lstm_lab_storage import get_active_prediction_run, load_active_predictions_df
        active = get_active_prediction_run()
        if active:
            run_output_dir = active.get("local_run_output_dir")
            preds = provenance(load_active_predictions_df(run_id=active["run_id"]).copy(), run_output_dir)
            if not preds.empty:
                preds["model_run_id"] = active["run_id"]
                amp = _compute_amp_ratio_from_run_dir(run_output_dir)
                return preds.merge(amp, on="ticker", how="left", validate="one_to_one") if not amp.empty else preds.assign(amp_ratio=float("nan"))
    except (ImportError, SQLAlchemyError, ConnectionError) as exc:
        database_error = exc

    local = ROOT / "outputs" / "predictions" / "return_predictions_lstm.csv"
    pointer = local.parent / "lstm_production.json"
    if not local.exists() or not pointer.exists():
        raise FileNotFoundError("No hay predicciones LSTM vigentes. Ejecutá el entrenamiento del laboratorio o la automatización.") from database_error
    info = json.loads(pointer.read_text(encoding="utf-8"))
    preds = pd.read_csv(local)
    preds["model_run_id"] = info.get("run_id", Path(info["run_output_dir"]).name)
    amp = _compute_amp_ratio_from_run_dir(info["run_output_dir"])
    return preds.merge(amp, on="ticker", how="left", validate="one_to_one") if not amp.empty else preds.assign(amp_ratio=float("nan"))
