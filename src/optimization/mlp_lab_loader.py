from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.load_data import assert_current_dataset


def load_active_mlp_predictions(validate_current: bool = True) -> pd.DataFrame:
    # Pin the selected run once: its database predictions and local provenance
    # must come from the same run even if another run is activated concurrently.
    from app.mlp_lab_storage import (
        get_active_mlp_prediction_run,
        get_mlp_run_tickers_df,
        _apply_selection_policy,
    )

    active = get_active_mlp_prediction_run()
    if active is None:
        return pd.DataFrame()
    predictions = get_mlp_run_tickers_df(active["run_id"]).copy()
    if predictions.empty:
        return predictions
    predictions["ticker"] = predictions["ticker"].astype(str).str.strip()
    predictions = _apply_selection_policy(predictions, active.get("selection_policy_json"))

    directory = active.get("run_output_dir") or active.get("local_run_output_dir")
    if directory:
        directory = Path(directory)
        forecast_file = directory / "predictions.csv"
        if forecast_file.exists():
            stored = pd.read_csv(forecast_file)
            columns = [name for name in ("dataset_version", "base_currency", "as_of", "prediction_units")
                       if name in stored and name not in predictions]
            if columns:
                predictions = predictions.merge(stored[["ticker"] + columns], on="ticker", how="left", validate="one_to_one")
        config_file = directory / "config.json"
        if config_file.exists():
            config = json.loads(config_file.read_text(encoding="utf-8"))
            for name in ("dataset_version", "base_currency", "as_of", "prediction_units"):
                if name not in predictions and config.get(name) is not None:
                    predictions[name] = config[name]

    if "base_currency" not in predictions or not predictions["base_currency"].eq("USD").all():
        raise ValueError("Las predicciones MLP no tienen moneda base USD verificable. Reentrená el modelo.")
    if validate_current:
        assert_current_dataset(predictions)
    predictions["model_run_id"] = active["run_id"]
    return predictions
