"""Signal compatibility and row decisions, independent of model artifacts/DB."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.data.load_data import active_data_dir
from src.optimization.errors import OptimizationInputError


@dataclass(frozen=True)
class PredictionPolicy:
    fallback: str = "historical"
    max_age_days: int = 7

    def __post_init__(self):
        if self.fallback not in {"historical", "required"} or self.max_age_days < 0:
            raise OptimizationInputError("Invalid prediction fallback or maximum age")


def checked_predictions(predictions, returns, value_column):
    if predictions.empty:
        return pd.DataFrame()
    frame = predictions.copy()
    if not {"ticker", value_column, "dataset_version", "base_currency"}.issubset(frame):
        raise OptimizationInputError("Predictions require ticker, daily value, dataset_version and base_currency. Regenerate legacy predictions.")
    frame["ticker"] = frame.ticker.astype(str).str.strip()
    if frame.ticker.duplicated().any():
        raise OptimizationInputError("Prediction tickers must be unique")
    expected = returns.attrs.get("dataset_version") or active_data_dir().name
    if not frame.dataset_version.eq(expected).all():
        raise OptimizationInputError(f"Predictions are incompatible with dataset_version={expected}")
    if not frame.base_currency.eq("USD").all():
        raise OptimizationInputError("Predictions must describe USD returns/volatility")
    # Existing column contracts are daily-log quantities, including GARCH vol_pred.
    expected_units = "daily_log_volatility" if value_column == "vol_pred" else "daily_log_return"
    if "prediction_units" in frame and not frame.prediction_units.eq(expected_units).all():
        raise OptimizationInputError(f"Prediction units must be {expected_units}")
    if "as_of" not in frame and "train_end" in frame:
        frame["as_of"] = frame["train_end"]
    if "as_of" not in frame:
        raise OptimizationInputError("Predictions require as_of (GARCH train_end is accepted)")
    return frame.set_index("ticker")


def row_reason(row, series, value_column, policy):
    if row is None:
        return "missing_prediction"
    try:
        value = float(row[value_column])
    except (TypeError, ValueError):
        return "invalid_numeric_prediction"
    if not np.isfinite(value) or (value_column == "vol_pred" and value <= 0):
        return "non_finite_or_non_positive_prediction"
    date = pd.to_datetime(row.get("as_of"), errors="coerce", utc=True)
    if pd.isna(date):
        return "invalid_as_of"
    cutoff = pd.Timestamp(series.dropna().index.max())
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    age = (cutoff.normalize() - date.normalize()).days
    if age < 0:
        return "prediction_uses_future_data"
    if age > policy.max_age_days:
        return "stale_prediction"
    return None


def require_or_fallback(ticker, reason, policy):
    if reason and policy.fallback == "required":
        raise OptimizationInputError(f"Required prediction unavailable for {ticker}: {reason}")
