from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError
from src.optimization.errors import OptimizationInputError
from src.optimization.covariance import RiskConfig, estimate_historical_risk, regularize_covariance, ledoit_wolf_covariance
from src.optimization.predictions import PredictionPolicy, checked_predictions, row_reason, require_or_fallback, LSTMSelection
from src.optimization.lstm_lab_loader import load_active_lstm_predictions_df
from src.optimization.mlp_lab_loader import load_active_mlp_predictions

ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = ROOT / "outputs" / "predictions"
LSTM_DIR_ACC_MIN = .52
LSTM_AMP_RATIO_MIN = .15
_UNSET_PREDICTIONS = object()


def _read_prediction_csv(filename):
    path = PRED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")
    return pd.read_csv(path)


def load_predictions(prediction_file="predictions.csv"):
    return _read_prediction_csv(prediction_file)


def load_volatility_predictions(prediction_file="volatility_predictions.csv"):
    return _read_prediction_csv(prediction_file)


def _load_signals(mode):
    if mode == "classic":
        return pd.DataFrame()
    try:
        if mode == "ai_returns_lstm":
            return load_active_lstm_predictions_df()
        if mode == "ai_volatility":
            return load_volatility_predictions()
        if mode == "ai_returns":
            try:
                predictions = load_active_mlp_predictions(validate_current=False)
            except (ImportError, SQLAlchemyError):
                predictions = pd.DataFrame()
            return predictions if not predictions.empty else load_predictions()
    except FileNotFoundError:
        return pd.DataFrame()
    except ValueError as exc:
        raise OptimizationInputError(str(exc)) from exc
    raise OptimizationInputError(f"Invalid model_type: {mode}")


def _pinned_signals(mode, returns):
    released = returns.attrs.get("release_dir")
    if released and mode in {"ai_returns_lstm", "ai_volatility"}:
        name = "lstm" if mode == "ai_returns_lstm" else "garch"
        return pd.read_csv(Path(released) / "predictions" / f"{name}.csv")
    return _load_signals(mode)


def _explicit_or_pinned(value, mode, returns):
    if value is _UNSET_PREDICTIONS:
        return _pinned_signals(mode, returns)
    if value is None:
        return pd.DataFrame()
    return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value)


def _build_ai_vol_cov(historical_cov, pred_vol):
    historical_vol = np.sqrt(np.diag(historical_cov))
    corr = historical_cov / np.outer(historical_vol, historical_vol)
    return corr.mul(pred_vol, axis=0).mul(pred_vol, axis=1)


def _finite_metric(row, key):
    value = pd.to_numeric(row.get(key), errors="coerce") if row is not None else np.nan
    return float(value) if value is not None and np.isfinite(value) else None


def build_model_inputs(model_type, returns, risk_config=None, prediction_fallback="historical", max_prediction_age_days=7, covariance_estimator=None, lstm_direction_min=.52, lstm_amplitude_min=.15, return_predictions=_UNSET_PREDICTIONS, volatility_predictions=_UNSET_PREDICTIONS):
    model_type = str(model_type).strip().lower()
    if model_type not in {"classic", "ai_returns", "ai_returns_lstm", "ai_volatility"}:
        raise OptimizationInputError(f"Invalid model_type: {model_type}")
    risk_config = risk_config or RiskConfig()
    selection = LSTMSelection(lstm_direction_min, lstm_amplitude_min)
    policy = PredictionPolicy(prediction_fallback, max_prediction_age_days)
    returns, cov, diagnostics = estimate_historical_risk(returns, risk_config)
    estimator = covariance_estimator or ("gjr_garch" if model_type == "ai_volatility" else "historical")
    if estimator not in {"historical", "ledoit_wolf", "gjr_garch"}:
        raise OptimizationInputError(f"Invalid covariance estimator: {estimator}")
    if estimator == "ledoit_wolf":
        cov, selected_diagnostics = ledoit_wolf_covariance(returns, risk_config)
        diagnostics["selected_estimator_diagnostics"] = selected_diagnostics
    diagnostics["estimator"] = estimator
    mu = returns.mean()
    value_column = "vol_pred" if model_type == "ai_volatility" else "mu_pred"
    if model_type == "classic" or (model_type == "ai_volatility" and estimator != "gjr_garch"):
        raw_predictions = pd.DataFrame()
    elif model_type == "ai_volatility":
        raw_predictions = _explicit_or_pinned(volatility_predictions, model_type, returns)
    else:
        raw_predictions = _explicit_or_pinned(return_predictions, model_type, returns)
    predictions = checked_predictions(raw_predictions, returns, value_column)
    vol = pd.Series(np.sqrt(np.diag(cov)), index=cov.index)
    signals = []
    for ticker in returns:
        row = predictions.loc[ticker] if ticker in predictions.index else None
        reason = "classic_historical" if model_type == "classic" or (model_type == "ai_volatility" and estimator != "gjr_garch") else row_reason(row, returns[ticker], value_column, policy)
        if reason is None and model_type == "ai_returns_lstm":
            direction = pd.to_numeric(row.get("dir_acc_val"), errors="coerce")
            amplitude = pd.to_numeric(row.get("amp_ratio"), errors="coerce")
            if not np.isfinite(direction) or not direction > selection.direction_min:
                reason = "validation_direction_below_threshold"
            elif not np.isfinite(amplitude) or not amplitude > selection.amplitude_min:
                reason = "validation_amplitude_below_threshold"
        if reason is None and row is not None and "use_for_optimizer" in row and str(row["use_for_optimizer"]).lower() != "true":
            reason = "producer_validation_rejected"
        if model_type != "classic" and not (model_type == "ai_volatility" and estimator != "gjr_garch"):
            require_or_fallback(ticker, reason, policy)
        if reason is None:
            if model_type == "ai_volatility":
                vol[ticker] = float(row[value_column])
            else:
                mu[ticker] = float(row[value_column])
        signal = {"ticker": ticker, "source": "ai" if reason is None else "historical", "reason": reason or "accepted",
                  "as_of": str(row.get("as_of")) if row is not None else None,
                  "dir_acc_val": _finite_metric(row, "dir_acc_val"),
                  "amp_ratio": _finite_metric(row, "amp_ratio")}
        if row is not None:
            for key in ("model_run_id", "model_source", "dataset_version", "selection_reason", "selection_split", "registry_model_id", "artifact_uri"):
                if key in row:
                    signal[key] = str(row[key])
        if row is not None and "selection_policy" in row:
            signal["producer_selection_policy"] = row["selection_policy"]
        signals.append(signal)
    if estimator == "gjr_garch":
        if model_type != "ai_volatility":
            checked_volatility = checked_predictions(_explicit_or_pinned(volatility_predictions, "ai_volatility", returns), returns, "vol_pred")
            risk_signals = []
            for ticker in returns:
                row = checked_volatility.loc[ticker] if ticker in checked_volatility.index else None
                reason = row_reason(row, returns[ticker], "vol_pred", policy)
                require_or_fallback(ticker, reason, policy)
                if reason is None:
                    vol[ticker] = float(row.vol_pred)
                risk_signals.append({"ticker": ticker, "source": "ai" if reason is None else "historical", "reason": reason or "accepted",
                                     **{key: str(row[key]) for key in ("model_run_id", "model_source", "dataset_version", "as_of", "registry_model_id", "artifact_uri") if row is not None and key in row}})
            diagnostics["risk_signals"] = risk_signals
        cov = _build_ai_vol_cov(cov, vol)
    cov, final_diagnostics = regularize_covariance(cov, risk_config)
    mu.attrs["input_diagnostics"] = {**diagnostics, "final_covariance": final_diagnostics,
                                    "signals": signals, "lstm_selection": {"direction_min": selection.direction_min, "amplitude_min": selection.amplitude_min, "split": "validation", "comparison": "strictly_greater"}, "prediction_policy": {"fallback": policy.fallback, "max_age_days": policy.max_age_days}}
    return mu, cov, list(mu.index)
