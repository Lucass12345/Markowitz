from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.data.load_data import assert_current_dataset
from src.optimization.lstm_lab_loader import load_active_lstm_predictions_df
from src.optimization.mlp_lab_loader import load_active_mlp_predictions

ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = ROOT / "outputs" / "predictions"
LSTM_DIR_ACC_MIN = 0.52
LSTM_AMP_RATIO_MIN = 0.15


def _read_prediction_csv(filename: str) -> pd.DataFrame:
    path = PRED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Generalo primero desde el pipeline correspondiente.")
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        raise ValueError(f"{filename} debe contener la columna 'ticker'.")
    assert_current_dataset(df)
    df["ticker"] = df["ticker"].astype(str).str.strip()
    return df


def load_predictions(prediction_file: str = "predictions.csv") -> pd.DataFrame:
    return _read_prediction_csv(prediction_file)


def load_volatility_predictions(prediction_file: str = "volatility_predictions.csv") -> pd.DataFrame:
    preds = _read_prediction_csv(prediction_file)
    required_cols = {"ticker", "vol_pred"}
    missing = required_cols - set(preds.columns)
    if missing:
        raise ValueError(f"{prediction_file} no tiene columnas requeridas: {sorted(missing)}")
    assert_current_dataset(preds)
    preds["vol_pred"] = preds["vol_pred"].astype(float)
    preds = preds.dropna(subset=["vol_pred"]).copy()
    preds = preds.drop_duplicates(subset=["ticker"], keep="last").copy()
    return preds


def _build_ai_vol_cov(returns_subset: pd.DataFrame, pred_vol: pd.Series) -> pd.DataFrame:
    hist_cov = returns_subset.cov()
    hist_vol = returns_subset.std().replace(0, np.nan)
    corr = hist_cov.div(hist_vol, axis=0).div(hist_vol, axis=1)
    corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d = pd.Series(pred_vol, index=returns_subset.columns, dtype=float)
    cov_ai = corr.mul(d, axis=0).mul(d, axis=1)
    return (cov_ai + cov_ai.T) / 2.0


def _build_ai_returns_from_active_mlp(returns: pd.DataFrame):
    preds = load_active_mlp_predictions().copy()
    if preds.empty:
        raise ValueError("No hay un run activo de MLP Lab para usar en ai_returns.")
    preds["ticker"] = preds["ticker"].astype(str).str.strip()
    preds = preds.drop_duplicates(subset=["ticker"], keep="last").set_index("ticker")
    if "mu_pred" not in preds.columns:
        raise ValueError("Active MLP Lab predictions must contain column 'mu_pred'.")
    common = [t for t in returns.columns if t in preds.index]
    if not common:
        raise ValueError("No hay tickers en común entre returns.csv y el run activo de MLP Lab.")
    returns_sub = returns[common].copy()
    preds = preds.loc[common].copy()
    hist_mu = returns_sub.mean()
    mu = hist_mu.copy()
    use_mask = preds["use_for_optimizer"].astype(bool) if "use_for_optimizer" in preds.columns else pd.Series(True, index=preds.index)
    mu.loc[use_mask.index[use_mask]] = preds.loc[use_mask, "mu_pred"].astype(float)
    cov = returns_sub.cov()
    return mu.loc[common], cov.loc[common, common], common


def build_model_inputs(model_type: str, returns: pd.DataFrame):
    model_type = str(model_type).strip().lower()
    returns = returns.copy()
    returns.columns = returns.columns.astype(str).str.strip()

    if model_type == "classic":
        mu = returns.mean()
        cov = returns.cov()
        return mu, cov, list(mu.index)

    if model_type == "ai_returns":
        try:
            return _build_ai_returns_from_active_mlp(returns)
        except Exception:
            preds = load_predictions("predictions.csv").set_index("ticker")
            if "mu_pred" not in preds.columns:
                raise ValueError("predictions.csv debe contener columna 'mu_pred'")
            common = [t for t in returns.columns if t in preds.index]
            if not common:
                raise ValueError("No hay tickers en común entre returns.csv y predictions.csv")
            returns_sub = returns[common].copy()
            mu = preds.loc[common, "mu_pred"].astype(float)
            cov = returns_sub.cov()
            return mu, cov, common

    if model_type == "ai_returns_lstm":
        preds = load_active_lstm_predictions_df().copy()
        assert_current_dataset(preds)
        preds["ticker"] = preds["ticker"].astype(str).str.strip()
        preds = preds.set_index("ticker")
        if "mu_pred" not in preds.columns:
            raise ValueError("Active LSTM Lab predictions must contain column 'mu_pred'")
        if "dir_acc_val" not in preds.columns:
            raise ValueError("Active LSTM Lab predictions must contain column 'dir_acc_val'")
        if "amp_ratio" not in preds.columns:
            raise ValueError("Active LSTM Lab predictions must contain column 'amp_ratio'")
        common = [t for t in returns.columns if t in preds.index]
        if not common:
            raise ValueError("No active LSTM Lab predictions available for optimizer universe.")
        returns_sub = returns[common].copy()
        preds = preds.loc[common].copy()
        hist_mu = returns_sub.mean()
        mu = hist_mu.copy()
        good_mask = (
            (preds["dir_acc_val"].astype(float) > LSTM_DIR_ACC_MIN)
            & (preds["amp_ratio"].astype(float) > LSTM_AMP_RATIO_MIN)
        )
        if "use_for_optimizer" in preds:
            good_mask &= preds["use_for_optimizer"].fillna(False).astype(str).str.lower().eq("true")
        mu.loc[good_mask.index[good_mask]] = preds.loc[good_mask, "mu_pred"].astype(float)
        cov = returns_sub.cov()
        return mu.loc[common], cov.loc[common, common], common

    if model_type == "ai_volatility":
        preds = load_volatility_predictions("volatility_predictions.csv").set_index("ticker")
        common = [t for t in returns.columns if t in preds.index]
        if not common:
            raise ValueError("No hay tickers en común entre returns.csv y volatility_predictions.csv")
        returns_sub = returns[common].copy()
        mu = returns_sub.mean()
        pred_vol = preds.loc[common, "vol_pred"].astype(float)
        cov = _build_ai_vol_cov(returns_sub, pred_vol)
        return mu.loc[common], cov.loc[common, common], common

    raise ValueError(f"model_type inválido: {model_type}. Usá 'classic', 'ai_returns', 'ai_returns_lstm' o 'ai_volatility'.")
