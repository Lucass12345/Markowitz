from src.optimization.lstm_lab_loader import load_active_lstm_predictions_df
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = ROOT / "outputs" / "predictions"


def load_predictions(prediction_file: str = "predictions.csv") -> pd.DataFrame:
    path = PRED_DIR / prediction_file
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}."
        )

    preds = pd.read_csv(path)
    preds["ticker"] = preds["ticker"].astype(str).str.strip()
    return preds


def _build_ai_vol_cov(returns_subset: pd.DataFrame, pred_vol: pd.Series) -> pd.DataFrame:
    """
    Reescala la cov histórica usando las volatilidades predichas,
    manteniendo la estructura de correlaciones histórica.

    cov_ai = D * Corr_hist * D
    donde D = diag(vol_pred)
    """
    hist_cov = returns_subset.cov()
    hist_vol = returns_subset.std()

    hist_vol = hist_vol.replace(0, np.nan)

    corr = hist_cov.div(hist_vol, axis=0).div(hist_vol, axis=1)
    corr = corr.fillna(0.0)

    d = pd.Series(pred_vol, index=returns_subset.columns, dtype=float)
    cov_ai = corr.mul(d, axis=0).mul(d, axis=1)

    cov_ai = (cov_ai + cov_ai.T) / 2.0
    return cov_ai


def build_model_inputs(
    model_type: str,
    returns: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame, list[str]]:
    """
    Devuelve:
    - mu
    - cov
    - tickers alineados

    model_type:
    - classic
    - ai_returns
    - ai_returns_lstm
    - ai_volatility
    """
    model_type = str(model_type).strip().lower()

    returns = returns.copy()
    returns.columns = returns.columns.astype(str).str.strip()

    # =========================
    # CLASSIC
    # =========================
    if model_type == "classic":
        mu = returns.mean()
        cov = returns.cov()
        tickers = list(mu.index)
        return mu, cov, tickers

    # =========================
    # AI RETURNS - MLP
    # =========================
    if model_type == "ai_returns":
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

    # =========================
    # AI RETURNS - LSTM
    # =========================
    if model_type == "ai_returns_lstm":
        preds = load_active_lstm_predictions_df().set_index("ticker")

        if "mu_pred" not in preds.columns:
            raise ValueError("Active LSTM Lab predictions must contain column 'mu_pred'")

        common = [t for t in returns.columns if t in preds.index]
        if not common:
            raise ValueError("No active LSTM Lab predictions available for optimizer universe.")

        returns_sub = returns[common].copy()
        mu = preds.loc[common, "mu_pred"].astype(float)
        cov = returns_sub.cov()

        return mu, cov, common

    # =========================
    # AI VOLATILITY - GARCH
    # =========================
    if model_type == "ai_volatility":
        preds = load_predictions("predictions.csv").set_index("ticker")

        required_cols = {"vol_pred"}
        missing = required_cols - set(preds.columns)
        if missing:
            raise ValueError(f"predictions.csv no tiene columnas requeridas: {missing}")

        common = [t for t in returns.columns if t in preds.index]
        if not common:
            raise ValueError("No hay tickers en común entre returns.csv y predictions.csv")

        returns_sub = returns[common].copy()
        mu = returns_sub.mean()
        pred_vol = preds.loc[common, "vol_pred"].astype(float)
        cov = _build_ai_vol_cov(returns_sub, pred_vol)

        return mu, cov, common

    raise ValueError(
        f"model_type inválido: {model_type}. "
        f"Usá 'classic', 'ai_returns', 'ai_returns_lstm' o 'ai_volatility'."
    )