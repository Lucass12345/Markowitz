import numpy as np
import pandas as pd


def build_lstm_sequences(
    series: pd.Series,
    seq_len: int = 20,
    forecast_horizon: int = 5
):
    """
    Input:
        ventana de retornos pasados de largo seq_len

    Target:
        retorno acumulado futuro de los próximos forecast_horizon días
    """
    s = series.dropna().astype(float).values

    X, y = [], []

    for i in range(seq_len, len(s) - forecast_horizon + 1):
        x_window = s[i - seq_len:i]
        y_future = s[i:i + forecast_horizon].mean()

        X.append(x_window.reshape(-1, 1))
        y.append(y_future)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y


# ---------------------------
# NUEVA FUNCIÓN MULTIFEATURE
# ---------------------------

def build_lstm_sequences_multifeature(
    feature_df: pd.DataFrame,
    target_series: pd.Series,
    seq_len: int = 20,
    forecast_horizon: int = 5,
):
    """
    Construye secuencias LSTM cuando tenemos múltiples features.

    feature_df:
        DataFrame con columnas de features

    target_series:
        Serie con el retorno que queremos predecir
    """

    X, y = [], []

    df = feature_df.copy()
    target = target_series.copy().astype(float)

    aligned = df.copy()
    aligned["target"] = target
    aligned = aligned.dropna()

    values = aligned.drop(columns=["target"]).values.astype(np.float32)
    t = aligned["target"].values.astype(np.float32)

    for i in range(seq_len, len(aligned) - forecast_horizon + 1):
        x_window = values[i - seq_len:i]
        y_future = t[i:i + forecast_horizon].mean()

        X.append(x_window)
        y.append(y_future)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y

