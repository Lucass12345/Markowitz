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
        y_future = s[i:i + forecast_horizon].sum()

        X.append(x_window.reshape(-1, 1))
        y.append(y_future)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y
