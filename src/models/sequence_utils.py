import numpy as np
import pandas as pd


def build_lstm_sequences(
    series: pd.Series,
    seq_len: int = 20,
    forecast_horizon: int = 5
):
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


def build_lstm_sequences_multifeature(
    feature_df: pd.DataFrame,
    target_series: pd.Series,
    seq_len: int = 20,
    forecast_horizon: int = 5,
    target_aggregation: str = "mean",
):
    if forecast_horizon < 1 or seq_len < 1 or target_aggregation not in {"mean", "sum"}:
        raise ValueError("Invalid sequence length, horizon or target aggregation")
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
        future_slice = t[i:i + forecast_horizon]

        if target_aggregation == "sum":
            y_future = future_slice.sum()
        else:
            y_future = future_slice.mean()

        X.append(x_window)
        y.append(y_future)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y
