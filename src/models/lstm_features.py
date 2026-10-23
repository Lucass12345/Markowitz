from __future__ import annotations

import pandas as pd

AVAILABLE_FEATURES = [
    "return",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_5",
    "rolling_std_5",
    "rolling_mean_20",
    "rolling_std_20",
    "momentum_5",
    "momentum_20",
    "log_return_sum_5",
    "log_return_sum_20",
    "downside_vol_20",
    "ewma_vol_20",
    "rolling_skew_20",
]


def build_lstm_feature_frame(series: pd.Series) -> pd.DataFrame:
    s = series.dropna().astype(float).copy()

    df = pd.DataFrame(index=s.index)
    df["return"] = s
    df["lag_1"] = s.shift(1)
    df["lag_2"] = s.shift(2)
    df["lag_3"] = s.shift(3)
    df["rolling_mean_5"] = s.rolling(5).mean()
    df["rolling_std_5"] = s.rolling(5).std()
    df["rolling_mean_20"] = s.rolling(20).mean()
    df["rolling_std_20"] = s.rolling(20).std()
    df["momentum_5"] = s - s.shift(5)
    df["momentum_20"] = s - s.shift(20)

    df["log_return_sum_5"] = s.rolling(5).sum()
    df["log_return_sum_20"] = s.rolling(20).sum()
    df["downside_vol_20"] = s.clip(upper=0).pow(2).rolling(20).mean().pow(.5)
    df["ewma_vol_20"] = s.ewm(span=20, adjust=False).std()
    df["rolling_skew_20"] = s.rolling(20).skew().fillna(0.0)

    return df


def select_feature_columns(df: pd.DataFrame, selected_features: list[str]) -> pd.DataFrame:
    cols = [c for c in selected_features if c in df.columns]
    if not cols:
        raise ValueError("No valid LSTM features selected.")
    return df[cols].dropna()
