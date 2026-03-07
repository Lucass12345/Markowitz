
import pandas as pd

def build_features_for_ticker(series, lags=20):

    s = series.dropna()

    df = pd.DataFrame(index=s.index)
    df["ret"] = s

    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = s.shift(lag)

    df["roll_mean_5"] = s.shift(1).rolling(5).mean()
    df["roll_mean_20"] = s.shift(1).rolling(20).mean()

    df["roll_std_5"] = s.shift(1).rolling(5).std()
    df["roll_std_20"] = s.shift(1).rolling(20).std()

    df["target"] = s.shift(-1)

    return df.dropna()
