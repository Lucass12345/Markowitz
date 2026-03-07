
import pandas as pd


def ewma_variance(series, lam=0.94):

    s = series.dropna()

    var = pd.Series(index=s.index)

    var.iloc[0] = s.var()

    for i in range(1,len(s)):
        var.iloc[i] = lam*var.iloc[i-1] + (1-lam)*(s.iloc[i-1]**2)

    return var


def next_day_variance_forecast(series):

    v = ewma_variance(series)

    return float(v.iloc[-1])
