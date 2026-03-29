
import streamlit as st
import pandas as pd
import numpy as np
from arch import arch_model

from app.garch_lab_storage import save_garch_run


def forecast_gjr(series):
    model = arch_model(series * 100, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="t")
    res = model.fit(disp="off")
    f = res.forecast(horizon=1)
    return float(f.variance.values[-1][0]) / 10000


def evaluate(series):
    s = series.dropna()
    preds = []
    realized = []

    for i in range(300, len(s) - 1):
        train = s.iloc[:i]
        try:
            pred = forecast_gjr(train)
        except:
            continue

        preds.append(pred)
        realized.append(s.iloc[i + 1] ** 2)

    df = pd.DataFrame({
        "pred": preds,
        "real": realized
    })

    rmse = np.sqrt(((df.pred - df.real) ** 2).mean())
    mae = np.abs(df.pred - df.real).mean()
    corr = df.pred.corr(df.real)

    return rmse, mae, corr


st.title("GARCH Lab (GJR-GARCH)")

uploaded = st.file_uploader("Upload returns CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded, index_col=0, parse_dates=True)

    results = []

    for col in df.columns:
        st.write(f"Processing {col}...")
        try:
            rmse, mae, corr = evaluate(df[col])
            results.append({
                "ticker": col,
                "rmse": rmse,
                "mae": mae,
                "corr": corr
            })
        except:
            continue

    res_df = pd.DataFrame(results)
    st.dataframe(res_df)

    if st.button("Save Run"):
        run_id = save_garch_run(
            res_df,
            {"model": "GJR-GARCH"}
        )
        st.success(f"Saved run {run_id}")
