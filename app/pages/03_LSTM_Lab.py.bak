from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.load_data import load_returns
from src.models.lstm_features import AVAILABLE_FEATURES
from src.models.train_returns_lstm_lab import run_lstm_lab

st.set_page_config(page_title="LSTM Lab", page_icon="🧠", layout="wide")

st.title("LSTM Lab")
st.caption("Train configurable LSTM return models and persist predictions cleanly.")

returns = load_returns()
all_tickers = list(returns.columns)

with st.sidebar:
    st.header("Configuration")

    selected_features = st.multiselect(
        "Features",
        options=AVAILABLE_FEATURES,
        default=["return", "lag_1", "lag_2", "rolling_mean_5", "rolling_std_5"],
    )

    selected_tickers = st.multiselect(
        "Tickers",
        options=all_tickers,
        default=all_tickers,
    )

    seq_len = st.slider("Sequence length", 5, 60, 20)
    forecast_horizon = st.slider("Forecast horizon", 1, 10, 5)
    hidden_size = st.slider("Hidden size", 16, 128, 64, step=16)
    num_layers = st.slider("Num layers", 1, 4, 2)
    dropout = st.slider("Dropout", 0.0, 0.5, 0.2, step=0.05)
    epochs = st.slider("Epochs", 10, 150, 60, step=10)
    batch_size = st.selectbox("Batch size", [16, 32, 64], index=1)
    lr = st.selectbox("Learning rate", [1e-4, 5e-4, 1e-3, 5e-3], index=2)

run_button = st.button("Train and generate predictions", use_container_width=True)

if run_button:
    if not selected_features:
        st.error("Select at least one feature.")
        st.stop()

    with st.spinner("Training configurable LSTM models..."):
        result = run_lstm_lab(
            selected_features=selected_features,
            seq_len=seq_len,
            forecast_horizon=forecast_horizon,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            tickers=selected_tickers,
        )

    st.success(f"LSTM run completed: {result['run_id']}")

    pred_df = result["predictions"]
    metrics_df = result["metrics"]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Run ID", result["run_id"])
    with c2:
        st.metric("Assets predicted", len(pred_df))
    with c3:
        st.metric("Mean RMSE", f"{metrics_df['rmse_test'].mean():.6f}" if not metrics_df.empty else "-")

    st.subheader("Predictions")
    st.dataframe(pred_df, use_container_width=True, hide_index=True)

    if not pred_df.empty:
        fig = px.histogram(pred_df, x="mu_pred", nbins=30, title="Distribution of LSTM predictions")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Metrics by ticker")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.subheader("Saved artifacts")
    st.code(
        f"Predictions: {result['predictions_path']}\n"
        f"Metrics: {result['metrics_path']}\n"
        f"Models: {result['model_dir']}"
    )
