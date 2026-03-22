from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.load_data import load_returns
try:
    from src.data.load_data import load_prices
except Exception:
    load_prices = None

from src.models.train_returns_lstm_lab import run_lstm_lab
from app.lstm_lab_storage import get_prediction_runs_df, set_active_prediction_run


st.set_page_config(page_title="LSTM Lab", page_icon="🧠", layout="wide")
st.title("LSTM Lab")
st.caption("Full-universe configurable LSTM training, prediction, ranking, filtering and ticker-level inspection.")

RETURNS = load_returns()
ALL_TICKERS = list(RETURNS.columns)


def get_prices():
    global load_prices
    if load_prices is not None:
        try:
            p = load_prices()
            if isinstance(p, pd.DataFrame) and not p.empty:
                return p
        except Exception:
            pass

    synthetic = (1 + RETURNS.fillna(0)).cumprod() * 100
    return synthetic


PRICES = get_prices()


def load_latest_saved_result():
    latest_path = ROOT / "outputs" / "lstm_lab" / "latest_run.json"
    if not latest_path.exists():
        return None

    try:
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        run_output_dir = Path(latest["run_output_dir"])
        predictions_path = run_output_dir / "predictions.csv"
        metrics_path = run_output_dir / "metrics.csv"
        summary_path = run_output_dir / "summary.json"
        config_path = run_output_dir / "config.json"

        if not predictions_path.exists() or not metrics_path.exists():
            return None

        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}

        return {
            "run_id": latest["run_id"],
            "predictions_path": str(predictions_path),
            "metrics_path": str(metrics_path),
            "per_ticker_dir": str(run_output_dir / "per_ticker"),
            "curves_dir": str(run_output_dir / "loss_curves"),
            "predictions": pd.read_csv(predictions_path),
            "metrics": pd.read_csv(metrics_path),
            "summary": {**config, **summary},
        }
    except Exception:
        return None


def build_selected_features():
    selected = []

    with st.sidebar:
        st.header("Configuration")

        st.subheader("Feature selection")

        with st.expander("Base features", expanded=True):
            f_return = st.checkbox("return", value=True)
            f_lag1 = st.checkbox("lag_1", value=True)
            f_lag2 = st.checkbox("lag_2", value=True)
            f_lag3 = st.checkbox("lag_3", value=False)

        with st.expander("Rolling features", expanded=True):
            f_rm5 = st.checkbox("rolling_mean_5", value=True)
            f_rs5 = st.checkbox("rolling_std_5", value=True)
            f_rm20 = st.checkbox("rolling_mean_20", value=True)
            f_rs20 = st.checkbox("rolling_std_20", value=True)

        with st.expander("Momentum features", expanded=False):
            f_m5 = st.checkbox("momentum_5", value=False)
            f_m20 = st.checkbox("momentum_20", value=False)

        if f_return: selected.append("return")
        if f_lag1: selected.append("lag_1")
        if f_lag2: selected.append("lag_2")
        if f_lag3: selected.append("lag_3")
        if f_rm5: selected.append("rolling_mean_5")
        if f_rs5: selected.append("rolling_std_5")
        if f_rm20: selected.append("rolling_mean_20")
        if f_rs20: selected.append("rolling_std_20")
        if f_m5: selected.append("momentum_5")
        if f_m20: selected.append("momentum_20")

        start_date = st.selectbox(
            "Training data start date",
            options=["full", "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"],
            index=3,
        )

        seq_len = st.slider("Lookback window (trading days)", 20, 60, 40)
        forecast_horizon = st.slider("Forecast horizon (trading days)", 1, 10, 5)
        target_aggregation = st.radio("Target aggregation", options=["mean", "sum"], horizontal=True)

        hidden_size = st.slider("Hidden size", 16, 128, 64, step=16)
        num_layers = st.slider("Num layers", 1, 4, 2)
        dropout = st.slider("Dropout", 0.0, 0.5, 0.2, step=0.05)
        epochs = st.slider("Epochs", 10, 150, 50, step=10)
        batch_size = st.selectbox("Batch size", [16, 32, 64], index=1)
        lr = st.selectbox("Learning rate", [1e-4, 5e-4, 1e-3, 5e-3], index=2)

    return {
        "selected_features": selected,
        "start_date": start_date,
        "seq_len": seq_len,
        "forecast_horizon": forecast_horizon,
        "target_aggregation": target_aggregation,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
    }


cfg = build_selected_features()

st.info(
    f"""**Universe:** all available tickers in returns dataset (`{len(ALL_TICKERS)}` assets)  
**Target:** `{cfg["target_aggregation"]}` of next `{cfg["forecast_horizon"]}` trading-day returns  
**Input window:** last `{cfg["seq_len"]}` trading days  
**Training data start:** `{cfg["start_date"]}`  
**Selected features:** {", ".join(cfg["selected_features"]) if cfg["selected_features"] else "None"}"""
)

run_button = st.button("Train and generate full-universe predictions", use_container_width=True)

if run_button:
    if not cfg["selected_features"]:
        st.error("Select at least one feature.")
        st.stop()

    with st.spinner("Training configurable LSTM models for all available tickers..."):
        result = run_lstm_lab(
            selected_features=cfg["selected_features"],
            seq_len=cfg["seq_len"],
            forecast_horizon=cfg["forecast_horizon"],
            target_aggregation=cfg["target_aggregation"],
            start_date=cfg["start_date"],
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        )
    st.session_state["lstm_lab_result"] = result

if "lstm_lab_result" not in st.session_state:
    latest = load_latest_saved_result()
    if latest is not None:
        st.session_state["lstm_lab_result"] = latest

result = st.session_state.get("lstm_lab_result")

runs_df = get_prediction_runs_df()

with st.expander("Stored LSTM prediction runs (Postgres)", expanded=False):
    if runs_df.empty:
        st.info("No prediction runs stored in Postgres yet.")
    else:
        st.dataframe(runs_df, use_container_width=True, hide_index=True)
        active_candidates = runs_df["run_id"].tolist()
        selected_active = st.selectbox("Set active run for optimizer", active_candidates)
        if st.button("Set selected run as active"):
            set_active_prediction_run(selected_active)
            st.success(f"Active prediction run set to: {selected_active}")
            st.rerun()

if result is None:
    st.warning("No LSTM Lab run available yet. Launch one from this page.")
    st.stop()

pred_df = result["predictions"].copy()
metrics_df = result["metrics"].copy()
summary = result.get("summary", {})

if pred_df.empty or metrics_df.empty:
    st.warning("Run completed but no predictions/metrics were generated.")
    st.stop()

merged = metrics_df.merge(pred_df, on=["ticker", "mu_pred"], how="outer")

st.success(f"LSTM run available: {result['run_id']}")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Universe size", int(summary.get("total_tickers_in_universe", len(ALL_TICKERS))))
with c2:
    st.metric("Tickers predicted", int(summary.get("tickers_with_predictions", len(pred_df))))
with c3:
    st.metric("Mean RMSE test", f"{merged['rmse_test'].mean():.6f}" if "rmse_test" in merged.columns else "-")
with c4:
    st.metric("Mean dir. acc. test", f"{merged['dir_acc_test'].mean():.2%}" if "dir_acc_test" in merged.columns else "-")

c5, c6, c7, c8 = st.columns(4)
with c5:
    st.metric("Mean mu_pred", f"{pred_df['mu_pred'].mean():.6f}")
with c6:
    st.metric("Median mu_pred", f"{pred_df['mu_pred'].median():.6f}")
with c7:
    st.metric("Min mu_pred", f"{pred_df['mu_pred'].min():.6f}")
with c8:
    st.metric("Max mu_pred", f"{pred_df['mu_pred'].max():.6f}")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Rankings", "Filter & Table", "Ticker Detail"])

with tab1:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Predicted return distribution")
        fig_hist = px.histogram(pred_df, x="mu_pred", nbins=30)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_right:
        st.subheader("RMSE test distribution")
        if "rmse_test" in merged.columns:
            fig_rmse = px.histogram(merged.dropna(subset=["rmse_test"]), x="rmse_test", nbins=30)
            st.plotly_chart(fig_rmse, use_container_width=True)

    st.subheader("Artifacts")
    st.code(
        f"Run ID: {result['run_id']}\\n"
        f"Predictions: {result['predictions_path']}\\n"
        f"Metrics: {result['metrics_path']}\\n"
        f"Per ticker predictions: {result['per_ticker_dir']}\\n"
        f"Loss curves: {result['curves_dir']}"
    )

with tab2:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Top 5 best model quality")
        quality_df = merged.dropna(subset=["rmse_test"]).sort_values(["rmse_test", "mae_test"], ascending=[True, True]).head(5)
        st.dataframe(
            quality_df[["ticker", "rmse_test", "mae_test", "dir_acc_test", "corr_test", "mu_pred"]],
            use_container_width=True,
            hide_index=True,
        )

        if not quality_df.empty:
            fig_quality = px.bar(quality_df, x="ticker", y="rmse_test", title="Top 5 lowest RMSE test")
            st.plotly_chart(fig_quality, use_container_width=True)

    with c2:
        st.subheader("Top 5 highest predicted returns")
        opp_df = merged.sort_values("mu_pred", ascending=False).head(5)
        st.dataframe(
            opp_df[["ticker", "mu_pred", "rmse_test", "dir_acc_test", "corr_test"]],
            use_container_width=True,
            hide_index=True,
        )

        if not opp_df.empty:
            fig_opp = px.bar(opp_df, x="ticker", y="mu_pred", title="Top 5 highest mu_pred")
            st.plotly_chart(fig_opp, use_container_width=True)

with tab3:
    st.subheader("Global filtered table")

    colf1, colf2, colf3 = st.columns(3)
    with colf1:
        search_ticker = st.text_input("Search ticker", "")
    with colf2:
        min_dir_acc = st.slider("Min directional accuracy test", 0.0, 1.0, 0.0, 0.01)
    with colf3:
        max_rmse = st.number_input(
            "Max RMSE test",
            min_value=0.0,
            value=float(np.nanmax(merged["rmse_test"])) if "rmse_test" in merged.columns and merged["rmse_test"].notna().any() else 1.0
        )

    filtered = merged.copy()

    if search_ticker.strip():
        filtered = filtered[filtered["ticker"].astype(str).str.contains(search_ticker.strip(), case=False, na=False)]

    if "dir_acc_test" in filtered.columns:
        filtered = filtered[(filtered["dir_acc_test"].isna()) | (filtered["dir_acc_test"] >= min_dir_acc)]

    if "rmse_test" in filtered.columns:
        filtered = filtered[(filtered["rmse_test"].isna()) | (filtered["rmse_test"] <= max_rmse)]

    filtered = filtered.sort_values(["mu_pred", "dir_acc_test"], ascending=[False, False])
    st.dataframe(filtered, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Ticker detail")

    ticker_options = list(merged["ticker"].dropna().astype(str).unique())
    selected_ticker = st.selectbox("Select ticker", options=ticker_options)

    ticker_row = merged[merged["ticker"] == selected_ticker].iloc[0]
    detail_cols = st.columns(5)
    with detail_cols[0]:
        st.metric("mu_pred", f"{float(ticker_row['mu_pred']):.6f}")
    with detail_cols[1]:
        st.metric("RMSE test", f"{float(ticker_row['rmse_test']):.6f}" if pd.notna(ticker_row.get("rmse_test")) else "-")
    with detail_cols[2]:
        st.metric("DirAcc test", f"{float(ticker_row['dir_acc_test']):.2%}" if pd.notna(ticker_row.get("dir_acc_test")) else "-")
    with detail_cols[3]:
        st.metric("Corr test", f"{float(ticker_row['corr_test']):.4f}" if pd.notna(ticker_row.get("corr_test")) else "-")
    with detail_cols[4]:
        st.metric("Best val loss", f"{float(ticker_row['best_val_loss']):.6f}" if pd.notna(ticker_row.get("best_val_loss")) else "-")

    sub1, sub2 = st.columns(2)

    with sub1:
        st.markdown("**Metrics by split**")
        split_metrics = pd.DataFrame([
            {"split": "train", "mae": ticker_row.get("mae_train"), "rmse": ticker_row.get("rmse_train"), "dir_acc": ticker_row.get("dir_acc_train"), "corr": ticker_row.get("corr_train")},
            {"split": "val", "mae": ticker_row.get("mae_val"), "rmse": ticker_row.get("rmse_val"), "dir_acc": ticker_row.get("dir_acc_val"), "corr": ticker_row.get("corr_val")},
            {"split": "test", "mae": ticker_row.get("mae_test"), "rmse": ticker_row.get("rmse_test"), "dir_acc": ticker_row.get("dir_acc_test"), "corr": ticker_row.get("corr_test")},
        ])
        st.dataframe(split_metrics, use_container_width=True, hide_index=True)

    with sub2:
        prices_t = PRICES[selected_ticker].dropna().copy()
        last_5 = prices_t.tail(5)

        if not last_5.empty:
            last_date = pd.to_datetime(last_5.index[-1])
            future_date = pd.bdate_range(last_date, periods=int(summary.get("forecast_horizon", 5)) + 1)[-1]
            last_price = float(last_5.iloc[-1])
            mu_pred = float(ticker_row["mu_pred"])
            horizon = int(summary.get("forecast_horizon", 5))
            projected_price = float(last_price * ((1 + mu_pred) ** horizon))

            chart_df = pd.DataFrame({
                "date": list(pd.to_datetime(last_5.index)) + [future_date],
                "price": list(last_5.values) + [projected_price],
                "type": ["historical"] * len(last_5) + ["projected"],
            })

            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                x=chart_df[chart_df["type"] == "historical"]["date"],
                y=chart_df[chart_df["type"] == "historical"]["price"],
                mode="lines+markers",
                name="Historical price"
            ))
            fig_price.add_trace(go.Scatter(
                x=[last_5.index[-1], future_date],
                y=[last_price, projected_price],
                mode="lines+markers",
                name="Projection"
            ))
            fig_price.update_layout(title=f"{selected_ticker}: last 5 prices + projected price")
            st.plotly_chart(fig_price, use_container_width=True)

            st.caption(f"Projected price in {horizon} trading days: {projected_price:.4f}")

    pred_file = Path(result["per_ticker_dir"]) / f"{selected_ticker}_predictions.csv"
    if pred_file.exists():
        pred_detail = pd.read_csv(pred_file)
        pred_detail["date"] = pd.to_datetime(pred_detail["date"])

        st.markdown("**Actual vs predicted returns by split**")
        split_choice = st.multiselect("Splits to show", options=["train", "val", "test"], default=["val", "test"])

        plot_detail = pred_detail[pred_detail["split"].isin(split_choice)].copy()

        if not plot_detail.empty:
            fig_det = go.Figure()
            for split in split_choice:
                sub = plot_detail[plot_detail["split"] == split]
                if not sub.empty:
                    fig_det.add_trace(go.Scatter(x=sub["date"], y=sub["y_true"], mode="lines", name=f"y_true_{split}"))
                    fig_det.add_trace(go.Scatter(x=sub["date"], y=sub["y_pred"], mode="lines", name=f"y_pred_{split}"))
            fig_det.update_layout(title=f"{selected_ticker}: actual vs predicted returns")
            st.plotly_chart(fig_det, use_container_width=True)

            st.dataframe(plot_detail.tail(30), use_container_width=True, hide_index=True)

    loss_file = Path(result["curves_dir"]) / f"{selected_ticker}_losses.csv"
    if loss_file.exists():
        losses = pd.read_csv(loss_file)
        st.markdown("**Loss curves**")
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=losses["epoch"], y=losses["train_loss"], mode="lines", name="train_loss"))
        fig_loss.add_trace(go.Scatter(x=losses["epoch"], y=losses["val_loss"], mode="lines", name="val_loss"))
        fig_loss.update_layout(title=f"{selected_ticker}: train vs val loss")
        st.plotly_chart(fig_loss, use_container_width=True)
