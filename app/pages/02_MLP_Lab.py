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

from src.models.train_returns_mlp_lab import run_mlp_lab
from app.mlp_lab_storage import get_mlp_prediction_runs_df, set_active_mlp_prediction_run


def get_prices(returns: pd.DataFrame) -> pd.DataFrame:
    global load_prices
    if load_prices is not None:
        try:
            p = load_prices()
            if isinstance(p, pd.DataFrame) and not p.empty:
                return p
        except Exception:
            pass
    return (1 + returns.fillna(0)).cumprod() * 100


def load_latest_saved_result():
    latest_path = ROOT / "outputs" / "mlp_lab" / "latest_run.json"
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
            "run_output_dir": str(run_output_dir),
            "predictions": pd.read_csv(predictions_path),
            "metrics": pd.read_csv(metrics_path),
            "summary": {**config, **summary},
        }
    except Exception:
        return None


def build_config() -> dict:
    with st.sidebar:
        st.header("MLP Lab configuration")
        start_date = st.selectbox("Training data start date", ["full", "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"], index=3)
        forecast_horizon = st.slider("Forecast horizon (trading days)", 1, 10, 5)
        target_aggregation = st.radio("Target aggregation", ["mean", "sum"], horizontal=True)

        st.subheader("Feature engineering")
        lags = st.slider("Number of lags", 5, 60, 20, step=5)
        short_window = st.slider("Short rolling window", 3, 20, 5)
        long_window = st.slider("Long rolling window", 10, 60, 20)
        include_abs_features = st.checkbox("Include absolute-return features", value=True)
        include_square_features = st.checkbox("Include squared-return features", value=True)
        include_sign_features = st.checkbox("Include sign features", value=True)
        winsorize_quantile = st.slider("Winsorize quantile", 0.0, 0.10, 0.01, 0.01)
        clip_pred = st.slider("Clip final mu_pred", 0.005, 0.05, 0.02, 0.005)

        st.subheader("MLP architecture")
        hidden_layers = st.text_input("Hidden layers (comma separated)", value="256,128,64")
        activation = st.selectbox("Activation", ["relu", "tanh", "logistic", "identity"], index=0)
        solver = st.selectbox("Solver", ["adam", "sgd", "lbfgs"], index=0)
        scaler = st.selectbox("Scaler", ["standard", "robust", "minmax", "none"], index=0)
        alpha = st.selectbox("L2 alpha", [1e-5, 1e-4, 1e-3, 1e-2], index=1)
        learning_rate = st.selectbox("Learning-rate schedule", ["constant", "adaptive", "invscaling"], index=1)
        learning_rate_init = st.selectbox("Learning-rate init", [1e-4, 5e-4, 1e-3, 5e-3], index=2)
        max_iter = st.slider("Max iterations", 100, 2000, 500, step=100)
        batch_size = st.selectbox("Batch size", ["auto", 16, 32, 64, 128], index=0)
        early_stopping = st.checkbox("Early stopping", value=True)
        validation_fraction = st.slider("Internal validation fraction", 0.05, 0.30, 0.15, 0.05)
        n_iter_no_change = st.slider("n_iter_no_change", 5, 50, 20, step=5)

    return {
        "start_date": start_date,
        "forecast_horizon": forecast_horizon,
        "target_aggregation": target_aggregation,
        "lags": lags,
        "short_window": short_window,
        "long_window": long_window,
        "include_abs_features": include_abs_features,
        "include_square_features": include_square_features,
        "include_sign_features": include_sign_features,
        "winsorize_quantile": winsorize_quantile,
        "clip_pred": clip_pred,
        "hidden_layers": hidden_layers,
        "activation": activation,
        "solver": solver,
        "scaler": scaler,
        "alpha": alpha,
        "learning_rate": learning_rate,
        "learning_rate_init": learning_rate_init,
        "max_iter": max_iter,
        "batch_size": batch_size,
        "early_stopping": early_stopping,
        "validation_fraction": validation_fraction,
        "n_iter_no_change": n_iter_no_change,
    }


def render_overview(pred_df: pd.DataFrame, merged: pd.DataFrame, result: dict):
    left, right = st.columns(2)
    with left:
        st.subheader("Predicted return distribution")
        st.plotly_chart(px.histogram(pred_df, x="mu_pred", nbins=30), use_container_width=True)
    with right:
        st.subheader("Composite quality score distribution")
        st.plotly_chart(px.histogram(merged, x="score", nbins=30), use_container_width=True)

    left2, right2 = st.columns(2)
    with left2:
        st.subheader("RMSE test distribution")
        st.plotly_chart(px.histogram(merged.dropna(subset=["rmse_test"]), x="rmse_test", nbins=30), use_container_width=True)
    with right2:
        st.subheader("Directional accuracy test distribution")
        st.plotly_chart(px.histogram(merged.dropna(subset=["dir_acc_test"]), x="dir_acc_test", nbins=30), use_container_width=True)

    st.code(
        f"Run ID: {result['run_id']}\n"
        f"Predictions: {result['predictions_path']}\n"
        f"Metrics: {result['metrics_path']}\n"
        f"Per-ticker detail: {result['per_ticker_dir']}"
    )


def render_rankings(merged: pd.DataFrame):
    left, right = st.columns(2)
    with left:
        st.subheader("Top 10 composite score")
        top_score = merged.sort_values("score", ascending=False).head(10)
        st.dataframe(top_score[["ticker", "score", "mu_pred", "dir_acc_test", "corr_test", "rmse_test"]], use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(top_score, x="ticker", y="score", title="Top 10 by composite score"), use_container_width=True)
    with right:
        st.subheader("Top 10 predicted returns")
        top_mu = merged.sort_values("mu_pred", ascending=False).head(10)
        st.dataframe(top_mu[["ticker", "mu_pred", "score", "dir_acc_test", "corr_test", "rmse_test"]], use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(top_mu, x="ticker", y="mu_pred", title="Top 10 by mu_pred"), use_container_width=True)


def render_filter_table(merged: pd.DataFrame):
    st.subheader("Global filter table")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        search_ticker = st.text_input("Search ticker", "")
    with c2:
        min_score = st.slider("Min score", 0.0, 1.0, 0.0, 0.01)
    with c3:
        min_dir_acc = st.slider("Min dir. acc. test", 0.0, 1.0, 0.0, 0.01)
    with c4:
        max_rmse = st.number_input("Max RMSE test", min_value=0.0, value=float(np.nanmax(merged["rmse_test"])) if merged["rmse_test"].notna().any() else 1.0)

    filtered = merged.copy()
    if search_ticker.strip():
        filtered = filtered[filtered["ticker"].astype(str).str.contains(search_ticker.strip(), case=False, na=False)]
    filtered = filtered[(filtered["score"].isna()) | (filtered["score"] >= min_score)]
    filtered = filtered[(filtered["dir_acc_test"].isna()) | (filtered["dir_acc_test"] >= min_dir_acc)]
    filtered = filtered[(filtered["rmse_test"].isna()) | (filtered["rmse_test"] <= max_rmse)]
    filtered = filtered.sort_values(["score", "mu_pred"], ascending=[False, False])
    st.dataframe(filtered, use_container_width=True, hide_index=True)


def render_ticker_detail(merged: pd.DataFrame, result: dict, summary: dict, prices: pd.DataFrame):
    st.subheader("Ticker detail")
    ticker_options = list(merged["ticker"].dropna().astype(str).unique())
    if not ticker_options:
        st.info("No ticker details available.")
        return
    selected_ticker = st.selectbox("Select ticker", options=ticker_options)
    row = merged[merged["ticker"] == selected_ticker].iloc[0]

    metric_cols = st.columns(6)
    metric_cols[0].metric("mu_pred", f"{float(row['mu_pred']):.6f}")
    metric_cols[1].metric("Score", f"{float(row['score']):.3f}")
    metric_cols[2].metric("RMSE test", f"{float(row['rmse_test']):.6f}" if pd.notna(row.get("rmse_test")) else "-")
    metric_cols[3].metric("DirAcc test", f"{float(row['dir_acc_test']):.2%}" if pd.notna(row.get("dir_acc_test")) else "-")
    metric_cols[4].metric("Corr test", f"{float(row['corr_test']):.4f}" if pd.notna(row.get("corr_test")) else "-")
    metric_cols[5].metric("R2 test", f"{float(row['r2_test']):.4f}" if pd.notna(row.get("r2_test")) else "-")

    left, right = st.columns(2)
    with left:
        split_metrics = pd.DataFrame([
            {"split": "train", "mae": row.get("mae_train"), "rmse": row.get("rmse_train"), "dir_acc": row.get("dir_acc_train"), "corr": row.get("corr_train"), "r2": row.get("r2_train")},
            {"split": "val", "mae": row.get("mae_val"), "rmse": row.get("rmse_val"), "dir_acc": row.get("dir_acc_val"), "corr": row.get("corr_val"), "r2": row.get("r2_val")},
            {"split": "test", "mae": row.get("mae_test"), "rmse": row.get("rmse_test"), "dir_acc": row.get("dir_acc_test"), "corr": row.get("corr_test"), "r2": row.get("r2_test")},
        ])
        st.dataframe(split_metrics, use_container_width=True, hide_index=True)
    with right:
        if selected_ticker in prices.columns:
            prices_t = prices[selected_ticker].dropna().copy()
            last_5 = prices_t.tail(5)
            if not last_5.empty:
                last_date = pd.to_datetime(last_5.index[-1])
                horizon = int(summary.get("forecast_horizon", 5))
                future_date = pd.bdate_range(last_date, periods=horizon + 1)[-1]
                last_price = float(last_5.iloc[-1])
                mu_pred = float(row["mu_pred"])
                projected_price = float(last_price * ((1 + mu_pred) ** horizon))
                fig_price = go.Figure()
                fig_price.add_trace(go.Scatter(x=last_5.index, y=last_5.values, mode="lines+markers", name="Historical price"))
                fig_price.add_trace(go.Scatter(x=[last_5.index[-1], future_date], y=[last_price, projected_price], mode="lines+markers", name="Projection"))
                fig_price.update_layout(title=f"{selected_ticker}: last 5 prices + projected price")
                st.plotly_chart(fig_price, use_container_width=True)
                st.caption(f"Projected price in {horizon} trading days: {projected_price:.4f}")

    pred_file = Path(result["per_ticker_dir"]) / f"{selected_ticker}_predictions.csv"
    if pred_file.exists():
        pred_detail = pd.read_csv(pred_file)
        pred_detail["date"] = pd.to_datetime(pred_detail["date"])
        st.markdown("**Actual vs predicted returns by split**")
        split_choice = st.multiselect("Splits to show", ["train", "val", "test"], default=["val", "test"])
        plot_df = pred_detail[pred_detail["split"].isin(split_choice)].copy()
        if not plot_df.empty:
            fig = go.Figure()
            for split in split_choice:
                sub = plot_df[plot_df["split"] == split]
                if not sub.empty:
                    fig.add_trace(go.Scatter(x=sub["date"], y=sub["y_true"], mode="lines", name=f"y_true_{split}"))
                    fig.add_trace(go.Scatter(x=sub["date"], y=sub["y_pred"], mode="lines", name=f"y_pred_{split}"))
            fig.update_layout(title=f"{selected_ticker}: actual vs predicted returns")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(plot_df.tail(40), use_container_width=True, hide_index=True)


def build_selection_policy_ui(runs_df: pd.DataFrame) -> dict:
    policy_type = st.selectbox(
        "Optimizer selection policy",
        ["all_assets", "dir_acc_min", "corr_and_rmse", "composite_min_score", "composite_top_n", "hybrid_quality"],
        help=(
            "all_assets: usa mu_pred para todos.\n"
            "dir_acc_min: solo activos con directional accuracy mínima.\n"
            "corr_and_rmse: requiere correlación mínima y RMSE máxima.\n"
            "composite_min_score: filtra por score compuesto.\n"
            "composite_top_n: usa solo los top N por score.\n"
            "hybrid_quality: combina dir_acc, corr y score."
        ),
    )
    if policy_type == "all_assets":
        return {"policy_type": policy_type}
    if policy_type == "dir_acc_min":
        return {"policy_type": policy_type, "dir_acc_min": st.slider("dir_acc_min", 0.0, 1.0, 0.52, 0.01)}
    if policy_type == "corr_and_rmse":
        rmse_default = float(runs_df["mean_rmse_test"].dropna().median()) if not runs_df.empty and runs_df["mean_rmse_test"].notna().any() else 0.02
        return {"policy_type": policy_type, "corr_min": st.slider("corr_min", -1.0, 1.0, 0.0, 0.01), "rmse_max": st.number_input("rmse_max", min_value=0.0, value=rmse_default)}
    if policy_type == "composite_min_score":
        return {"policy_type": policy_type, "min_score": st.slider("min_score", 0.0, 1.0, 0.60, 0.01)}
    if policy_type == "composite_top_n":
        return {"policy_type": policy_type, "top_n": int(st.number_input("top_n", min_value=1, value=10, step=1))}
    return {
        "policy_type": policy_type,
        "dir_acc_min": st.slider("hybrid dir_acc_min", 0.0, 1.0, 0.52, 0.01),
        "corr_min": st.slider("hybrid corr_min", -1.0, 1.0, 0.0, 0.01),
        "min_score": st.slider("hybrid min_score", 0.0, 1.0, 0.55, 0.01),
    }


def main():
    st.set_page_config(page_title="MLP Lab", page_icon="🧠", layout="wide")
    st.title("MLP Lab")
    st.caption(
        "Esta página usa directamente Data/returns.csv, guarda historial de runs, permite elegir un run activo para Markowitz y muestra detalle por activo. "
        "En ai_returns, Markowitz consumirá el run activo y aplicará la selection policy elegida."
    )

    returns = load_returns()
    prices = get_prices(returns)
    cfg = build_config()

    st.info(
        f"""**Universe:** `{len(returns.columns)}` assets  
**Target:** `{cfg['target_aggregation']}` of next `{cfg['forecast_horizon']}` trading-day returns  
**Feature set:** lags=`{cfg['lags']}`, short_window=`{cfg['short_window']}`, long_window=`{cfg['long_window']}`  
**Network:** hidden_layers=`{cfg['hidden_layers']}`, activation=`{cfg['activation']}`, solver=`{cfg['solver']}`, scaler=`{cfg['scaler']}`"""
    )

    if st.button("Train and generate full-universe MLP predictions", use_container_width=True):
        with st.spinner("Training configurable MLP models for all available tickers..."):
            result = run_mlp_lab(**cfg)
        st.session_state["mlp_lab_result"] = result

    if "mlp_lab_result" not in st.session_state:
        latest = load_latest_saved_result()
        if latest is not None:
            st.session_state["mlp_lab_result"] = latest

    result = st.session_state.get("mlp_lab_result")
    runs_df = get_mlp_prediction_runs_df()

    with st.expander("Stored MLP prediction runs", expanded=False):
        if runs_df.empty:
            st.info("No MLP runs stored yet.")
        else:
            st.dataframe(runs_df, use_container_width=True, hide_index=True)
            selected_run = st.selectbox("Set active run for optimizer", runs_df["run_id"].tolist())
            selection_policy = build_selection_policy_ui(runs_df)
            if st.button("Set selected MLP run as active"):
                set_active_mlp_prediction_run(selected_run, selection_policy)
                st.success(f"Active MLP run set to: {selected_run}")
                st.rerun()

    if result is None:
        st.warning("No MLP Lab run available yet. Launch one from this page.")
        return

    pred_df = result["predictions"].copy()
    metrics_df = result["metrics"].copy()
    summary = result.get("summary", {})
    merged = metrics_df.copy()

    st.success(f"MLP run available: {result['run_id']}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Universe size", int(summary.get("total_tickers_in_universe", len(returns.columns))))
    c2.metric("Tickers predicted", int(summary.get("tickers_with_predictions", len(pred_df))))
    c3.metric("Mean RMSE test", f"{merged['rmse_test'].mean():.6f}")
    c4.metric("Mean dir. acc. test", f"{merged['dir_acc_test'].mean():.2%}")
    c5.metric("Mean score", f"{merged['score'].mean():.3f}")

    t1, t2, t3, t4 = st.tabs(["Overview", "Rankings", "Filter & Table", "Ticker Detail"])
    with t1:
        render_overview(pred_df, merged, result)
    with t2:
        render_rankings(merged)
    with t3:
        render_filter_table(merged)
    with t4:
        render_ticker_detail(merged, result, summary, prices)


if __name__ == "__main__":
    main()
