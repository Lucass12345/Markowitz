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

from src.data.load_data import load_returns, assert_current_dataset
try:
    from src.data.load_data import load_prices
except Exception:
    load_prices = None

from src.models.train_returns_lstm_lab import run_lstm_lab
from app.lstm_lab_storage import get_prediction_runs_df, set_active_prediction_run

DISCLAIMER_TEXT = "Este sistema es únicamente informativo y educativo. No constituye una recomendación de inversión ni asesoramiento financiero."


def get_prices(returns: pd.DataFrame) -> pd.DataFrame:
    global load_prices
    if load_prices is not None:
        try:
            p = load_prices()
            if isinstance(p, pd.DataFrame) and not p.empty:
                return p
        except Exception:
            pass

    synthetic = np.exp(returns.fillna(0).cumsum()) * 100
    return synthetic


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

        predictions = pd.read_csv(predictions_path)
        assert_current_dataset(predictions)
        return {
            "run_id": latest["run_id"],
            "predictions_path": str(predictions_path),
            "metrics_path": str(metrics_path),
            "per_ticker_dir": str(run_output_dir / "per_ticker"),
            "curves_dir": str(run_output_dir / "loss_curves"),
            "predictions": predictions,
            "metrics": pd.read_csv(metrics_path),
            "summary": {**config, **summary},
        }
    except Exception:
        return None


def build_selected_features():
    selected = []

    with st.sidebar:
        st.header("Configuración")
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

        if f_return:
            selected.append("return")
        if f_lag1:
            selected.append("lag_1")
        if f_lag2:
            selected.append("lag_2")
        if f_lag3:
            selected.append("lag_3")
        if f_rm5:
            selected.append("rolling_mean_5")
        if f_rs5:
            selected.append("rolling_std_5")
        if f_rm20:
            selected.append("rolling_mean_20")
        if f_rs20:
            selected.append("rolling_std_20")
        if f_m5:
            selected.append("momentum_5")
        if f_m20:
            selected.append("momentum_20")

        selected.extend(st.multiselect("Features adicionales", ["log_return_sum_5", "log_return_sum_20", "downside_vol_20", "ewma_vol_20", "rolling_skew_20"], default=["log_return_sum_5", "downside_vol_20", "ewma_vol_20"]))

        start_date = st.selectbox(
            "Fecha inicial de entrenamiento",
            options=["full", "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"],
            index=3,
        )

        seq_len = st.slider("Ventana lookback (días hábiles)", 20, 60, 40)
        forecast_horizon = 5
        st.caption("Horizonte fijo: próximas cinco ruedas bursátiles")
        target_aggregation = st.radio("Agregación del target", options=["mean", "sum"], horizontal=True)

        feature_search = st.checkbox("Comparar features seleccionadas contra retornos solos", value=True)
        search = st.checkbox("Comparar arquitecturas, activaciones y Adam", value=True)
        hidden_size = st.slider("Tamaño de capa oculta", 16, 128, 64, step=16, disabled=search)
        num_layers = st.slider("Cantidad de capas", 1, 4, 2, disabled=search)
        dropout = st.slider("Dropout", 0.0, 0.5, 0.2, step=0.05)
        epochs = st.slider("Épocas", 10, 150, 50, step=10)
        batch_size = st.selectbox("Batch size", [16, 32, 64], index=1)
        lr = st.selectbox("Learning rate", [1e-4, 5e-4, 1e-3, 5e-3], index=2, disabled=search)

    return {
        "feature_search": feature_search,
        "search": search,
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


def render_overview(pred_df: pd.DataFrame, merged: pd.DataFrame, result: dict):
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Distribución de retornos predichos")
        fig_hist = px.histogram(pred_df, x="mu_pred", nbins=30)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_right:
        st.subheader("Distribución de RMSE test")
        if "rmse_test" in merged.columns:
            fig_rmse = px.histogram(
                merged.dropna(subset=["rmse_test"]),
                x="rmse_test",
                nbins=30,
            )
            st.plotly_chart(fig_rmse, use_container_width=True)

    st.subheader("Archivos generados")
    st.code(
        f"Run ID: {result['run_id']}\n"
        f"Predicciones: {result['predictions_path']}\n"
        f"Métricas: {result['metrics_path']}\n"
        f"Predicciones por ticker: {result['per_ticker_dir']}\n"
        f"Curvas de pérdida: {result['curves_dir']}"
    )


def render_rankings(merged: pd.DataFrame):
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Top 5 con mejor calidad de modelo")
        quality_df = (
            merged.dropna(subset=["rmse_test"])
            .sort_values(["rmse_test", "mae_test"], ascending=[True, True])
            .head(5)
        )
        if not quality_df.empty:
            st.dataframe(
                quality_df[["ticker", "rmse_test", "mae_test", "dir_acc_test", "corr_test", "mu_pred"]],
                use_container_width=True,
                hide_index=True,
            )
            fig_quality = px.bar(quality_df, x="ticker", y="rmse_test", title="Top 5 menor RMSE test")
            st.plotly_chart(fig_quality, use_container_width=True)
        else:
            st.info("No hay métricas de calidad disponibles.")

    with c2:
        st.subheader("Top 5 mayores retornos predichos")
        opp_df = merged.sort_values("mu_pred", ascending=False).head(5)
        if not opp_df.empty:
            st.dataframe(
                opp_df[["ticker", "mu_pred", "rmse_test", "dir_acc_test", "corr_test"]],
                use_container_width=True,
                hide_index=True,
            )
            fig_opp = px.bar(opp_df, x="ticker", y="mu_pred", title="Top 5 mayor mu_pred")
            st.plotly_chart(fig_opp, use_container_width=True)
        else:
            st.info("No hay predicciones disponibles.")


def render_filter_table(merged: pd.DataFrame):
    st.subheader("Tabla global filtrada")

    colf1, colf2, colf3 = st.columns(3)
    with colf1:
        search_ticker = st.text_input("Buscar ticker", "")
    with colf2:
        min_dir_acc = st.slider("Mínima directional accuracy test", 0.0, 1.0, 0.0, 0.01)
    with colf3:
        rmse_default = (
            float(np.nanmax(merged["rmse_test"]))
            if "rmse_test" in merged.columns and merged["rmse_test"].notna().any()
            else 1.0
        )
        max_rmse = st.number_input("Máximo RMSE test", min_value=0.0, value=rmse_default)

    filtered = merged.copy()

    if search_ticker.strip():
        filtered = filtered[
            filtered["ticker"].astype(str).str.contains(search_ticker.strip(), case=False, na=False)
        ]

    if "dir_acc_test" in filtered.columns:
        filtered = filtered[
            (filtered["dir_acc_test"].isna()) | (filtered["dir_acc_test"] >= min_dir_acc)
        ]

    if "rmse_test" in filtered.columns:
        filtered = filtered[
            (filtered["rmse_test"].isna()) | (filtered["rmse_test"] <= max_rmse)
        ]

    sort_cols = [c for c in ["mu_pred", "dir_acc_test"] if c in filtered.columns]
    if sort_cols:
        filtered = filtered.sort_values(sort_cols, ascending=[False] * len(sort_cols))

    st.dataframe(filtered, use_container_width=True, hide_index=True)


def render_ticker_detail(merged: pd.DataFrame, result: dict, summary: dict, prices: pd.DataFrame):
    st.subheader("Detalle por ticker")

    ticker_options = list(merged["ticker"].dropna().astype(str).unique())
    if not ticker_options:
        st.info("No hay detalle disponible por ticker.")
        return

    selected_ticker = st.selectbox("Seleccionar ticker", options=ticker_options)
    ticker_row = merged[merged["ticker"] == selected_ticker].iloc[0]

    if pd.notna(ticker_row.get("selected_config")):
        with st.expander("Configuración ganadora por validación"):
            st.json(json.loads(ticker_row["selected_config"]))
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
        st.markdown("**Métricas por split**")
        split_metrics = pd.DataFrame([
            {"split": "train", "mae": ticker_row.get("mae_train"), "rmse": ticker_row.get("rmse_train"), "dir_acc": ticker_row.get("dir_acc_train"), "corr": ticker_row.get("corr_train")},
            {"split": "val", "mae": ticker_row.get("mae_val"), "rmse": ticker_row.get("rmse_val"), "dir_acc": ticker_row.get("dir_acc_val"), "corr": ticker_row.get("corr_val")},
            {"split": "test", "mae": ticker_row.get("mae_test"), "rmse": ticker_row.get("rmse_test"), "dir_acc": ticker_row.get("dir_acc_test"), "corr": ticker_row.get("corr_test")},
        ])
        st.dataframe(split_metrics, use_container_width=True, hide_index=True)

    with sub2:
        if selected_ticker in prices.columns:
            prices_t = prices[selected_ticker].dropna().copy()
            last_5 = prices_t.tail(5)

            if not last_5.empty:
                last_date = pd.to_datetime(last_5.index[-1])
                calendar = pd.date_range if (pd.DatetimeIndex(prices_t.tail(30).index).dayofweek >= 5).any() else pd.bdate_range
                future_date = calendar(last_date, periods=int(summary.get("forecast_horizon", 5)) + 1)[-1]
                last_price = float(last_5.iloc[-1])
                mu_pred = float(ticker_row["mu_pred"])
                horizon = int(summary.get("forecast_horizon", 5))
                projected_price = float(last_price * np.exp(mu_pred * horizon))

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
                    name="Precio histórico",
                ))
                fig_price.add_trace(go.Scatter(
                    x=[last_5.index[-1], future_date],
                    y=[last_price, projected_price],
                    mode="lines+markers",
                    name="Proyección",
                ))
                fig_price.update_layout(title=f"{selected_ticker}: últimas 5 cotizaciones + precio proyectado")
                st.plotly_chart(fig_price, use_container_width=True)
                st.caption(f"Precio proyectado en USD a {horizon} observaciones diarias: {projected_price:.4f}. Fecha orientativa; no incorpora feriados del mercado.")

    pred_file = Path(result["per_ticker_dir"]) / f"{selected_ticker}_predictions.csv"
    if pred_file.exists():
        pred_detail = pd.read_csv(pred_file)
        pred_detail["date"] = pd.to_datetime(pred_detail["date"])

        st.markdown("**Retornos reales vs. predichos por split**")
        split_choice = st.multiselect("Splits a mostrar", options=["train", "val", "test"], default=["val", "test"])

        plot_detail = pred_detail[pred_detail["split"].isin(split_choice)].copy()

        if not plot_detail.empty:
            fig_det = go.Figure()
            for split in split_choice:
                sub = plot_detail[plot_detail["split"] == split]
                if not sub.empty:
                    fig_det.add_trace(go.Scatter(x=sub["date"], y=sub["y_true"], mode="lines", name=f"y_true_{split}"))
                    fig_det.add_trace(go.Scatter(x=sub["date"], y=sub["y_pred"], mode="lines", name=f"y_pred_{split}"))
            fig_det.update_layout(title=f"{selected_ticker}: retornos reales vs. predichos")
            st.plotly_chart(fig_det, use_container_width=True)
            st.dataframe(plot_detail.tail(30), use_container_width=True, hide_index=True)

    loss_file = Path(result["curves_dir"]) / f"{selected_ticker}_losses.csv"
    if loss_file.exists():
        losses = pd.read_csv(loss_file)
        st.markdown("**Curvas de pérdida**")
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=losses["epoch"], y=losses["train_loss"], mode="lines", name="train_loss"))
        fig_loss.add_trace(go.Scatter(x=losses["epoch"], y=losses["val_loss"], mode="lines", name="val_loss"))
        fig_loss.update_layout(title=f"{selected_ticker}: pérdida train vs. val")
        st.plotly_chart(fig_loss, use_container_width=True)


def main():
    st.set_page_config(page_title="Laboratorio LSTM", page_icon="🧠", layout="wide")
    st.title("Laboratorio LSTM")
    st.warning(DISCLAIMER_TEXT)
    st.caption("Entrenamiento y predicción de retornos en USD; cotizaciones en ARS convertidas con CCL histórico.")

    returns = load_returns()
    all_tickers = list(returns.columns)
    prices = get_prices(returns)
    cfg = build_selected_features()

    st.info(
        f"""**Universo:** todos los tickers disponibles en `returns.csv` (`{len(all_tickers)}` activos)  
**Target:** `{cfg["target_aggregation"]}` de los próximos `{cfg["forecast_horizon"]}` retornos diarios  
**Ventana de entrada:** últimos `{cfg["seq_len"]}` días hábiles  
**Inicio de entrenamiento:** `{cfg["start_date"]}`  
**Features seleccionadas:** {", ".join(cfg["selected_features"]) if cfg["selected_features"] else "None"}"""
    )

    st.caption("Se comparan 64→64 y 120→64, activaciones ReLU/Tanh/GELU en la capa densa y dos configuraciones Adam. Selección por MSE de validación; test reservado. El horizonte es de cinco ruedas.")
    run_button = st.button("Entrenar y generar predicciones para el universo completo", use_container_width=True)

    if run_button:
        if not cfg["selected_features"]:
            st.error("Seleccioná al menos una feature.")
            st.stop()

        with st.spinner("Entrenando modelos LSTM configurables para todos los tickers disponibles..."):
            result = run_lstm_lab(
                search=cfg["search"],
                feature_search=cfg["feature_search"],
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
    if result is not None:
        try:
            assert_current_dataset(result["predictions"])
        except ValueError:
            st.session_state.pop("lstm_lab_result", None)
            result = None

    runs_df = get_prediction_runs_df()

    with st.expander("Corridas LSTM guardadas (Postgres)", expanded=False):
        if runs_df.empty:
            st.info("Todavía no hay corridas almacenadas en Postgres.")
        else:
            st.dataframe(runs_df, use_container_width=True, hide_index=True)
            active_candidates = runs_df["run_id"].tolist()
            selected_active = st.selectbox("Definir corrida activa para el optimizador", active_candidates)
            if st.button("Definir corrida seleccionada como activa"):
                set_active_prediction_run(selected_active)
                st.success(f"Corrida activa definida: {selected_active}")
                st.rerun()

    if result is None:
        st.warning("Todavía no hay una corrida LSTM disponible. Lanzá una desde esta página.")
        return

    pred_df = result["predictions"].copy()
    metrics_df = result["metrics"].copy()
    summary = result.get("summary", {})

    if pred_df.empty or metrics_df.empty:
        st.warning("La corrida terminó, pero no se generaron predicciones o métricas.")
        return

    merged = metrics_df.merge(pred_df, on=["ticker", "mu_pred"], how="outer")

    st.success(f"Corrida LSTM disponible: {result['run_id']}")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Tamaño del universo", int(summary.get("total_tickers_in_universe", len(all_tickers))))
    with c2:
        st.metric("Tickers predichos", int(summary.get("tickers_with_predictions", len(pred_df))))
    with c3:
        st.metric("RMSE test promedio", f"{merged['rmse_test'].mean():.6f}" if "rmse_test" in merged.columns else "-")
    with c4:
        st.metric("Directional accuracy test promedio", f"{merged['dir_acc_test'].mean():.2%}" if "dir_acc_test" in merged.columns else "-")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric("mu_pred promedio", f"{pred_df['mu_pred'].mean():.6f}")
    with c6:
        st.metric("mu_pred mediana", f"{pred_df['mu_pred'].median():.6f}")
    with c7:
        st.metric("mu_pred mínima", f"{pred_df['mu_pred'].min():.6f}")
    with c8:
        st.metric("mu_pred máxima", f"{pred_df['mu_pred'].max():.6f}")

    tab1, tab2, tab3, tab4 = st.tabs(["Resumen", "Rankings", "Filtros y tabla", "Detalle por ticker"])

    with tab1:
        render_overview(pred_df, merged, result)

    with tab2:
        render_rankings(merged)

    with tab3:
        render_filter_table(merged)

    with tab4:
        render_ticker_detail(merged, result, summary, prices)

    st.warning(DISCLAIMER_TEXT)


if __name__ == "__main__":
    main()
