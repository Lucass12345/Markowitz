from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.load_data import load_returns
from src.models.train_volatility import forecast_all_volatility
from app.garch_lab_storage import save_garch_run

PRED_PATH = ROOT / "outputs" / "predictions" / "volatility_predictions.csv"


def _get_file_mtime(path: Path) -> str:
    if not path.exists():
        return "No generado todavía"
    return pd.Timestamp(path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")


def _load_current_predictions() -> pd.DataFrame:
    if not PRED_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(PRED_PATH)
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.strip()
    return df


st.title("GARCH Lab")
st.caption(
    "Esta página usa directamente Data/returns.csv. "
    "Cuando actualizás las predicciones, se escribe outputs/predictions/volatility_predictions.csv "
    "y el modo ai_volatility de Markowitz consume ese archivo en forma directa."
)

returns = load_returns().copy()
returns.index = pd.to_datetime(returns.index)

with st.sidebar:
    st.header("Configuración")
    start_date = st.selectbox(
        "Training start date",
        options=["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"],
        index=2,
    )
    min_obs = st.number_input(
        "Minimum observations per ticker",
        min_value=60,
        max_value=5000,
        value=252,
        step=10,
    )
    save_run = st.checkbox("Guardar run en outputs/garch_lab/runs", value=True)

st.subheader("Dataset fuente")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(returns):,}")
c2.metric("Tickers", f"{returns.shape[1]:,}")
c3.metric("Date range", f"{returns.index.min().date()} → {returns.index.max().date()}")

with st.expander("Preview Data/returns.csv", expanded=False):
    st.dataframe(returns.tail(20), use_container_width=True)

st.subheader("Archivo que usa Markowitz")
st.code(
    f"Fuente optimizer ai_volatility:\n{PRED_PATH}\n"
    f"Última actualización: {_get_file_mtime(PRED_PATH)}"
)

col_run, col_preview = st.columns([1, 1])

with col_run:
    if st.button("Actualizar volatilidades desde Data/returns.csv", use_container_width=True):
        with st.spinner("Corriendo GJR-GARCH / EWMA fallback por ticker..."):
            df_pred = forecast_all_volatility(
                start_date=str(start_date),
                min_obs=int(min_obs),
            )

        if df_pred.empty:
            st.warning("No se generaron predicciones. Revisá start_date / min_obs.")
        else:
            run_id = None
            if save_run:
                metadata = {
                    "model": "GJR-GARCH(1,1,1) Student-t",
                    "source_dataset": "Data/returns.csv",
                    "prediction_output": str(PRED_PATH),
                    "start_date": str(start_date),
                    "min_obs": int(min_obs),
                    "rows": int(len(df_pred)),
                    "created_at": pd.Timestamp.utcnow().isoformat(),
                }
                run_id = save_garch_run(df_pred, metadata)

            st.success("Archivo de volatilidades actualizado correctamente.")
            if run_id:
                st.info(f"Run guardado: {run_id}")

            st.write("Resumen por fuente de modelo")
            if "model_source" in df_pred.columns:
                summary = (
                    df_pred["model_source"]
                    .value_counts(dropna=False)
                    .rename_axis("model_source")
                    .reset_index(name="count")
                )
                st.dataframe(summary, use_container_width=True, hide_index=True)

            st.write("Preview de volatility_predictions.csv")
            st.dataframe(df_pred, use_container_width=True, hide_index=True)

with col_preview:
    current = _load_current_predictions()
    if current.empty:
        st.info("Todavía no existe outputs/predictions/volatility_predictions.csv")
    else:
        st.write("Preview archivo actual")
        st.dataframe(current, use_container_width=True, hide_index=True)

        if {"vol_pred"}.issubset(current.columns):
            st.write("Top 15 volatilidades predichas")
            top_df = current.sort_values("vol_pred", ascending=False).head(15).copy()
            st.bar_chart(top_df.set_index("ticker")["vol_pred"])

st.divider()
st.subheader("Notas operativas")
st.markdown(
    "- Ya no hace falta subir ningún CSV a Streamlit.\n"
    "- Esta pantalla lee directamente `Data/returns.csv`.\n"
    "- El optimizer `ai_volatility` ya no depende de `predictions.csv`; toma `volatility_predictions.csv` en forma directa.\n"
    "- El modo `ai_returns` sigue usando `predictions.csv` para `mu_pred`, sin tocar tu flujo MLP actual."
)
