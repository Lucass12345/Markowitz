from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.load_data import load_returns, active_data_dir, assert_current_dataset
from src.models.train_volatility import forecast_all_volatility

DISCLAIMER_TEXT = "Este sistema es únicamente informativo y educativo. No constituye una recomendación de inversión ni asesoramiento financiero."
VOL_PATH = ROOT / "outputs" / "predictions" / "volatility_predictions.csv"
RETURNS_PATH = ROOT / "Data" / "returns.csv"


def file_timestamp(path: Path) -> str:
    if not path.exists():
        return "Archivo no disponible"
    return pd.Timestamp(path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")


def load_current_vols() -> pd.DataFrame:
    if not VOL_PATH.exists():
        return pd.DataFrame()
    try:
        forecasts = pd.read_csv(VOL_PATH)
        assert_current_dataset(forecasts)
        return forecasts
    except Exception:
        return pd.DataFrame()


def main() -> None:
    st.set_page_config(page_title="Laboratorio GARCH", page_icon="📉", layout="wide")
    st.title("Laboratorio GARCH")
    st.warning(DISCLAIMER_TEXT)

    with st.sidebar:
        st.header("Configuración")
        start_date = st.selectbox(
            "Fecha inicial de entrenamiento",
            options=["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"],
            index=2,
        )
        min_obs = st.slider("Observaciones mínimas por activo", 126, 756, 252, step=21)
        update_now = st.button("Actualizar predicciones de volatilidad", use_container_width=True, type="primary")

    st.caption("Retornos base en USD. GJR-GARCH-t: volatilidad a una y cinco ruedas. Diagnósticos de residuos y parámetros, y backtest de 20 ruedas contra EWMA. var_pred sigue en escala diaria para el optimizador.")
    st.subheader("Fuente del optimizador ai_volatility")
    col1, col2 = st.columns(2)
    with col1:
        st.code(str(VOL_PATH))
    with col2:
        st.markdown(f"**Última actualización:** {file_timestamp(VOL_PATH)}")

    import json
    diagnostics = ROOT / "outputs" / "predictions" / "garch_diagnostics.json"
    if diagnostics.exists():
        with st.expander("Diagnósticos GARCH por activo"):
            st.json(json.loads(diagnostics.read_text(encoding="utf-8")))
    returns = load_returns()

    with st.expander("Vista previa del dataset activo", expanded=False):
        st.dataframe(returns.head(10), use_container_width=True, hide_index=False)

    if update_now:
        with st.spinner("Estimando volatilidades y actualizando volatility_predictions.csv..."):
            df_new = forecast_all_volatility(
                start_date=start_date,
                min_obs=int(min_obs),
            )
        st.session_state["garch_current_df"] = df_new
        st.success("Predicciones de volatilidad actualizadas correctamente.")

    current_df = st.session_state.get("garch_current_df")
    if current_df is None:
        current_df = load_current_vols()

    if current_df is not None and not current_df.empty:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Activos estimados", int(len(current_df)))
        with c2:
            gjr_count = int((current_df.get("model_source", pd.Series(dtype=str)) == "gjr_garch_t").sum())
            st.metric("Corridas GJR-GARCH", gjr_count)
        with c3:
            ewma_count = int((current_df.get("model_source", pd.Series(dtype=str)) == "ewma_fallback").sum())
            st.metric("Fallback EWMA", ewma_count)

        st.subheader("Resumen actual de volatilidades estimadas")
        st.dataframe(current_df.head(20), use_container_width=True, hide_index=True)
    else:
        st.info("Todavía no existe un archivo de volatilidades vigente. Generá una corrida para crearlo.")

    st.warning(DISCLAIMER_TEXT)


if __name__ == "__main__":
    main()
