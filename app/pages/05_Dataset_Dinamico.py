from datetime import date
import os
from types import SimpleNamespace
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import streamlit as st
from src.data.load_data import load_metadata, active_data_dir, assert_usd_dataset
from src.data.yahoo_dataset import refresh_dataset

st.set_page_config(page_title="Dataset Yahoo Finance", layout="wide")
st.title("Dataset dinámico de Yahoo Finance")
st.caption("Precios ajustados y retornos logarítmicos diarios en USD. Cotizaciones en ARS convertidas con CCL vendedor antes de calcular retornos.")
try:
    assert_usd_dataset()
except ValueError as exc:
    st.warning(str(exc))
current = load_metadata()
tickers = st.text_area("Tickers de Yahoo Finance (separados por espacios o comas)", ", ".join(current["ticker"]))
start = st.date_input("Inicio del historial", value=date(2015, 1, 1))
st.info("Moneda base: USD. ARS: precio en pesos ÷ CCL histórico. Otras monedas: cambio histórico de Yahoo. No se publica un dataset si falta un cambio o supera siete días de antigüedad.")
if os.getenv("APP_ENV") == "production":
    st.info("Serving monta los datos en modo lectura. Publicá un candidato desde el job de mantenimiento; ver docs/publication.md.")
    st.dataframe(current, use_container_width=True)
    st.stop()
retrain = st.checkbox("Reentrenar LSTM y GARCH después de descargar", value=False)
if st.button("Descargar y activar dataset", type="primary"):
    try:
        if retrain:
            from scripts.run_refresh_dataset import build_candidate
            with st.spinner("Construyendo y validando candidato. Para trabajos extensos se recomienda el workflow o CLI de mantenimiento…"):
                release = build_candidate(SimpleNamespace(tickers=tickers.replace(",", " ").split(), start=start.isoformat(),
                    ccl_file=None, max_fx_age_days=7, epochs=30, min_signal_coverage=.25, candidate_only=False))
            st.success(f"Release {release.name} activado con datos y modelos verificados.")
        else:
            with st.spinner("Descargando precios e identificación de cada activo…"):
                manifest = refresh_dataset(tickers.replace(",", " ").split(), start=start.isoformat())
            st.success(f"Dataset {manifest['version']} activado.")
            st.warning("Reentrená las predicciones antes de usar IA sobre esta versión.")
    except Exception as exc:
        st.error(str(exc))
st.write("Dataset activo:", str(active_data_dir()))
st.dataframe(load_metadata(), use_container_width=True)
st.caption("País y sector desconocidos se muestran vacíos cuando Yahoo no los informa. El país del emisor no implica exposición geográfica del fondo. OTHER requiere definir bloques compatibles al optimizar.")
