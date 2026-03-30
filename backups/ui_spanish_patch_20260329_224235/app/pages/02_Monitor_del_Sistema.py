from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

st.set_page_config(
    page_title="Monitor del sistema",
    page_icon="🖥️",
    layout="wide",
)

API_BROWSER_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
MLFLOW_BROWSER_URL = os.getenv("MLFLOW_UI_URL", "http://127.0.0.1:5005")
MINIO_BROWSER_URL = os.getenv("MINIO_CONSOLE_URL", "http://127.0.0.1:9001")

API_INTERNAL_URL = os.getenv("API_INTERNAL_URL", "http://fastapi:8000")
MLFLOW_INTERNAL_URL = os.getenv("MLFLOW_INTERNAL_URL", "http://mlflow:5000")
MINIO_INTERNAL_URL = os.getenv("MINIO_INTERNAL_URL", "http://minio:9001")

DISCLAIMER_TEXT = "Este sistema es únicamente informativo y educativo. No constituye una recomendación de inversión ni asesoramiento financiero."


def render_metric_card(label: str, value: str, color: str = "#1e293b") -> None:
    st.markdown(
        f"""
        <div style="
            background: {color};
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            color: white;
            box-shadow: 0 12px 30px rgba(0,0,0,0.18);
        ">
            <div style="font-size:0.88rem;opacity:0.78;margin-bottom:0.25rem;">{label}</div>
            <div style="font-size:1.55rem;font-weight:700;line-height:1.1;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_get_json(url: str, timeout: int = 15) -> tuple[bool, Any]:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return True, response.json()
    except Exception as exc:
        return False, str(exc)


def safe_get_status(url: str, timeout: int = 10) -> tuple[bool, int | None, str | None]:
    try:
        response = requests.get(url, timeout=timeout)
        return True, response.status_code, None
    except Exception as exc:
        return False, None, str(exc)


st.title("Monitor del sistema")
st.warning(DISCLAIMER_TEXT)
st.caption("Estado operativo de la plataforma de optimización de carteras.")

col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
with col_cfg1:
    api_browser_url = st.text_input("URL pública de la API", value=API_BROWSER_URL)
with col_cfg2:
    mlflow_browser_url = st.text_input("URL pública de MLflow", value=MLFLOW_BROWSER_URL)
with col_cfg3:
    minio_browser_url = st.text_input("URL pública de MinIO", value=MINIO_BROWSER_URL)

st.markdown("---")

api_ok, api_result = safe_get_json(f"{API_INTERNAL_URL}/health")
runs_ok, runs_result = safe_get_json(f"{API_INTERNAL_URL}/runs?limit=5")
mlflow_ok, mlflow_status, mlflow_error = safe_get_status(MLFLOW_INTERNAL_URL)
minio_ok, minio_status, minio_error = safe_get_status(MINIO_INTERNAL_URL)

top1, top2, top3 = st.columns(3)

with top1:
    render_metric_card("FastAPI", "ONLINE" if api_ok else "OFFLINE", "#14532d" if api_ok else "#7f1d1d")

with top2:
    if runs_ok and isinstance(runs_result, list):
        render_metric_card("Corridas guardadas", str(len(runs_result)), "#1d4ed8")
    else:
        render_metric_card("Corridas guardadas", "N/D", "#7f1d1d")

with top3:
    render_metric_card("MLflow", "ONLINE" if mlflow_ok else "OFFLINE", "#14532d" if mlflow_ok else "#7f1d1d")

mid1, mid2, mid3 = st.columns(3)
with mid1:
    render_metric_card("MinIO", "ONLINE" if minio_ok else "OFFLINE", "#14532d" if minio_ok else "#7f1d1d")
with mid2:
    render_metric_card("API interna", API_INTERNAL_URL)
with mid3:
    render_metric_card("Base de datos de corridas", "OK" if runs_ok else "ERROR", "#14532d" if runs_ok else "#7f1d1d")

st.subheader("Enlaces rápidos")
c1, c2, c3 = st.columns(3)
with c1:
    st.link_button("Abrir FastAPI", api_browser_url, use_container_width=True)
with c2:
    st.link_button("Abrir MLflow", mlflow_browser_url, use_container_width=True)
with c3:
    st.link_button("Abrir MinIO", minio_browser_url, use_container_width=True)

st.subheader("Detalle técnico")
left, right = st.columns(2)
with left:
    st.markdown("**Healthcheck de FastAPI**")
    if api_ok:
        st.json(api_result)
    else:
        st.error(api_result)

    st.markdown("**Últimas corridas (API)**")
    if runs_ok:
        st.json(runs_result)
    else:
        st.error(runs_result)

with right:
    st.markdown("**Estado de MLflow**")
    if mlflow_ok:
        st.success(f"Código HTTP: {mlflow_status}")
    else:
        st.error(mlflow_error)

    st.markdown("**Estado de MinIO**")
    if minio_ok:
        st.success(f"Código HTTP: {minio_status}")
    else:
        st.error(minio_error)

st.warning(DISCLAIMER_TEXT)
