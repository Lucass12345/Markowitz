from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

st.set_page_config(
    page_title="System Monitor",
    page_icon="🖥️",
    layout="wide",
)

# URLs para mostrar en el navegador del usuario
API_BROWSER_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
MLFLOW_BROWSER_URL = os.getenv("MLFLOW_UI_URL", "http://127.0.0.1:5005")
MINIO_BROWSER_URL = os.getenv("MINIO_CONSOLE_URL", "http://127.0.0.1:9001")

# URLs internas entre containers
API_INTERNAL_URL = os.getenv("API_INTERNAL_URL", "http://fastapi:8000")
MLFLOW_INTERNAL_URL = os.getenv("MLFLOW_INTERNAL_URL", "http://mlflow:5000")
MINIO_INTERNAL_URL = os.getenv("MINIO_INTERNAL_URL", "http://minio:9001")


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


st.title("System Monitor")
st.caption("Operational status of the portfolio optimization platform.")

col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
with col_cfg1:
    api_browser_url = st.text_input("API browser URL", value=API_BROWSER_URL)
with col_cfg2:
    mlflow_browser_url = st.text_input("MLflow browser URL", value=MLFLOW_BROWSER_URL)
with col_cfg3:
    minio_browser_url = st.text_input("MinIO browser URL", value=MINIO_BROWSER_URL)

st.markdown("---")

# Health checks internos entre containers
api_ok, api_result = safe_get_json(f"{API_INTERNAL_URL}/health")
runs_ok, runs_result = safe_get_json(f"{API_INTERNAL_URL}/runs?limit=5")
mlflow_ok, mlflow_status, mlflow_error = safe_get_status(MLFLOW_INTERNAL_URL)
minio_ok, minio_status, minio_error = safe_get_status(MINIO_INTERNAL_URL)

top1, top2, top3 = st.columns(3)

with top1:
    if api_ok:
        render_metric_card("FastAPI", "ONLINE", "#14532d")
    else:
        render_metric_card("FastAPI", "OFFLINE", "#7f1d1d")

with top2:
    if runs_ok and isinstance(runs_result, list):
        render_metric_card("Saved Runs", str(len(runs_result)), "#1d4ed8")
    else:
        render_metric_card("Saved Runs", "N/A", "#7f1d1d")

with top3:
    if runs_ok and isinstance(runs_result, list) and len(runs_result) > 0:
        latest_id = runs_result[0].get("id", "N/A")
        render_metric_card("Latest Run ID", str(latest_id), "#7c3aed")
    else:
        render_metric_card("Latest Run ID", "N/A", "#7f1d1d")

st.subheader("Service Status")

svc1, svc2, svc3 = st.columns(3)

with svc1:
    st.markdown(f"**FastAPI docs:** [Open]({api_browser_url}/docs)")
    if api_ok:
        st.success("API responding correctly")
    else:
        st.error(f"API error: {api_result}")

with svc2:
    st.markdown(f"**MLflow UI:** [Open]({mlflow_browser_url})")
    if mlflow_ok and mlflow_status == 200:
        st.success("MLflow UI reachable")
    elif mlflow_ok:
        st.warning(f"MLflow returned status {mlflow_status}")
    else:
        st.error(f"MLflow error: {mlflow_error}")

with svc3:
    st.markdown(f"**MinIO Console:** [Open]({minio_browser_url})")
    if minio_ok and minio_status == 200:
        st.success("MinIO Console reachable")
    elif minio_ok:
        st.warning(f"MinIO returned status {minio_status}")
    else:
        st.error(f"MinIO error: {minio_error}")

st.markdown("---")

st.subheader("Latest Runs Snapshot")

if runs_ok and isinstance(runs_result, list) and len(runs_result) > 0:
    st.dataframe(runs_result, use_container_width=True, hide_index=True)
else:
    st.info("No runs available or API unavailable.")

st.markdown("---")

with st.expander("Environment Summary"):
    st.json(
        {
            "API_BROWSER_URL": api_browser_url,
            "MLFLOW_BROWSER_URL": mlflow_browser_url,
            "MINIO_BROWSER_URL": minio_browser_url,
            "API_INTERNAL_URL": API_INTERNAL_URL,
            "MLFLOW_INTERNAL_URL": MLFLOW_INTERNAL_URL,
            "MINIO_INTERNAL_URL": MINIO_INTERNAL_URL,
        },
        expanded=True,
    )