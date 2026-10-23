from __future__ import annotations

from app.api_client import api_headers

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(
    page_title="Historial de corridas",
    page_icon="🗂️",
    layout="wide",
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
DISCLAIMER_TEXT = "Este sistema es únicamente informativo y educativo. No constituye una recomendación de inversión ni asesoramiento financiero."


def pretty_pct(x, digits: int = 2) -> str:
    try:
        return f"{100 * float(x):,.{digits}f}%"
    except Exception:
        return "-"


def render_disclaimer() -> None:
    st.warning(DISCLAIMER_TEXT)


def render_metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
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


def fetch_runs(api_base_url: str, limit: int = 200) -> list[dict]:
    response = requests.get(headers=api_headers(), url=f"{api_base_url}/runs", params={"limit": limit}, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_run_detail(api_base_url: str, run_id: int) -> dict:
    response = requests.get(headers=api_headers(), url=f"{api_base_url}/runs/{run_id}", timeout=30)
    response.raise_for_status()
    return response.json()["run"]


def build_frontier_figure(frontier_df: pd.DataFrame, comparison_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if not frontier_df.empty:
        fig.add_trace(
            go.Scatter(
                x=frontier_df["vol_annual"],
                y=frontier_df["return_annual"],
                mode="lines",
                name="Frontera eficiente",
                line=dict(width=3),
            )
        )

    for _, row in comparison_df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["vol_annual"]],
                y=[row["return_annual"]],
                mode="markers+text",
                name=row["portfolio"],
                text=[row["portfolio"]],
                textposition="top center",
                marker=dict(size=14),
            )
        )

    fig.update_layout(
        title="Frontera guardada",
        xaxis_title="Volatilidad anual",
        yaxis_title="Retorno logarítmico anual",
        height=600,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    return fig


st.title("Historial de corridas")
render_disclaimer()
st.caption("Historial de optimizaciones servido por el backend FastAPI.")

api_base_url = st.sidebar.text_input("URL base de la API", value=API_BASE_URL, disabled=os.getenv("APP_ENV") == "production")

left_top, right_top = st.columns([0.9, 1.1])
with left_top:
    if st.button("Actualizar corridas", use_container_width=True):
        st.rerun()
with right_top:
    st.markdown(f"**API conectada:** `{api_base_url}`")

try:
    runs = fetch_runs(api_base_url=api_base_url, limit=200)
except Exception as exc:
    st.error(f"No se pudieron cargar las corridas desde la API: {exc}")
    st.stop()

if not runs:
    st.info("Todavía no hay corridas guardadas.")
    st.stop()

history_df = pd.DataFrame(runs)

st.subheader("Últimas corridas")
display_df = history_df.copy()

st.dataframe(
    display_df.style.format(
        {
            "max_weight": "{:.2%}",
            "rf_annual_simple": "{:.2%}",
            "return_annual": "{:.2%}",
            "vol_annual": "{:.2%}",
            "sharpe_annual": "{:.3f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

run_options = history_df["id"].tolist()

selected_run_id = st.selectbox(
    "Elegí una corrida",
    options=run_options,
    format_func=lambda rid: (
        lambda row: f"#{int(row['id'])} | {row['model_type']} | {row['created_at']} | {row['run_label'] or 'sin_etiqueta'}"
    )(history_df.loc[history_df["id"] == rid].iloc[0]),
)

try:
    run = fetch_run_detail(api_base_url=api_base_url, run_id=int(selected_run_id))
except Exception as exc:
    st.error(f"No se pudo cargar el detalle de la corrida: {exc}")
    st.stop()

metrics = run["metrics_json"]
if metrics.get("metrics_version", 1) < 2:
    st.info("Corrida histórica: conserva sus métricas originales. El campo Sharpe anual anterior a v2 podía contener Sharpe diario; no se recalcula este registro.")
comparison_df = pd.DataFrame(run["comparison_json"])
weights_df = pd.DataFrame(run["weights_json"])
exposure_df = pd.DataFrame(run["block_exposure_json"])
frontier_df = pd.DataFrame(run["frontier_json"])
targets = run["targets_json"]

m1, m2, m3 = st.columns(3)
with m1:
    render_metric_card("Retorno logarítmico anual", pretty_pct(metrics["return_annual"]))
with m2:
    render_metric_card("Volatilidad anual", pretty_pct(metrics["vol_annual"]))
with m3:
    render_metric_card("Sharpe anualizado", f"{metrics['sharpe_annual']:.3f}")

left, right = st.columns([0.95, 1.05])
with left:
    st.subheader("Configuración de la corrida")
    st.json(
        {
            "id": run["id"],
            "created_at": str(run["created_at"]),
            "run_label": run["run_label"],
            "model_type": run["model_type"],
            "max_weight": run["max_weight"],
            "rf_annual_simple": run["rf_annual_simple"],
            "n_frontier_points": run["n_frontier_points"],
            "targets": targets,
            "solver_message": run["solver_message"],
        },
        expanded=True,
    )

with right:
    st.subheader("Comparación guardada")
    st.dataframe(
        comparison_df.style.format(
            {
                "return_annual": "{:.2%}",
                "vol_annual": "{:.2%}",
                "sharpe_annual": "{:.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

st.plotly_chart(build_frontier_figure(frontier_df, comparison_df), use_container_width=True)

tab1, tab2, tab3, tab4 = st.tabs(["Pesos", "Exposición por bloques", "Datos de frontera", "JSON crudo"])

with tab1:
    st.dataframe(
        weights_df.style.format({"weight": "{:.2%}"}),
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    st.dataframe(
        exposure_df.style.format({"target": "{:.2%}", "achieved": "{:.2%}", "diff": "{:+.2%}"}),
        use_container_width=True,
        hide_index=True,
    )

with tab3:
    st.dataframe(
        frontier_df.style.format(
            {
                "return_daily": "{:.4%}",
                "vol_daily": "{:.4%}",
                "return_annual": "{:.2%}",
                "vol_annual": "{:.2%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab4:
    st.json(run)

render_disclaimer()
