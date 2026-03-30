
#!/usr/bin/env python3
from __future__ import annotations

import shutil
import os
import stat
import sys
from datetime import datetime
from pathlib import Path


MAIN_APP_CODE = """
from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.db import init_db

st.set_page_config(
    page_title="Optimizador Markowitz",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BLOCKS = ["AR", "US", "FUNDS_EQUITY", "FUNDS_BONDS", "GOLD"]
DEFAULT_TARGETS = {
    "AR": 0.20,
    "US": 0.50,
    "FUNDS_EQUITY": 0.10,
    "FUNDS_BONDS": 0.10,
    "GOLD": 0.10,
}
MODES = ["classic", "ai_returns", "ai_returns_lstm", "ai_volatility"]
MODE_LABELS = {
    "classic": "Markowitz clásico",
    "ai_returns": "Retornos IA (MLP)",
    "ai_returns_lstm": "Retornos IA (LSTM)",
    "ai_volatility": "Volatilidad IA (GJR-GARCH)",
}
MODE_HELP = {
    "classic": "Retornos esperados históricos + covarianza histórica.",
    "ai_returns": "Retornos esperados predichos por MLP + covarianza histórica.",
    "ai_returns_lstm": "Retornos esperados predichos por LSTM + covarianza histórica.",
    "ai_volatility": "Retornos esperados históricos + covarianza reescalada con volatilidad predicha desde outputs/predictions/volatility_predictions.csv.",
}

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
DISCLAIMER_TEXT = "Este sistema es únicamente informativo y educativo. No constituye una recomendación de inversión ni asesoramiento financiero."


def inject_css() -> None:
    st.markdown(
        \"\"\"
        <style>
            .block-container {
                padding-top: 1.0rem;
                padding-bottom: 1.0rem;
                max-width: 1500px;
            }
            .metric-card {
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 18px;
                padding: 1rem 1.1rem;
                color: white;
                box-shadow: 0 12px 30px rgba(0,0,0,0.18);
            }
            .metric-label {
                font-size: 0.88rem;
                opacity: 0.78;
                margin-bottom: 0.25rem;
            }
            .metric-value {
                font-size: 1.55rem;
                font-weight: 700;
                line-height: 1.1;
            }
            .small-note {
                color: #6b7280;
                font-size: 0.90rem;
            }
            .disclaimer-box {
                background: #fff7ed;
                border: 1px solid #fdba74;
                color: #9a3412;
                border-radius: 12px;
                padding: 0.85rem 1rem;
                font-size: 0.95rem;
                margin-bottom: 0.8rem;
            }
        </style>
        \"\"\",
        unsafe_allow_html=True,
    )


def render_disclaimer() -> None:
    st.markdown(
        f\"<div class='disclaimer-box'><strong>Aviso:</strong> {DISCLAIMER_TEXT}</div>\",
        unsafe_allow_html=True,
    )


def pretty_pct(x: Any, digits: int = 2) -> str:
    try:
        return f"{100 * float(x):,.{digits}f}%"
    except Exception:
        return "-"


def normalize_targets(targets: dict[str, float]) -> dict[str, float]:
    total = float(sum(targets.values()))
    if total <= 0:
        equal = 1.0 / len(targets)
        return {k: equal for k in targets}
    return {k: float(v) / total for k, v in targets.items()}


def render_metric_card(label: str, value: str) -> None:
    st.markdown(
        f\"\"\"
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        \"\"\",
        unsafe_allow_html=True,
    )


def build_frontier_figure(result: dict) -> go.Figure:
    frontier_df = pd.DataFrame(result["frontier"])
    minvar = result["minvar"]["stats"]
    maxsharpe = result["maxsharpe"]["stats"]
    blocks = result["blocks"]["stats"]

    fig = go.Figure()

    if not frontier_df.empty:
        fig.add_trace(
            go.Scatter(
                x=frontier_df["vol_annual"],
                y=frontier_df["return_annual"],
                mode="lines",
                name="Frontera eficiente",
                line=dict(width=3),
                hovertemplate="Volatilidad: %{x:.2%}<br>Retorno: %{y:.2%}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[minvar["vol_annual"]],
            y=[minvar["return_annual"]],
            mode="markers+text",
            name="Mínima varianza",
            text=["MinVar"],
            textposition="top center",
            marker=dict(size=14, symbol="diamond"),
            hovertemplate="Mínima varianza<br>Volatilidad: %{x:.2%}<br>Retorno: %{y:.2%}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[maxsharpe["vol_annual"]],
            y=[maxsharpe["return_annual"]],
            mode="markers+text",
            name="Máximo Sharpe",
            text=["MaxSharpe"],
            textposition="top center",
            marker=dict(size=14, symbol="star"),
            hovertemplate="Máximo Sharpe<br>Volatilidad: %{x:.2%}<br>Retorno: %{y:.2%}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[blocks["vol_annual"]],
            y=[blocks["return_annual"]],
            mode="markers+text",
            name="Máximo Sharpe + bloques",
            text=["Bloques"],
            textposition="bottom center",
            marker=dict(size=16, symbol="x"),
            hovertemplate="Bloques<br>Volatilidad: %{x:.2%}<br>Retorno: %{y:.2%}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Frontera eficiente y selección de cartera",
        xaxis_title="Volatilidad anual",
        yaxis_title="Retorno anual",
        height=620,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=20, r=20, t=70, b=20),
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    return fig


def build_weights_bar(weights_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    df = weights_df.copy().sort_values("weight", ascending=False).head(top_n)
    fig = go.Figure(
        go.Bar(
            x=df["ticker"],
            y=df["weight"],
            hovertemplate="%{x}<br>Peso: %{y:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Top {min(top_n, len(df))} ponderaciones",
        xaxis_title="Ticker",
        yaxis_title="Peso",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_yaxes(tickformat=".0%")
    return fig


def build_block_exposure_chart(exposure_df: pd.DataFrame) -> go.Figure:
    df = exposure_df.copy()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["block"],
            y=df["target"],
            name="Objetivo",
            hovertemplate="%{x}<br>Objetivo: %{y:.2%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=df["block"],
            y=df["achieved"],
            name="Logrado",
            hovertemplate="%{x}<br>Logrado: %{y:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Exposición por bloque: objetivo vs. logrado",
        barmode="group",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_yaxes(tickformat=".0%")
    return fig


@st.cache_data(show_spinner=False)
def check_api_health(api_base_url: str) -> dict:
    response = requests.get(f"{api_base_url}/health", timeout=15)
    response.raise_for_status()
    return response.json()


def call_optimize_api(
    api_base_url: str,
    model_type: str,
    targets: dict[str, float],
    max_weight: float,
    rf_annual_simple: float,
    n_frontier_points: int,
    run_label: str,
) -> dict:
    payload = {
        "model_type": model_type,
        "targets": targets,
        "max_weight": float(max_weight),
        "rf_annual_simple": float(rf_annual_simple),
        "n_frontier_points": int(n_frontier_points),
        "run_label": run_label,
    }

    response = requests.post(
        f"{api_base_url}/optimize",
        json=payload,
        timeout=300,
    )

    if not response.ok:
        raise RuntimeError(f"Error de API {response.status_code}: {response.text}")

    return response.json()


def build_comparison_df(result: dict) -> pd.DataFrame:
    rows = []
    for key, label in [
        ("minvar", "Mínima varianza"),
        ("maxsharpe", "Máximo Sharpe"),
        ("blocks", "Máximo Sharpe + bloques"),
    ]:
        stats = result[key]["stats"] if key != "blocks" else result["blocks"]["stats"]
        rows.append(
            {
                "portfolio": label,
                "return_annual": stats["return_annual"],
                "vol_annual": stats["vol_annual"],
                "sharpe_annual": stats["sharpe_annual"],
            }
        )
    return pd.DataFrame(rows)


def sidebar_controls() -> dict:
    st.sidebar.title("Parámetros de cartera")
    st.sidebar.caption("Interfaz Streamlit conectada al backend FastAPI.")

    api_base_url = st.sidebar.text_input(
        "URL base de la API",
        value=API_BASE_URL,
        help="URL del backend FastAPI.",
    )

    mode = st.sidebar.selectbox(
        "Modo de optimización",
        options=MODES,
        format_func=lambda x: MODE_LABELS[x],
    )
    st.sidebar.info(MODE_HELP[mode])

    st.sidebar.markdown("### Objetivos por bloque")
    raw_targets = {}
    for block in BLOCKS:
        raw_targets[block] = st.sidebar.number_input(
            f"{block}",
            min_value=0.0,
            max_value=1.0,
            value=float(DEFAULT_TARGETS[block]),
            step=0.01,
            format="%.2f",
        )

    raw_total = float(sum(raw_targets.values()))
    normalize = st.sidebar.checkbox("Normalizar automáticamente a 100%", value=True)

    if abs(raw_total - 1.0) < 1e-6:
        st.sidebar.success(f"Suma ingresada: {raw_total:.2%}")
    else:
        if normalize:
            st.sidebar.warning(
                f"La suma ingresada es {raw_total:.2%}. Se normalizará automáticamente antes de correr la optimización."
            )
        else:
            st.sidebar.error(
                f"La suma ingresada es {raw_total:.2%}. Para continuar sin normalizar, los pesos deben sumar 100%."
            )

    max_weight = st.sidebar.slider("Peso máximo por activo", 0.01, 1.00, 0.15, 0.01)
    rf_annual_simple = st.sidebar.number_input(
        "Tasa libre de riesgo anual",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.005,
        format="%.3f",
    )
    n_frontier_points = st.sidebar.slider("Puntos de la frontera", 30, 200, 80, 10)

    run_label = st.sidebar.text_input(
        "Etiqueta de corrida",
        value=f"{mode}_streamlit_run",
    )

    targets = normalize_targets(raw_targets) if normalize else raw_targets
    final_total = float(sum(targets.values()))
    invalid_manual_targets = (not normalize) and abs(raw_total - 1.0) >= 1e-6

    if abs(final_total - 1.0) < 1e-6:
        st.sidebar.success(f"Suma final utilizada: {final_total:.2%}")
    else:
        st.sidebar.warning(f"Suma final utilizada: {final_total:.2%}")

    run_clicked = st.sidebar.button(
        "Correr optimización",
        type="primary",
        use_container_width=True,
        disabled=invalid_manual_targets,
    )

    return {
        "api_base_url": api_base_url,
        "mode": mode,
        "targets": targets,
        "raw_targets": raw_targets,
        "raw_total": raw_total,
        "normalize": normalize,
        "invalid_manual_targets": invalid_manual_targets,
        "max_weight": max_weight,
        "rf_annual_simple": rf_annual_simple,
        "n_frontier_points": n_frontier_points,
        "run_label": run_label,
        "run_clicked": run_clicked,
    }


def main() -> None:
    init_db()
    inject_css()

    st.title("Optimizador Markowitz")
    render_disclaimer()
    st.caption("Plataforma de optimización de carteras conectada a FastAPI.")

    controls = sidebar_controls()

    top_health_left, top_health_right = st.columns([1, 2])
    with top_health_left:
        if st.button("Verificar salud de la API", use_container_width=True):
            try:
                health = check_api_health(controls["api_base_url"])
                st.success(f"API OK: {health}")
            except Exception as exc:
                st.error(f"Falló el healthcheck de la API: {exc}")
    with top_health_right:
        st.markdown(f"**API conectada:** `{controls['api_base_url']}`")

    selected_targets_df = pd.DataFrame(
        {
            "block": list(controls["targets"].keys()),
            "target": list(controls["targets"].values()),
        }
    )

    top_left, top_right = st.columns([1.15, 1])
    with top_left:
        st.subheader("Objetivos seleccionados")
        st.dataframe(
            selected_targets_df.style.format({"target": "{:.2%}"}),
            use_container_width=True,
            hide_index=True,
        )
    with top_right:
        st.subheader("Configuración elegida")
        st.markdown(f"**Modo:** {MODE_LABELS[controls['mode']]}")
        st.markdown(f"**Peso máximo por activo:** {pretty_pct(controls['max_weight'])}")
        st.markdown(f"**Tasa libre de riesgo anual:** {pretty_pct(controls['rf_annual_simple'])}")
        st.markdown(f"**Puntos de frontera:** {controls['n_frontier_points']}")
        st.markdown(f"**Etiqueta de corrida:** {controls['run_label']}")
        st.markdown(f"**Normalización automática:** {'Sí' if controls['normalize'] else 'No'}")
        st.markdown(f"<div class='small-note'>{MODE_HELP[controls['mode']]}</div>", unsafe_allow_html=True)

    if controls["invalid_manual_targets"]:
        st.warning(
            "Los pesos por bloque no suman 100%. Corregilos o activá la normalización automática para poder correr la optimización."
        )

    if controls["run_clicked"]:
        try:
            with st.spinner("Llamando a FastAPI /optimize ..."):
                api_response = call_optimize_api(
                    api_base_url=controls["api_base_url"],
                    model_type=controls["mode"],
                    targets=controls["targets"],
                    max_weight=float(controls["max_weight"]),
                    rf_annual_simple=float(controls["rf_annual_simple"]),
                    n_frontier_points=int(controls["n_frontier_points"]),
                    run_label=controls["run_label"],
                )

            st.session_state["last_api_response"] = api_response
            st.session_state["last_result"] = api_response["result"]
            st.session_state["last_run_id"] = api_response.get("run_id")
            st.session_state["last_mlflow_run_id"] = api_response.get("mlflow_run_id")
            st.session_state["last_mlflow_error"] = api_response.get("mlflow_error")
            st.success("La optimización finalizó correctamente.")

        except Exception as exc:
            st.error(f"La optimización falló: {exc}")
            st.stop()

    if "last_result" not in st.session_state:
        st.info("Configurá la cartera y hacé clic en “Correr optimización”.")
        return

    result = st.session_state["last_result"]

    info1, info2, info3 = st.columns(3)
    with info1:
        st.info(f"Run ID: {st.session_state.get('last_run_id')}")
    with info2:
        st.info(f"MLflow Run ID: {st.session_state.get('last_mlflow_run_id')}")
    with info3:
        mlflow_error = st.session_state.get("last_mlflow_error")
        if mlflow_error:
            st.warning(f"MLflow: {mlflow_error}")
        else:
            st.success("Registro en MLflow OK")

    block_stats = result["blocks"]["stats"]
    m1, m2, m3 = st.columns(3)
    with m1:
        render_metric_card("Retorno anual", pretty_pct(block_stats["return_annual"]))
    with m2:
        render_metric_card("Volatilidad anual", pretty_pct(block_stats["vol_annual"]))
    with m3:
        render_metric_card("Ratio de Sharpe", f"{block_stats['sharpe_annual']:.3f}")

    st.plotly_chart(build_frontier_figure(result), use_container_width=True)

    st.subheader("Comparación de carteras")
    comparison_df = build_comparison_df(result)
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

    weights_df = pd.DataFrame(result["blocks"]["weights_df"])
    exposure_df = pd.DataFrame(result["blocks"]["exposure_by_block"])
    frontier_df = pd.DataFrame(result["frontier"])

    c1, c2 = st.columns([1.15, 0.85])
    with c1:
        st.plotly_chart(build_weights_bar(weights_df), use_container_width=True)
    with c2:
        st.plotly_chart(build_block_exposure_chart(exposure_df), use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Detalle de pesos", "Exposición por bloques", "Datos de frontera", "Respuesta cruda de API"])

    with tab1:
        st.dataframe(
            weights_df.style.format({"weight": "{:.2%}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Descargar pesos en CSV",
            data=weights_df.to_csv(index=False).encode("utf-8"),
            file_name=f"weights_{controls['mode']}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with tab2:
        st.dataframe(
            exposure_df.style.format(
                {"target": "{:.2%}", "achieved": "{:.2%}", "diff": "{:+.2%}"}
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Descargar exposición por bloques en CSV",
            data=exposure_df.to_csv(index=False).encode("utf-8"),
            file_name=f"block_exposure_{controls['mode']}.csv",
            mime="text/csv",
            use_container_width=True,
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
        st.download_button(
            "Descargar frontera en CSV",
            data=frontier_df.to_csv(index=False).encode("utf-8"),
            file_name=f"frontier_{controls['mode']}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with tab4:
        st.json(st.session_state["last_api_response"])

    with st.expander("Notas técnicas"):
        st.markdown(
            f\"\"\"
- La interfaz llama a FastAPI en: `{controls['api_base_url']}`
- Modo seleccionado: `{controls['mode']}`
- Mensaje del solver: `{result['blocks'].get('solver_message', 'N/A')}`
- Run ID: `{st.session_state.get('last_run_id')}`
- MLflow Run ID: `{st.session_state.get('last_mlflow_run_id')}`
- Esta app no llama al optimizador en forma directa.
            \"\"\"
        )

    render_disclaimer()


if __name__ == "__main__":
    main()
"""

STREAMLIT_WRAPPER_CODE = """
from Optimizador_Markowitz import main

if __name__ == "__main__":
    main()
"""

RUN_HISTORY_CODE = """
from __future__ import annotations

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
        f\"\"\"
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
        \"\"\",
        unsafe_allow_html=True,
    )


def fetch_runs(api_base_url: str, limit: int = 200) -> list[dict]:
    response = requests.get(f"{api_base_url}/runs", params={"limit": limit}, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_run_detail(api_base_url: str, run_id: int) -> dict:
    response = requests.get(f"{api_base_url}/runs/{run_id}", timeout=30)
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
        yaxis_title="Retorno anual",
        height=600,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    return fig


st.title("Historial de corridas")
render_disclaimer()
st.caption("Historial de optimizaciones servido por el backend FastAPI.")

api_base_url = st.sidebar.text_input("URL base de la API", value=API_BASE_URL)

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
comparison_df = pd.DataFrame(run["comparison_json"])
weights_df = pd.DataFrame(run["weights_json"])
exposure_df = pd.DataFrame(run["block_exposure_json"])
frontier_df = pd.DataFrame(run["frontier_json"])
targets = run["targets_json"]

m1, m2, m3 = st.columns(3)
with m1:
    render_metric_card("Retorno anual", pretty_pct(metrics["return_annual"]))
with m2:
    render_metric_card("Volatilidad anual", pretty_pct(metrics["vol_annual"]))
with m3:
    render_metric_card("Ratio de Sharpe", f"{metrics['sharpe_annual']:.3f}")

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
"""

SYSTEM_MONITOR_CODE = """
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
        f\"\"\"
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
        \"\"\",
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
"""

LSTM_PAGE_CODE = """
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

    synthetic = (1 + returns.fillna(0)).cumprod() * 100
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

        start_date = st.selectbox(
            "Fecha inicial de entrenamiento",
            options=["full", "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"],
            index=3,
        )

        seq_len = st.slider("Ventana lookback (días hábiles)", 20, 60, 40)
        forecast_horizon = st.slider("Horizonte de pronóstico (días hábiles)", 1, 10, 5)
        target_aggregation = st.radio("Agregación del target", options=["mean", "sum"], horizontal=True)

        hidden_size = st.slider("Tamaño de capa oculta", 16, 128, 64, step=16)
        num_layers = st.slider("Cantidad de capas", 1, 4, 2)
        dropout = st.slider("Dropout", 0.0, 0.5, 0.2, step=0.05)
        epochs = st.slider("Épocas", 10, 150, 50, step=10)
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
        f"Run ID: {result['run_id']}\\n"
        f"Predicciones: {result['predictions_path']}\\n"
        f"Métricas: {result['metrics_path']}\\n"
        f"Predicciones por ticker: {result['per_ticker_dir']}\\n"
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
                st.caption(f"Precio proyectado en {horizon} días hábiles: {projected_price:.4f}")

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
    st.caption("Entrenamiento, predicción, ranking y análisis por ticker sobre el universo completo.")

    returns = load_returns()
    all_tickers = list(returns.columns)
    prices = get_prices(returns)
    cfg = build_selected_features()

    st.info(
        f\"\"\"**Universo:** todos los tickers disponibles en `returns.csv` (`{len(all_tickers)}` activos)  
**Target:** `{cfg["target_aggregation"]}` de los próximos `{cfg["forecast_horizon"]}` retornos diarios  
**Ventana de entrada:** últimos `{cfg["seq_len"]}` días hábiles  
**Inicio de entrenamiento:** `{cfg["start_date"]}`  
**Features seleccionadas:** {", ".join(cfg["selected_features"]) if cfg["selected_features"] else "None"}\"\"\"
    )

    run_button = st.button("Entrenar y generar predicciones para el universo completo", use_container_width=True)

    if run_button:
        if not cfg["selected_features"]:
            st.error("Seleccioná al menos una feature.")
            st.stop()

        with st.spinner("Entrenando modelos LSTM configurables para todos los tickers disponibles..."):
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
"""

GARCH_PAGE_CODE = """
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
        return pd.read_csv(VOL_PATH)
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

    st.subheader("Fuente del optimizador ai_volatility")
    col1, col2 = st.columns(2)
    with col1:
        st.code(str(VOL_PATH))
    with col2:
        st.markdown(f"**Última actualización:** {file_timestamp(VOL_PATH)}")

    returns = load_returns()

    with st.expander("Vista previa de Data/returns.csv", expanded=False):
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
"""

DOCKERFILE_CONTENT = """
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /apprepo

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app/Optimizador_Markowitz.py", "--server.address=0.0.0.0", "--server.port=8501"]
"""


def backup_file(path: Path, backup_root: Path) -> None:
    if not path.exists():
        return
    target = backup_root / path.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.lstrip("\n"), encoding="utf-8")


def _on_rm_error(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def remove_if_exists(path: Path) -> tuple[bool, str | None]:
    try:
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path, onerror=_on_rm_error)
        return True, None
    except PermissionError as e:
        return False, f"{path}: {e}"
    except OSError as e:
        return False, f"{path}: {e}"


def cleanup_repo(repo_root: Path) -> tuple[list[str], list[str]]:
    removed: list[str] = []
    skipped: list[str] = []

    patterns = [
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
        "**/*.bak",
        "**/*bak2.0.txt",
    ]
    for pattern in patterns:
        for path in repo_root.glob(pattern):
            if ".git" in path.parts:
                continue
            if not path.exists():
                continue
            ok, err = remove_if_exists(path)
            rel = str(path.relative_to(repo_root))
            if ok:
                removed.append(rel)
            else:
                skipped.append(err or rel)

    compose_path = repo_root / "docker-compose.yml"
    if compose_path.exists():
        raw = compose_path.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            compose_path.write_bytes(raw[3:])

    return sorted(set(removed)), sorted(set(skipped))


def ensure_root(path: Path) -> None:
    required = [
        path / "app",
        path / "src",
        path / "docker",
        path / "docker-compose.yml",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(
            "No encontré la raíz esperada del repo. Faltan: " + ", ".join(missing)
        )


ROOT = Path.cwd()


def main() -> None:
    ensure_root(ROOT)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = ROOT / "backups" / f"ui_spanish_patch_{timestamp}"
    backup_root.mkdir(parents=True, exist_ok=True)

    files_to_backup = [
        ROOT / "app" / "streamlit_app.py",
        ROOT / "docker" / "streamlit.Dockerfile",
        ROOT / "app" / "pages" / "01_Run_History.py",
        ROOT / "app" / "pages" / "02_System_Monitor.py",
        ROOT / "app" / "pages" / "03_LSTM_Lab.py",
        ROOT / "app" / "pages" / "4_GARCH_Lab.py",
        ROOT / "app" / "pages" / "01_Historial_de_Corridas.py",
        ROOT / "app" / "pages" / "02_Monitor_del_Sistema.py",
        ROOT / "app" / "pages" / "03_Laboratorio_LSTM.py",
        ROOT / "app" / "pages" / "04_Laboratorio_GARCH.py",
        ROOT / "app" / "Optimizador_Markowitz.py",
    ]
    for file_path in files_to_backup:
        backup_file(file_path, backup_root)

    # Escribo launcher principal y compatibilidad.
    write_text(ROOT / "app" / "Optimizador_Markowitz.py", MAIN_APP_CODE)
    write_text(ROOT / "app" / "streamlit_app.py", STREAMLIT_WRAPPER_CODE)
    write_text(ROOT / "docker" / "streamlit.Dockerfile", DOCKERFILE_CONTENT)

    # Renombro/normalizo páginas.
    old_new_pages = [
        (ROOT / "app" / "pages" / "01_Run_History.py", ROOT / "app" / "pages" / "01_Historial_de_Corridas.py", RUN_HISTORY_CODE),
        (ROOT / "app" / "pages" / "02_System_Monitor.py", ROOT / "app" / "pages" / "02_Monitor_del_Sistema.py", SYSTEM_MONITOR_CODE),
        (ROOT / "app" / "pages" / "03_LSTM_Lab.py", ROOT / "app" / "pages" / "03_Laboratorio_LSTM.py", LSTM_PAGE_CODE),
        (ROOT / "app" / "pages" / "4_GARCH_Lab.py", ROOT / "app" / "pages" / "04_Laboratorio_GARCH.py", GARCH_PAGE_CODE),
    ]

    for old_path, new_path, code in old_new_pages:
        if old_path.exists() and old_path != new_path:
            backup_file(old_path, backup_root)
            remove_if_exists(old_path)
        write_text(new_path, code)

    removed, skipped = cleanup_repo(ROOT)

    print("PATCH UI + limpieza aplicado correctamente.")
    print(f"Backups: {backup_root}")
    print("Archivos principales actualizados:")
    print(" - app/Optimizador_Markowitz.py")
    print(" - app/streamlit_app.py")
    print(" - docker/streamlit.Dockerfile")
    print(" - app/pages/01_Historial_de_Corridas.py")
    print(" - app/pages/02_Monitor_del_Sistema.py")
    print(" - app/pages/03_Laboratorio_LSTM.py")
    print(" - app/pages/04_Laboratorio_GARCH.py")
    if removed:
        print("Residuos eliminados:")
        for item in removed:
            print(f" - {item}")
    else:
        print("No se encontraron residuos para eliminar.")
    print("\\nDespués reiniciá Streamlit:")
    print("docker compose restart streamlit fastapi")


if __name__ == "__main__":
    main()
