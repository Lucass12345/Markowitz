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
        """
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
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer() -> None:
    st.markdown(
        f"<div class='disclaimer-box'><strong>Aviso:</strong> {DISCLAIMER_TEXT}</div>",
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
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
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
            f"""
- La interfaz llama a FastAPI en: `{controls['api_base_url']}`
- Modo seleccionado: `{controls['mode']}`
- Mensaje del solver: `{result['blocks'].get('solver_message', 'N/A')}`
- Run ID: `{st.session_state.get('last_run_id')}`
- MLflow Run ID: `{st.session_state.get('last_mlflow_run_id')}`
- Esta app no llama al optimizador en forma directa.
            """
        )

    render_disclaimer()


if __name__ == "__main__":
    main()
