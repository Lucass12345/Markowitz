from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(
    page_title="Run History",
    page_icon="🗂️",
    layout="wide",
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def pretty_pct(x, digits: int = 2) -> str:
    try:
        return f"{100 * float(x):,.{digits}f}%"
    except Exception:
        return "-"


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
                name="Efficient Frontier",
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
        title="Saved frontier",
        xaxis_title="Annual Volatility",
        yaxis_title="Annual Return",
        height=600,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    return fig


st.title("Saved Run History")
st.caption("Run history served by FastAPI backend.")

api_base_url = st.sidebar.text_input("API base URL", value=API_BASE_URL)

left_top, right_top = st.columns([0.9, 1.1])
with left_top:
    if st.button("Refresh runs", use_container_width=True):
        st.rerun()
with right_top:
    st.markdown(f"**Connected API:** `{api_base_url}`")

try:
    runs = fetch_runs(api_base_url=api_base_url, limit=200)
except Exception as exc:
    st.error(f"Could not load runs from API: {exc}")
    st.stop()

if not runs:
    st.info("No saved runs yet.")
    st.stop()

history_df = pd.DataFrame(runs)

st.subheader("Latest runs")
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
    "Choose a run",
    options=run_options,
    format_func=lambda rid: (
        lambda row: f"#{int(row['id'])} | {row['model_type']} | {row['created_at']} | {row['run_label'] or 'no-label'}"
    )(history_df.loc[history_df["id"] == rid].iloc[0]),
)

try:
    run = fetch_run_detail(api_base_url=api_base_url, run_id=int(selected_run_id))
except Exception as exc:
    st.error(f"Could not load run detail: {exc}")
    st.stop()

metrics = run["metrics_json"]
comparison_df = pd.DataFrame(run["comparison_json"])
weights_df = pd.DataFrame(run["weights_json"])
exposure_df = pd.DataFrame(run["block_exposure_json"])
frontier_df = pd.DataFrame(run["frontier_json"])
targets = run["targets_json"]

m1, m2, m3 = st.columns(3)
with m1:
    render_metric_card("Annual Return", pretty_pct(metrics["return_annual"]))
with m2:
    render_metric_card("Annual Volatility", pretty_pct(metrics["vol_annual"]))
with m3:
    render_metric_card("Sharpe Ratio", f"{metrics['sharpe_annual']:.3f}")

left, right = st.columns([0.95, 1.05])
with left:
    st.subheader("Run config")
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
    st.subheader("Saved comparison")
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

tab1, tab2, tab3, tab4 = st.tabs(["Weights", "Block exposures", "Frontier data", "Raw run JSON"])

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
