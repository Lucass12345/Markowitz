from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.db import init_db
from app.repository import get_run_by_id, get_run_history

st.set_page_config(
    page_title="Run History",
    page_icon="🗂️",
    layout="wide",
)

init_db()


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
st.caption("Database-backed history of saved Streamlit optimization runs.")

history_df = get_run_history(limit=200)

if history_df.empty:
    st.info("No saved runs yet. Go back to the main app and save one.")
    st.stop()

st.subheader("Latest runs")
st.dataframe(
    history_df.style.format(
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

run = get_run_by_id(int(selected_run_id))
if not run:
    st.error("Selected run not found.")
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

tab1, tab2, tab3 = st.tabs(["Weights", "Block exposures", "Frontier data"])

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
