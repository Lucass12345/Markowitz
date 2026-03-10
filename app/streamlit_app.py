from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.db import init_db
from app.repository import save_run
from app.tracking import track_run_mlflow
from src.optimization.optimizer_service import optimize_portfolio_mode

st.set_page_config(
    page_title="Markowitz AI Optimizer",
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
    "classic": "Classic Markowitz",
    "ai_returns": "AI Returns (MLP)",
    "ai_returns_lstm": "AI Returns (LSTM)",
    "ai_volatility": "AI Volatility (GARCH)",
}
MODE_HELP = {
    "classic": "Historical mean returns + historical covariance.",
    "ai_returns": "Predicted expected returns from MLP + historical covariance.",
    "ai_returns_lstm": "Predicted expected returns from LSTM + historical covariance.",
    "ai_volatility": "Historical expected returns + covariance rescaled using predicted volatility.",
}


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
        </style>
        """,
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
    frontier = result["frontier"].copy()
    minvar = result["minvar"]["stats"]
    maxsharpe = result["maxsharpe"]["stats"]
    blocks = result["blocks"]["stats"]

    fig = go.Figure()

    if not frontier.empty:
        fig.add_trace(
            go.Scatter(
                x=frontier["vol_annual"],
                y=frontier["return_annual"],
                mode="lines",
                name="Efficient Frontier",
                line=dict(width=3),
                hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[minvar["vol_annual"]],
            y=[minvar["return_annual"]],
            mode="markers+text",
            name="Min Variance",
            text=["MinVar"],
            textposition="top center",
            marker=dict(size=14, symbol="diamond"),
            hovertemplate="MinVar<br>Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[maxsharpe["vol_annual"]],
            y=[maxsharpe["return_annual"]],
            mode="markers+text",
            name="Max Sharpe",
            text=["MaxSharpe"],
            textposition="top center",
            marker=dict(size=14, symbol="star"),
            hovertemplate="Max Sharpe<br>Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[blocks["vol_annual"]],
            y=[blocks["return_annual"]],
            mode="markers+text",
            name="Max Sharpe + Blocks",
            text=["Blocks"],
            textposition="bottom center",
            marker=dict(size=16, symbol="x"),
            hovertemplate="Blocks<br>Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Efficient Frontier and Portfolio Selection",
        xaxis_title="Annual Volatility",
        yaxis_title="Annual Return",
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
            hovertemplate="%{x}<br>Weight: %{y:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Top {min(top_n, len(df))} Asset Weights",
        xaxis_title="Ticker",
        yaxis_title="Weight",
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
            name="Target",
            hovertemplate="%{x}<br>Target: %{y:.2%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=df["block"],
            y=df["achieved"],
            name="Achieved",
            hovertemplate="%{x}<br>Achieved: %{y:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Block Exposure: Target vs Achieved",
        barmode="group",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_yaxes(tickformat=".0%")
    return fig


@st.cache_data(show_spinner=False)
def run_optimizer_cached(
    model_type: str,
    targets_tuple: tuple,
    max_weight: float,
    rf_annual_simple: float,
    n_frontier_points: int,
) -> dict:
    targets = dict(targets_tuple)
    return optimize_portfolio_mode(
        model_type=model_type,
        targets=targets,
        max_weight=max_weight,
        rf_annual_simple=rf_annual_simple,
        n_frontier_points=n_frontier_points,
    )


def build_comparison_df(result: dict) -> pd.DataFrame:
    rows = []
    for key, label in [
        ("minvar", "Min Variance"),
        ("maxsharpe", "Max Sharpe"),
        ("blocks", "Max Sharpe + Blocks"),
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
    st.sidebar.title("Portfolio Controls")
    st.sidebar.caption("One optimization mode at a time, mounted on the current backend.")

    mode = st.sidebar.selectbox(
        "Optimization mode",
        options=MODES,
        format_func=lambda x: MODE_LABELS[x],
    )
    st.sidebar.info(MODE_HELP[mode])

    st.sidebar.markdown("### Block targets")
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

    normalize = st.sidebar.checkbox("Normalize targets to 100%", value=True)

    max_weight = st.sidebar.slider("Max weight per asset", 0.01, 1.00, 0.15, 0.01)
    rf_annual_simple = st.sidebar.number_input(
        "Risk-free annual rate",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.005,
        format="%.3f",
    )
    n_frontier_points = st.sidebar.slider("Frontier points", 30, 200, 80, 10)

    targets = normalize_targets(raw_targets) if normalize else raw_targets
    total = sum(targets.values())

    if abs(total - 1.0) < 1e-6:
        st.sidebar.success(f"Target sum: {total:.2%}")
    else:
        st.sidebar.warning(f"Target sum: {total:.2%}")

    run_clicked = st.sidebar.button("Run optimization", type="primary", use_container_width=True)

    return {
        "mode": mode,
        "targets": targets,
        "max_weight": max_weight,
        "rf_annual_simple": rf_annual_simple,
        "n_frontier_points": n_frontier_points,
        "run_clicked": run_clicked,
    }


def main() -> None:
    init_db()
    inject_css()

    st.title("Markowitz AI Portfolio Optimizer")
    st.caption("Streamlit + DB + MLflow on top of the existing optimizer service. Zero repo rewrite.")

    controls = sidebar_controls()

    selected_targets_df = pd.DataFrame(
        {
            "block": list(controls["targets"].keys()),
            "target": list(controls["targets"].values()),
        }
    )

    top_left, top_right = st.columns([1.15, 1])
    with top_left:
        st.subheader("Selected targets")
        st.dataframe(
            selected_targets_df.style.format({"target": "{:.2%}"}),
            use_container_width=True,
            hide_index=True,
        )
    with top_right:
        st.subheader("Selected mode")
        st.markdown(f"**Mode:** {MODE_LABELS[controls['mode']]}")
        st.markdown(f"**Max weight per asset:** {pretty_pct(controls['max_weight'])}")
        st.markdown(f"**Risk-free annual rate:** {pretty_pct(controls['rf_annual_simple'])}")
        st.markdown(f"**Frontier points:** {controls['n_frontier_points']}")
        st.markdown(f"<div class='small-note'>{MODE_HELP[controls['mode']]}</div>", unsafe_allow_html=True)

    if controls["run_clicked"]:
        try:
            with st.spinner("Running optimizer service..."):
                result = run_optimizer_cached(
                    model_type=controls["mode"],
                    targets_tuple=tuple(controls["targets"].items()),
                    max_weight=float(controls["max_weight"]),
                    rf_annual_simple=float(controls["rf_annual_simple"]),
                    n_frontier_points=int(controls["n_frontier_points"]),
                )

            st.session_state["last_result"] = result
            st.session_state["last_config"] = controls

        except Exception as exc:
            st.error(f"Optimization failed: {exc}")
            st.stop()

    if "last_result" not in st.session_state:
        st.info("Set the parameters in the sidebar and click Run optimization.")
        return

    result = st.session_state["last_result"]
    config = st.session_state["last_config"]

    block_stats = result["blocks"]["stats"]
    m1, m2, m3 = st.columns(3)
    with m1:
        render_metric_card("Annual Return", pretty_pct(block_stats["return_annual"]))
    with m2:
        render_metric_card("Annual Volatility", pretty_pct(block_stats["vol_annual"]))
    with m3:
        render_metric_card("Sharpe Ratio", f"{block_stats['sharpe_annual']:.3f}")

    st.plotly_chart(build_frontier_figure(result), use_container_width=True)

    st.subheader("Portfolio comparison")
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

    c1, c2 = st.columns([1.15, 0.85])
    with c1:
        st.plotly_chart(build_weights_bar(result["blocks"]["weights_df"]), use_container_width=True)
    with c2:
        st.plotly_chart(build_block_exposure_chart(result["blocks"]["exposure_by_block"]), use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["Weights detail", "Block exposures", "Frontier data"])

    with tab1:
        weights_df = result["blocks"]["weights_df"].copy()
        st.dataframe(
            weights_df.style.format({"weight": "{:.2%}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download weights CSV",
            data=weights_df.to_csv(index=False).encode("utf-8"),
            file_name=f"weights_{config['mode']}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with tab2:
        exposure_df = result["blocks"]["exposure_by_block"].copy()
        st.dataframe(
            exposure_df.style.format(
                {"target": "{:.2%}", "achieved": "{:.2%}", "diff": "{:+.2%}"}
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download block exposure CSV",
            data=exposure_df.to_csv(index=False).encode("utf-8"),
            file_name=f"block_exposure_{config['mode']}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with tab3:
        frontier_df = result["frontier"].copy()
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
            "Download frontier CSV",
            data=frontier_df.to_csv(index=False).encode("utf-8"),
            file_name=f"frontier_{config['mode']}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("Persist run")

    save_col1, save_col2 = st.columns([1.2, 0.8])
    with save_col1:
        run_label = st.text_input(
            "Run label",
            value=f"{config['mode']}_run",
            help="Optional human-friendly label for DB and MLflow.",
        )
    with save_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        save_clicked = st.button("Save current run", type="secondary", use_container_width=True)

    if save_clicked:
        try:
            run_db_id = save_run(run_label=run_label, config=config, result=result)
            mlflow_run_id, mlflow_error = track_run_mlflow(
                run_db_id=run_db_id,
                run_label=run_label,
                config=config,
                result=result,
            )

            st.success(f"Run saved to database with id={run_db_id}")

            if mlflow_run_id:
                st.info(f"Also logged to MLflow with run_id={mlflow_run_id}")
            elif mlflow_error:
                st.warning(f"Saved in DB, but MLflow logging failed/skipped: {mlflow_error}")

        except Exception as exc:
            st.error(f"Could not save run: {exc}")

    with st.expander("Technical notes"):
        st.markdown(
            f"""
- Backend entry point: `src.optimization.optimizer_service.optimize_portfolio_mode`
- Selected mode: `{config['mode']}`
- Solver message: `{result['blocks'].get('solver_message', 'N/A')}`
- Storage layer: SQLAlchemy over Postgres or local SQLite fallback
- Tracking layer: MLflow
- Multipage app: Streamlit with saved run history page
            """
        )


if __name__ == "__main__":
    main()
