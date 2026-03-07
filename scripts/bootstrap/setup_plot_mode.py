from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]

files = {
    "src/visualization/plot_mode_result.py": r'''
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "figures"


def plot_mode_result(
    result: dict,
    save: bool = True,
    show: bool = True,
    figsize: tuple = (10, 7),
):
    model_type = result["model_type"]

    frontier = result["frontier"]
    minvar_stats = result["minvar"]["stats"]
    maxsharpe_stats = result["maxsharpe"]["stats"]
    blocks_stats = result["blocks"]["stats"]

    fig, ax = plt.subplots(figsize=figsize)

    # Frontera eficiente
    ax.plot(
        frontier["vol_annual"],
        frontier["return_annual"],
        linewidth=2,
        label=f"Frontier - {model_type}"
    )

    # Min Variance
    ax.scatter(
        minvar_stats["vol_annual"],
        minvar_stats["return_annual"],
        s=140,
        label="Min Variance"
    )

    # Max Sharpe
    ax.scatter(
        maxsharpe_stats["vol_annual"],
        maxsharpe_stats["return_annual"],
        s=140,
        label="Max Sharpe"
    )

    # Max Sharpe + bloques
    ax.scatter(
        blocks_stats["vol_annual"],
        blocks_stats["return_annual"],
        s=180,
        label="Max Sharpe + Blocks"
    )

    ax.set_title(f"Efficient Frontier - {model_type}")
    ax.set_xlabel("Annual Volatility")
    ax.set_ylabel("Annual Return")
    ax.grid(True)
    ax.legend()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if save:
        fig.savefig(OUT_DIR / f"frontier_{model_type}.png", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def save_mode_outputs(result: dict):
    """
    Guarda frontier, pesos y exposures para el modo elegido.
    """
    model_type = result["model_type"]

    base_dir = ROOT / "outputs"
    weights_dir = base_dir / "weights"
    metrics_dir = base_dir / "metrics"

    weights_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # frontera
    result["frontier"].to_csv(
        base_dir / f"frontier_{model_type}.csv",
        index=False
    )

    # pesos
    result["minvar"]["weights_df"].to_csv(
        weights_dir / f"weights_minvar_{model_type}.csv",
        index=False
    )

    result["maxsharpe"]["weights_df"].to_csv(
        weights_dir / f"weights_maxsharpe_{model_type}.csv",
        index=False
    )

    result["blocks"]["weights_df"].to_csv(
        weights_dir / f"weights_blocks_{model_type}.csv",
        index=False
    )

    result["blocks"]["exposure_by_block"].to_csv(
        weights_dir / f"block_exposure_{model_type}.csv",
        index=False
    )

    # métricas
    metrics_df = pd.DataFrame([
        {"portfolio": "minvar", **result["minvar"]["stats"]},
        {"portfolio": "maxsharpe", **result["maxsharpe"]["stats"]},
        {"portfolio": "blocks", **result["blocks"]["stats"]},
    ])

    metrics_df.to_csv(
        metrics_dir / f"metrics_{model_type}.csv",
        index=False
    )
''',

    "scripts/run_plot_mode_classic.py": r'''
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import DEFAULT_CONFIG
from src.optimization.optimizer_service import optimize_portfolio_mode
from src.visualization.plot_mode_result import plot_mode_result, save_mode_outputs


if __name__ == "__main__":
    result = optimize_portfolio_mode(
        model_type="classic",
        targets=DEFAULT_CONFIG.targets,
        max_weight=DEFAULT_CONFIG.max_weight_blocks,
        rf_annual_simple=DEFAULT_CONFIG.risk_free_annual_simple,
    )

    save_mode_outputs(result)
    plot_mode_result(result, save=True, show=True)

    print("\nSaved outputs for classic mode.")
''',

    "scripts/run_plot_mode_ai_returns.py": r'''
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import DEFAULT_CONFIG
from src.optimization.optimizer_service import optimize_portfolio_mode
from src.visualization.plot_mode_result import plot_mode_result, save_mode_outputs


if __name__ == "__main__":
    result = optimize_portfolio_mode(
        model_type="ai_returns",
        targets=DEFAULT_CONFIG.targets,
        max_weight=DEFAULT_CONFIG.max_weight_blocks,
        rf_annual_simple=DEFAULT_CONFIG.risk_free_annual_simple,
    )

    save_mode_outputs(result)
    plot_mode_result(result, save=True, show=True)

    print("\nSaved outputs for ai_returns mode.")
''',

    "scripts/run_plot_mode_ai_volatility.py": r'''
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import DEFAULT_CONFIG
from src.optimization.optimizer_service import optimize_portfolio_mode
from src.visualization.plot_mode_result import plot_mode_result, save_mode_outputs


if __name__ == "__main__":
    result = optimize_portfolio_mode(
        model_type="ai_volatility",
        targets=DEFAULT_CONFIG.targets,
        max_weight=DEFAULT_CONFIG.max_weight_blocks,
        rf_annual_simple=DEFAULT_CONFIG.risk_free_annual_simple,
    )

    save_mode_outputs(result)
    plot_mode_result(result, save=True, show=True)

    print("\nSaved outputs for ai_volatility mode.")
''',
}


def main():
    for relative_path, content in files.items():
        path = ROOT / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
        print(f"created: {relative_path}")

    print("\nAll plotting files created successfully.")


if __name__ == "__main__":
    main()