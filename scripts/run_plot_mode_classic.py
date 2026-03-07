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
