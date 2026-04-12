import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import DEFAULT_CONFIG
from src.optimization.optimizer_service import optimize_portfolio_mode

if __name__ == "__main__":
    result = optimize_portfolio_mode(
        model_type="ai_returns_lstm",
        targets=DEFAULT_CONFIG.targets,
        max_weight=DEFAULT_CONFIG.max_weight_blocks,
        rf_annual_simple=DEFAULT_CONFIG.risk_free_annual_simple,
    )

    print("\n=== AI RETURNS LSTM ===")
    print("MinVar:", result["minvar"]["stats"])
    print("MaxSharpe:", result["maxsharpe"]["stats"])
    print("Blocks:", result["blocks"]["stats"])
    print("\nTop weights blocks:")
    print(result["blocks"]["weights_df"].head(10))