from pathlib import Path
import pandas as pd
import numpy as np

from src.markowitz_classic import min_variance, max_sharpe
from src.utils_portfolio import portfolio_report_log_aligned

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"

returns = pd.read_csv(DATA_DIR / "returns.csv", index_col=0, parse_dates=True)

mean_returns = returns.mean().values
cov_matrix = returns.cov().values

# =========================
# Min Variance
# =========================
w_minvar = min_variance(mean_returns, cov_matrix)

rep_min = portfolio_report_log_aligned(
    w_minvar,
    mean_returns,
    cov_matrix,
    rf_annual_simple=0.0,
    periods=252
)

print("\nMIN VAR PORTFOLIO")
print(rep_min)

# =========================
# Max Sharpe
# =========================
w_sharpe = max_sharpe(mean_returns, cov_matrix, risk_free_rate=0.0)

rep_sh = portfolio_report_log_aligned(
    w_sharpe,
    mean_returns,
    cov_matrix,
    rf_annual_simple=0.0,
    periods=252
)

print("\nMAX SHARPE PORTFOLIO")
print(rep_sh)