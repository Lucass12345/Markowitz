import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
import numpy as np
from scipy.optimize import minimize

from src.config import DEFAULT_CONFIG, TRADING_DAYS
from src.markowitz_classic import min_variance, max_sharpe
from src.utils_portfolio import port_stats_log, annualize_log, log_to_simple_annual


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"

CFG = DEFAULT_CONFIG
PERIODS = TRADING_DAYS
RF_ANNUAL_SIMPLE = CFG.risk_free_annual_simple


def rf_annual_simple_to_log_daily(rf_annual_simple: float, periods: int) -> float:
    return float(np.log1p(rf_annual_simple) / periods)


def portfolio_report_log(w, mu_log_daily, Sigma_log_daily, rf_annual_simple: float, periods: int):
    rf_log_daily = rf_annual_simple_to_log_daily(rf_annual_simple, periods)

    mu_d, vol_d, sh_d = port_stats_log(w, mu_log_daily, Sigma_log_daily, rf_log_daily=rf_log_daily)
    mu_a, vol_a = annualize_log(mu_d, vol_d, periods=periods)

    rf_log_annual = float(np.log1p(rf_annual_simple))
    sh_a = (mu_a - rf_log_annual) / vol_a if vol_a > 0 else np.nan

    return {
        "mu_log_daily": mu_d,
        "vol_daily": vol_d,
        "sharpe_daily": sh_d,
        "mu_log_annual": mu_a,
        "vol_annual": vol_a,
        "sharpe_annual": sh_a,
        "mu_simple_annual_equiv": log_to_simple_annual(mu_a),
        "sum_weights": float(np.sum(w)),
        "min_weight": float(np.min(w)),
        "max_weight": float(np.max(w)),
    }


# =========================
# Load data (log returns)
# =========================
returns = pd.read_csv(DATA_DIR / "returns.csv", index_col=0, parse_dates=True).sort_index()
returns = returns.dropna(how="any")

mu_log_daily = returns.mean().values
Sigma_log_daily = returns.cov().values

# =========================
# Min Variance
# =========================
w_minvar = min_variance(mu_log_daily, Sigma_log_daily)
rep_min = portfolio_report_log(w_minvar, mu_log_daily, Sigma_log_daily, RF_ANNUAL_SIMPLE, PERIODS)

print("\nMIN VAR PORTFOLIO")
for k, v in rep_min.items():
    print(f"{k:>24}: {v}")

# =========================
# Max Sharpe
# =========================
rf_log_daily = rf_annual_simple_to_log_daily(RF_ANNUAL_SIMPLE, PERIODS)
w_sharpe = max_sharpe(mu_log_daily, Sigma_log_daily, risk_free_rate=rf_log_daily)
rep_sh = portfolio_report_log(w_sharpe, mu_log_daily, Sigma_log_daily, RF_ANNUAL_SIMPLE, PERIODS)

print("\nMAX SHARPE PORTFOLIO")
for k, v in rep_sh.items():
    print(f"{k:>24}: {v}")

# =========================
# Generate classic frontier + cloud (annual log)
# =========================
np.random.seed(42)

N_PORTFOLIOS = 8000   # nube
N_FRONTIER = 60       # puntos de frontera

# Random portfolios (cloud)
weights = np.random.dirichlet(np.ones(returns.shape[1]), size=N_PORTFOLIOS)

cloud_rows = []
for w in weights:
    mu_d, vol_d, sh_d = port_stats_log(w, mu_log_daily, Sigma_log_daily, rf_log_daily=rf_log_daily)
    mu_a, vol_a = annualize_log(mu_d, vol_d, periods=PERIODS)
    cloud_rows.append({"mu_log_annual": mu_a, "vol_annual": vol_a})

cloud_df = pd.DataFrame(cloud_rows)

# Efficient frontier approximation:
# sweep target returns between min and max feasible return for long-only
mu_vec = mu_log_daily
r_min = float(mu_vec.min())
r_max = float(mu_vec.max())
targets = np.linspace(r_min, r_max, N_FRONTIER)

frontier_rows = []
for tr in targets:
    # min variance with additional constraint mu'w = tr
    cons = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w, tr=tr: float(np.dot(w, mu_vec)) - float(tr)},
    )
    bnds = tuple((0.0, 1.0) for _ in range(len(mu_vec)))
    w0 = np.ones(len(mu_vec)) / len(mu_vec)

    res = minimize(lambda w: float(w.T @ Sigma_log_daily @ w), w0, method="SLSQP", bounds=bnds, constraints=cons,
                   options={"maxiter": 4000, "ftol": 1e-12})
    if not res.success:
        continue

    w = res.x
    mu_d, vol_d, _ = port_stats_log(w, mu_log_daily, Sigma_log_daily, rf_log_daily=rf_log_daily)
    mu_a, vol_a = annualize_log(mu_d, vol_d, periods=PERIODS)
    frontier_rows.append({"mu_log_annual": mu_a, "vol_annual": vol_a})

frontier_df = pd.DataFrame(frontier_rows).sort_values("vol_annual")

# Save
cloud_path = DATA_DIR / "cloud_classic_annual.csv"
frontier_path = DATA_DIR / "frontier_classic_annual.csv"

cloud_df.to_csv(cloud_path, index=False)
frontier_df.to_csv(frontier_path, index=False)

print("\n✔ Saved:", cloud_path)
print("✔ Saved:", frontier_path)