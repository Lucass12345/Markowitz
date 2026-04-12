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

# risk-free en log diario (coherente con log returns)
rf_log_daily = rf_annual_simple_to_log_daily(RF_ANNUAL_SIMPLE, PERIODS)

# =========================
# Min Variance (classic)
# =========================
w_minvar = min_variance(mu_log_daily, Sigma_log_daily)
rep_min = portfolio_report_log(w_minvar, mu_log_daily, Sigma_log_daily, RF_ANNUAL_SIMPLE, PERIODS)

print("\nMIN VAR PORTFOLIO")
for k, v in rep_min.items():
    print(f"{k:>24}: {v}")

# =========================
# Max Sharpe (classic)
# =========================
w_sharpe = max_sharpe(mu_log_daily, Sigma_log_daily, risk_free_rate=rf_log_daily)
rep_sh = portfolio_report_log(w_sharpe, mu_log_daily, Sigma_log_daily, RF_ANNUAL_SIMPLE, PERIODS)

print("\nMAX SHARPE PORTFOLIO")
for k, v in rep_sh.items():
    print(f"{k:>24}: {v}")

# =========================
# Save classic key points (for plotting on frontier)
# =========================
classic_points = pd.DataFrame([
    {
        "name": "Classic MinVar",
        "mu_log_annual": rep_min["mu_log_annual"],
        "vol_annual": rep_min["vol_annual"],
        "sharpe_annual": rep_min["sharpe_annual"],
    },
    {
        "name": "Classic MaxSharpe",
        "mu_log_annual": rep_sh["mu_log_annual"],
        "vol_annual": rep_sh["vol_annual"],
        "sharpe_annual": rep_sh["sharpe_annual"],
    },
])

classic_points_path = DATA_DIR / "classic_points_annual.csv"
classic_points.to_csv(classic_points_path, index=False)
print("✔ Saved:", classic_points_path)

# =========================
# Generate classic cloud + frontier (annual log)
# =========================
np.random.seed(42)

N_PORTFOLIOS = 8000  # nube
N_FRONTIER = 80      # puntos de frontera (más denso para curva más suave)

# ----- Cloud: random portfolios (long-only)
weights = np.random.dirichlet(np.ones(returns.shape[1]), size=N_PORTFOLIOS)

cloud_rows = []
for w in weights:
    mu_d, vol_d, _ = port_stats_log(w, mu_log_daily, Sigma_log_daily, rf_log_daily=rf_log_daily)
    mu_a, vol_a = annualize_log(mu_d, vol_d, periods=PERIODS)
    cloud_rows.append({"mu_log_annual": mu_a, "vol_annual": vol_a})

cloud_df = pd.DataFrame(cloud_rows)

# ----- Frontier: robust construction
mu_vec = mu_log_daily
Sigma = Sigma_log_daily
n = len(mu_vec)

bounds = tuple((0.0, 1.0) for _ in range(n))
w0 = np.ones(n) / n
cons_sum = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

# Feasible return range (long-only): min/max of mu'w under sum(w)=1, w>=0
def obj_min_ret(w):
    return float(w @ mu_vec)

def obj_max_ret(w):
    return float(-(w @ mu_vec))

res_rmin = minimize(obj_min_ret, w0, method="SLSQP", bounds=bounds, constraints=cons_sum,
                    options={"maxiter": 4000, "ftol": 1e-12})
res_rmax = minimize(obj_max_ret, w0, method="SLSQP", bounds=bounds, constraints=cons_sum,
                    options={"maxiter": 4000, "ftol": 1e-12})

r_min = float(res_rmin.x @ mu_vec)
r_max = float(res_rmax.x @ mu_vec)

# Grid dentro del rango factible (evita extremos)
targets = np.linspace(r_min + 0.01 * (r_max - r_min), r_max - 0.01 * (r_max - r_min), N_FRONTIER)

frontier_rows = []
for tr in targets:
    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w, tr=tr: float(w @ mu_vec) - float(tr)},  # mu'w >= tr
    ]

    res = minimize(lambda w: float(w.T @ Sigma @ w), w0, method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"maxiter": 4000, "ftol": 1e-12})
    if not res.success:
        continue

    w = res.x
    mu_d, vol_d, _ = port_stats_log(w, mu_log_daily, Sigma_log_daily, rf_log_daily=rf_log_daily)
    mu_a, vol_a = annualize_log(mu_d, vol_d, periods=PERIODS)

    if np.isfinite(mu_a) and np.isfinite(vol_a) and vol_a > 0:
        frontier_rows.append({"mu_log_annual": mu_a, "vol_annual": vol_a})

frontier_df = pd.DataFrame(frontier_rows)

# Ordenar por vol y quitar duplicados de vol (suaviza la línea)
frontier_df = frontier_df.sort_values("vol_annual").drop_duplicates("vol_annual").reset_index(drop=True)

# =========================
# Save outputs used by run_compare.py
# =========================
cloud_path = DATA_DIR / "cloud_classic_annual.csv"
frontier_path = DATA_DIR / "frontier_classic_annual.csv"

cloud_df.to_csv(cloud_path, index=False)
frontier_df.to_csv(frontier_path, index=False)

print("\n✔ Saved:", cloud_path)
print("✔ Saved:", frontier_path)