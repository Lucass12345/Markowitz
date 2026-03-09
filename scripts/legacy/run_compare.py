import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.markowitz import max_sharpe_with_blocks
from src.markowitz import rf_simple_annual_to_log_daily
from src.utils_portfolio import port_stats_log, annualize_log

from src.config import DEFAULT_CONFIG, TRADING_DAYS
CFG = DEFAULT_CONFIG

BLOCKS_ORDER = CFG.blocks_order
TARGETS = CFG.targets
RISK_FREE_ANNUAL_SIMPLE = CFG.risk_free_annual_simple
MAX_WEIGHT_BLOCKS = CFG.max_weight_blocks


def build_blocks(tickers, meta, targets):
    meta2 = meta.set_index("ticker").loc[tickers]
    blocks = {}
    for b in targets.keys():  # usa los targets como "verdad"
        idx = list(np.where(meta2["block"].values == b)[0])
        # Solo es error si target > 0 y el bloque está vacío
        if float(targets[b]) > 0.0 and len(idx) == 0:
            raise ValueError(f"Bloque {b} quedó vacío pero target>0. Revisá assets_metadata.csv")
        blocks[b] = idx
    return blocks


def main():
    DATA_DIR = ROOT / "Data"

    returns = pd.read_csv(DATA_DIR / "returns.csv", index_col=0, parse_dates=True).sort_index()
    meta = pd.read_csv(DATA_DIR / "assets_metadata.csv")

    allowed = meta["ticker"].astype(str).str.strip().tolist()
    returns = returns.loc[:, returns.columns.isin(allowed)]
    if returns.isna().any().any():
        raise ValueError("returns.csv tiene NaNs.")

    mean_log = returns.mean()
    cov_log = returns.cov()

    # rf en log diario (consistente con mean_log)
    rf_log_daily = rf_simple_annual_to_log_daily(RISK_FREE_ANNUAL_SIMPLE)
    rf_log_annual = rf_log_daily * TRADING_DAYS

    # --- leer outputs classic ya generados por run_classic.py
    frontier = pd.read_csv(DATA_DIR / "frontier_classic_annual.csv")
    classic_pts = pd.read_csv(DATA_DIR / "classic_points_annual.csv")

    # --- resolver punto bloqueado (SOLO MaxSharpe)
    blocks = build_blocks(mean_log.index, meta, TARGETS)

    res_sh_b = max_sharpe_with_blocks(
        mean_log, cov_log, blocks, TARGETS,
        risk_free_rate=rf_log_daily,
        max_weight=MAX_WEIGHT_BLOCKS
    )
    if not res_sh_b.success:
        raise RuntimeError("Blocks MaxSharpe no convergió: " + str(res_sh_b.message))
    w_sh_b = res_sh_b.x

    # stats diarios (log) -> anualizados (log)  (solo MaxSharpe blocks)
    sh_mu_d, sh_vol_d, _ = port_stats_log(w_sh_b, mean_log, cov_log, rf_log_daily)
    sh_mu_a, sh_vol_a = annualize_log(sh_mu_d, sh_vol_d)
    sh_sh_a = (sh_mu_a - rf_log_annual) / sh_vol_a if sh_vol_a > 0 else np.nan

    # --- extraer puntos classic: MinVar y MaxSharpe
    minvar_row = classic_pts[classic_pts["name"] == "Classic MinVar"].iloc[0]
    maxsh_row = classic_pts[classic_pts["name"] == "Classic MaxSharpe"].iloc[0]

    # --- plot compare
    plt.figure(figsize=(12, 7))
    plt.plot(frontier["vol_annual"], frontier["mu_log_annual"], linewidth=3, label="Classic efficient frontier")

    # Classic key points
    plt.scatter([minvar_row["vol_annual"]], [minvar_row["mu_log_annual"]],
                marker="D", s=220, label=f'Classic MinVar sh={minvar_row["sharpe_annual"]:.3f}')
    plt.scatter([maxsh_row["vol_annual"]], [maxsh_row["mu_log_annual"]],
                marker="X", s=260, label=f'Classic MaxSharpe sh={maxsh_row["sharpe_annual"]:.3f}')

    # Blocks MaxSharpe point
    plt.scatter([sh_vol_a], [sh_mu_a], marker="X", s=260,
                label=f"Blocks MaxSharpe (cap={MAX_WEIGHT_BLOCKS}) sh={sh_sh_a:.3f}")

    plt.title("Compare: Classic Frontier + Key Points vs Block-Constrained MaxSharpe (LOG annualized)")
    plt.xlabel("Volatility (annual)")
    plt.ylabel("Expected Return (mu_log annual)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    out = DATA_DIR / "compare_classic_vs_blocks.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    print("✔ Saved:", out)

    print("\n--- POINTS (annual log) ---")
    print(f'Classic MinVar:    mu_log={minvar_row["mu_log_annual"]:.6f} vol={minvar_row["vol_annual"]:.6f} sharpe={minvar_row["sharpe_annual"]:.6f}')
    print(f'Classic MaxSharpe: mu_log={maxsh_row["mu_log_annual"]:.6f} vol={maxsh_row["vol_annual"]:.6f} sharpe={maxsh_row["sharpe_annual"]:.6f}')
    print(f"Blocks MaxSharpe:  mu_log={sh_mu_a:.6f} vol={sh_vol_a:.6f} sharpe={sh_sh_a:.6f}")


if __name__ == "__main__":
    main()