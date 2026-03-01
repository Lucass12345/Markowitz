import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.markowitz import max_sharpe_with_blocks, min_variance_with_blocks
from src.markowitz import rf_simple_annual_to_log_daily
from src.utils_portfolio import port_stats_log, annualize_log


BLOCKS_ORDER = ["AR", "US", "FUNDS_EQUITY", "FUNDS_BONDS", "GOLD"]
TARGETS = {"AR": 0.20, "US": 0.50, "FUNDS_EQUITY": 0.10, "FUNDS_BONDS": 0.10, "GOLD": 0.10}

RISK_FREE_ANNUAL_SIMPLE = 0.0
MAX_WEIGHT_BLOCKS = 0.15  # <- tu restricción


def build_blocks(tickers, meta):
    meta2 = meta.set_index("ticker").loc[tickers]
    blocks = {}
    for b in BLOCKS_ORDER:
        idx = list(np.where(meta2["block"].values == b)[0])
        if len(idx) == 0:
            raise ValueError(f"Bloque {b} quedó vacío. Revisá assets_metadata.csv")
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

    # --- leer frontera y nube clásicas ya generadas por run_classic.py (solo plot)
    frontier = pd.read_csv(DATA_DIR / "frontier_classic_annual.csv")
    cloud = pd.read_csv(DATA_DIR / "cloud_classic_annual.csv")

    # --- resolver puntos bloqueados
    blocks = build_blocks(mean_log.index, meta)

    res_sh_b = max_sharpe_with_blocks(
        mean_log, cov_log, blocks, TARGETS,
        risk_free_rate=rf_log_daily,
        max_weight=MAX_WEIGHT_BLOCKS
    )
    if not res_sh_b.success:
        raise RuntimeError("Blocks MaxSharpe no convergió: " + str(res_sh_b.message))
    w_sh_b = res_sh_b.x

    res_mv_b = min_variance_with_blocks(cov_log, blocks, TARGETS, max_weight=MAX_WEIGHT_BLOCKS)
    if not res_mv_b.success:
        raise RuntimeError("Blocks MinVar no convergió: " + str(res_mv_b.message))
    w_mv_b = res_mv_b.x

    # stats diarios (log) -> anualizados (log)
    sh_mu_d, sh_vol_d, sh_sh_d = port_stats_log(w_sh_b, mean_log, cov_log, rf_log_daily)
    mv_mu_d, mv_vol_d, mv_sh_d = port_stats_log(w_mv_b, mean_log, cov_log, rf_log_daily)

    sh_mu_a, sh_vol_a = annualize_log(sh_mu_d, sh_vol_d)
    mv_mu_a, mv_vol_a = annualize_log(mv_mu_d, mv_vol_d)

    # Sharpe anual consistente en escala log:
    # (mu_ann - rf_ann) / vol_ann, con rf_ann = rf_daily*252
    rf_log_annual = rf_log_daily * 252.0
    sh_sh_a = (sh_mu_a - rf_log_annual) / sh_vol_a if sh_vol_a > 0 else np.nan
    mv_sh_a = (mv_mu_a - rf_log_annual) / mv_vol_a if mv_vol_a > 0 else np.nan

    # --- plot compare
    plt.figure(figsize=(12, 7))
    plt.scatter(cloud["vol_annual"], cloud["mu_log_annual"], alpha=0.20, label="Classic random portfolios")
    plt.plot(frontier["vol_annual"], frontier["mu_log_annual"], linewidth=3, label="Classic efficient frontier")

    plt.scatter([sh_vol_a], [sh_mu_a], marker="X", s=260,
                label=f"Blocks MaxSharpe (cap={MAX_WEIGHT_BLOCKS}) sh={sh_sh_a:.3f}")
    plt.scatter([mv_vol_a], [mv_mu_a], marker="D", s=220,
                label=f"Blocks MinVar (cap={MAX_WEIGHT_BLOCKS}) sh={mv_sh_a:.3f}")

    plt.title("Compare: Classic Frontier + Cloud vs Block-Constrained Portfolios (LOG annualized)")
    plt.xlabel("Volatility (annual)")
    plt.ylabel("Expected Return (mu_log annual)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    out = DATA_DIR / "compare_classic_vs_blocks.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    print("✔ Saved:", out)

    print("\n--- BLOCKS POINTS (annual log) ---")
    print(f"Blocks MaxSharpe: mu_log={sh_mu_a:.6f} vol={sh_vol_a:.6f} sharpe={sh_sh_a:.6f}")
    print(f"Blocks MinVar:    mu_log={mv_mu_a:.6f} vol={mv_vol_a:.6f} sharpe={mv_sh_a:.6f}")


if __name__ == "__main__":
    main()