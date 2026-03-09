import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd

from src.markowitz import max_sharpe_with_blocks
# from src.markowitz import min_variance_with_blocks  # <- (comentado) NO usamos MinVar

from src.config import DEFAULT_CONFIG, TRADING_DAYS
CFG = DEFAULT_CONFIG

BLOCKS_ORDER = CFG.blocks_order
TARGETS = CFG.targets

RISK_FREE_RATE = CFG.risk_free_annual_simple
MAX_WEIGHT_PER_ASSET = CFG.max_weight_blocks


def build_blocks(tickers, meta, targets):
    meta2 = meta.set_index("ticker").loc[tickers]
    blocks = {}
    for b in targets.keys():
        idx = list(np.where(meta2["block"].values == b)[0])
        if float(targets[b]) > 0.0 and len(idx) == 0:
            raise ValueError(f"Bloque {b} quedó vacío pero target>0. Revisá assets_metadata.csv")
        blocks[b] = idx
    return blocks


def block_stats(w, meta2, block_name):
    wb = w[meta2["block"] == block_name]
    total = float(wb.sum())
    n = int(len(wb))
    if total <= 0 or n == 0:
        return np.nan, np.nan, np.nan, 0, n

    wn = wb / total
    max_in_block = float(wn.max())
    hhi = float((wn ** 2).sum())
    eff_n = float(1.0 / hhi) if hhi > 0 else np.nan
    n_gt_5 = int((wn > 0.05).sum())
    return max_in_block, hhi, eff_n, n_gt_5, n


def save_detailed_weights(data_dir, fname, w_sorted, meta2):
    out = pd.DataFrame({
        "ticker": w_sorted.index,
        "weight": w_sorted.values,
        "block": meta2.loc[w_sorted.index, "block"].values,
        "asset_class": meta2.loc[w_sorted.index, "asset_class"].values,
    }).sort_values("weight", ascending=False)

    out.to_csv(data_dir / fname, index=False)
    print(f"✔ Saved: Data/{fname}")


def print_exposures(w_series, meta2, targets):
    for b in BLOCKS_ORDER:
        exp_b = float(w_series[meta2["block"] == b].sum())
        print(f"Exposure {b}: {exp_b:.6f} (target {targets[b]:.6f})")


def print_block_stats(w_series, meta2):
    print("\n--- Block concentration stats (normalized within block) ---")
    for b in BLOCKS_ORDER:
        max_w, hhi, eff_n, n_gt_5, n = block_stats(w_series, meta2, b)
        print(f"{b}: max_in_block={max_w:.3f} | HHI={hhi:.3f} | effN={eff_n:.2f} | >5%={n_gt_5}/{n}")


def rf_annual_simple_to_log_daily(rf_annual_simple, periods=252):
    return float(np.log1p(rf_annual_simple) / periods)


def port_stats_log(w_aligned, mean_log_returns, cov_log_returns, rf_log_daily=0.0, periods=252):
    """
    w_aligned: np.array en el MISMO ORDEN que mean_log_returns.index
    """
    w = np.asarray(w_aligned, dtype=float)

    mu_log_d = float(w @ mean_log_returns.values)
    vol_d = float(np.sqrt(w @ cov_log_returns.values @ w))

    sharpe_d = (mu_log_d - float(rf_log_daily)) / vol_d if vol_d > 0 else np.nan

    mu_log_a = mu_log_d * periods
    vol_a = vol_d * np.sqrt(periods)
    sharpe_a = sharpe_d * np.sqrt(periods) if np.isfinite(sharpe_d) else np.nan

    ret_a_simple = float(np.expm1(mu_log_a))
    return mu_log_d, vol_d, sharpe_d, mu_log_a, vol_a, sharpe_a, ret_a_simple


def validate_targets(targets):
    missing = [b for b in BLOCKS_ORDER if b not in targets]
    if missing:
        raise ValueError(f"TARGETS no incluye bloques: {missing}")
    s = float(sum(targets.values()))
    if abs(s - 1.0) > 1e-9:
        raise ValueError(f"Los targets deben sumar 1.0 y suman {s}")


def main():
    validate_targets(TARGETS)

    data_dir = ROOT / "Data"
    returns = pd.read_csv(data_dir / "returns.csv", index_col=0, parse_dates=True).sort_index()
    meta = pd.read_csv(data_dir / "assets_metadata.csv")

    allowed = meta["ticker"].astype(str).str.strip().tolist()
    returns = returns.loc[:, returns.columns.isin(allowed)]

    if returns.isna().any().any():
        raise ValueError("returns.csv tiene NaNs. Limpialo o imputá antes de optimizar.")

    mean_returns = returns.mean()   # LOG diarios
    cov_matrix = returns.cov()      # cov de LOG diarios

    blocks = build_blocks(mean_returns.index, meta, TARGETS)
    meta2 = meta.set_index("ticker").loc[mean_returns.index]

    rf_log_daily = rf_annual_simple_to_log_daily(RISK_FREE_RATE, periods=TRADING_DAYS)

    print("\n==============================")
    print("MODEL PARAMETERS")
    print("==============================")
    print("Targets:", TARGETS)
    print("Risk-free (annual simple):", RISK_FREE_RATE)
    print("Risk-free (log daily used in optimizer):", f"{rf_log_daily:.10f}")
    print("Trading days:", TRADING_DAYS)
    print("Max weight per asset:", MAX_WEIGHT_PER_ASSET)

    # ==============================
    # MAX SHARPE (BLOCKS)
    # ==============================
    print("\n==============================")
    print("MAX SHARPE WITH BLOCK TARGETS")
    print("==============================")

    res_sh = max_sharpe_with_blocks(
        mean_returns, cov_matrix, blocks, TARGETS,
        risk_free_rate=rf_log_daily,
        max_weight=MAX_WEIGHT_PER_ASSET
    )

    if not res_sh.success:
        print("❌ Max Sharpe no convergió:", res_sh.message)
        return

    # ✅ crudo/alineado (para cálculos)
    w_sh_raw = pd.Series(res_sh.x, index=mean_returns.index)
    # ✅ ordenado (para output)
    w_sh = w_sh_raw.sort_values(ascending=False)

    print("✔ Converged:", res_sh.success)
    print("Message:", res_sh.message)
    print("Sum weights:", float(w_sh_raw.sum()))
    print_exposures(w_sh_raw, meta2, TARGETS)

    save_detailed_weights(data_dir, "weights_maxsharpe_blocks_detailed.csv", w_sh, meta2)

    print("\nTop 10 weights (Max Sharpe):")
    print(w_sh.head(10))

    print_block_stats(w_sh_raw, meta2)

    mu_d, v_d, s_d, mu_a, v_a, s_a, r_a_simple = port_stats_log(
        w_sh_raw.values, mean_returns, cov_matrix,
        rf_log_daily=rf_log_daily,
        periods=TRADING_DAYS
    )
    print(f"\nMaxSharpe stats (LOG input, aligned):")
    print(f"Daily:  mu_log={mu_d:.6f} vol={v_d:.6f} sharpe={s_d:.6f}")
    print(f"Annual: mu_log={mu_a:.3f} vol={v_a:.3f} sharpe={s_a:.3f} | simple_equiv_ret={r_a_simple:.3f}")

    # ==============================
    # MIN VAR (BLOCKS)  <-- COMENTADO
    # ==============================
    """
    print("\n==============================")
    print("MIN VAR WITH BLOCK TARGETS")
    print("==============================")

    res_mv = min_variance_with_blocks(
        cov_matrix, blocks, TARGETS,
        max_weight=MAX_WEIGHT_PER_ASSET
    )

    if not res_mv.success:
        print("❌ Min Var no convergió:", res_mv.message)
    else:
        w_mv_raw = pd.Series(res_mv.x, index=mean_returns.index)
        w_mv = w_mv_raw.sort_values(ascending=False)

        print("✔ Converged:", res_mv.success)
        print("Message:", res_mv.message)
        print("Sum weights:", float(w_mv_raw.sum()))
        print_exposures(w_mv_raw, meta2, TARGETS)

        save_detailed_weights(data_dir, "weights_minvar_blocks_detailed.csv", w_mv, meta2)

        print("\nTop 10 weights (Min Var):")
        print(w_mv.head(10))

        print_block_stats(w_mv_raw, meta2)

        mu_d, v_d, s_d, mu_a, v_a, s_a, r_a_simple = port_stats_log(
            w_mv_raw.values, mean_returns, cov_matrix,
            rf_log_daily=rf_log_daily,
            periods=TRADING_DAYS
        )
        print(f"\nMinVar stats (LOG input, aligned):")
        print(f"Daily:  mu_log={mu_d:.6f} vol={v_d:.6f} sharpe={s_d:.6f}")
        print(f"Annual: mu_log={mu_a:.3f} vol={v_a:.3f} sharpe={s_a:.3f} | simple_equiv_ret={r_a_simple:.3f}")
    """


if __name__ == "__main__":
    main()