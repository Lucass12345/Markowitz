import numpy as np
import pandas as pd

from src.markowitz import max_sharpe_with_blocks
from src.utils_portfolio import portfolio_statistics


def build_blocks_from_meta(tickers: list[str], meta: pd.DataFrame, targets: dict) -> dict:
    meta2 = meta.copy()
    meta2["ticker"] = meta2["ticker"].astype(str).str.strip()
    meta2 = meta2.set_index("ticker").loc[tickers]

    blocks = {}
    for block_name in targets.keys():
        idx = np.where(meta2["block"].values == block_name)[0]
        blocks[block_name] = list(idx)

    return blocks


def compute_block_exposures(weights: pd.Series, meta: pd.DataFrame, targets: dict) -> pd.DataFrame:
    meta2 = meta.copy()
    meta2["ticker"] = meta2["ticker"].astype(str).str.strip()
    meta2 = meta2.set_index("ticker").loc[weights.index]

    rows = []
    for block_name, target in targets.items():
        exposure = float(weights.loc[meta2.index[meta2["block"] == block_name]].sum())
        rows.append({
            "block": block_name,
            "target": float(target),
            "achieved": exposure,
            "diff": exposure - float(target),
        })

    return pd.DataFrame(rows)


def run_blocks_optimizer(
    mu,
    cov,
    tickers: list[str],
    meta: pd.DataFrame,
    targets: dict,
    max_weight: float = 0.15,
    rf_log_daily: float = 0.0,
    periods: int = 252,
) -> dict:
    blocks = build_blocks_from_meta(tickers, meta, targets)

    res = max_sharpe_with_blocks(
        mean_returns=mu,
        cov_matrix=cov,
        blocks=blocks,
        block_targets=targets,
        risk_free_rate=rf_log_daily,
        max_weight=max_weight,
    )

    if not res.success:
        raise RuntimeError(f"max_sharpe_with_blocks failed: {res.message}")

    w = pd.Series(res.x, index=tickers, name="weight")
    w_df = w.sort_values(ascending=False).reset_index()
    w_df.columns = ["ticker", "weight"]

    meta2 = meta.copy()
    meta2["ticker"] = meta2["ticker"].astype(str).str.strip()
    meta2 = meta2.set_index("ticker").loc[tickers].reset_index()

    w_df = w_df.merge(meta2, on="ticker", how="left")

    exposures_df = compute_block_exposures(w, meta, targets)

    stats = portfolio_statistics(w.values, mu, cov, rf_log_daily, periods)

    return {
        "weights_array": w.values,
        "weights_series": w,
        "weights_df": w_df,
        "stats": stats,
        "exposure_by_block": exposures_df,
        "solver_message": res.message,
    }
