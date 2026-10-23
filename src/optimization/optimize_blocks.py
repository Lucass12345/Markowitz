import numpy as np
import pandas as pd

from src.markowitz import max_sharpe_with_blocks, _align_mu_sigma, validate_block_feasibility
from src.optimization.errors import OptimizationInputError
from src.utils_portfolio import portfolio_statistics


def _aligned_metadata(tickers, meta):
    if not tickers or any(not isinstance(t, str) or not t.strip() for t in tickers):
        raise OptimizationInputError("Tickers must be a nonempty list of nonempty strings.")
    if len(tickers) != len(set(tickers)):
        raise OptimizationInputError("Tickers must be unique.")
    if not isinstance(meta, pd.DataFrame) or not meta.columns.is_unique or not {"ticker", "block"}.issubset(meta.columns):
        raise OptimizationInputError("Metadata requires unique ticker and block columns.")
    metadata = meta.copy()
    if metadata["ticker"].isna().any():
        raise OptimizationInputError("Metadata ticker labels cannot be missing.")
    metadata["ticker"] = metadata["ticker"].astype(str).str.strip()
    if metadata["ticker"].duplicated().any():
        raise OptimizationInputError("Metadata ticker labels must be unique.")
    missing = set(tickers) - set(metadata["ticker"])
    if missing:
        raise OptimizationInputError(f"Missing metadata for tickers: {sorted(missing)}.")
    metadata = metadata.set_index("ticker").loc[tickers]
    if metadata["block"].isna().any() or not metadata["block"].map(lambda b: isinstance(b, str) and bool(b.strip())).all():
        raise OptimizationInputError("Every selected ticker requires a nonempty block label.")
    return metadata


def build_blocks_from_meta(tickers: list[str], meta: pd.DataFrame, targets: dict) -> dict:
    metadata = _aligned_metadata(tickers, meta)
    return {name: list(np.flatnonzero(metadata["block"].values == name)) for name in targets}


def validate_optimizer_inputs(mu, cov, tickers, meta, targets, max_weight):
    """Preflight after model filtering, before either frontier or block solves."""
    mu, cov = _align_mu_sigma(mu, cov)
    if len(mu) != len(tickers):
        raise OptimizationInputError("Ticker count must match the mean-return vector.")
    index = getattr(mu, "index", None)
    if index is not None and not callable(index) and list(index) != list(tickers):
        raise OptimizationInputError("Mean-return labels must match ticker order.")
    if hasattr(cov, "index") and set(cov.index) != set(tickers):
        raise OptimizationInputError("Covariance labels must match tickers.")
    blocks = build_blocks_from_meta(tickers, meta, targets)
    validate_block_feasibility(len(tickers), blocks, targets, max_weight)
    return blocks


def compute_block_exposures(weights: pd.Series, meta: pd.DataFrame, targets: dict) -> pd.DataFrame:
    meta2 = _aligned_metadata(list(weights.index), meta)

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
    blocks = validate_optimizer_inputs(mu, cov, tickers, meta, targets, max_weight)
    if hasattr(cov, "loc"):
        cov = cov.loc[tickers, tickers]

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

    meta2 = _aligned_metadata(tickers, meta).reset_index()

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
