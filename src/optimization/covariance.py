"""Observed-calendar covariance; no returns are imputed or forward-filled."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.optimization.errors import OptimizationInputError


@dataclass(frozen=True)
class RiskConfig:
    min_history: int = 252
    min_overlap: int = 126
    eigenvalue_floor: float = 1e-8
    max_psd_adjustment: float = 1.0

    def __post_init__(self):
        if self.min_history < 2 or self.min_overlap < 2:
            raise OptimizationInputError("min_history and min_overlap must be at least 2")
        if not np.isfinite(self.eigenvalue_floor) or not np.isfinite(self.max_psd_adjustment) or not 0 < self.eigenvalue_floor < 1 or not 0 <= self.max_psd_adjustment:
            raise OptimizationInputError("Invalid covariance regularization controls")


def regularize_covariance(cov, config=None):
    config = config or RiskConfig()
    array = np.asarray(cov, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or not np.isfinite(array).all():
        raise OptimizationInputError("Covariance must be a finite square matrix")
    symmetric = (array + array.T) / 2
    eig, vectors = np.linalg.eigh(symmetric)
    floor = max(float(np.max(np.abs(eig))), np.finfo(float).eps) * config.eigenvalue_floor
    repaired = (vectors * np.maximum(eig, floor)) @ vectors.T
    repaired = (repaired + repaired.T) / 2
    relative = float(np.linalg.norm(repaired - symmetric) / max(np.linalg.norm(symmetric), np.finfo(float).eps))
    if relative > config.max_psd_adjustment + 1e-12:
        raise OptimizationInputError(f"Covariance PSD adjustment {relative:.6f} exceeds allowed {config.max_psd_adjustment}")
    after = np.linalg.eigvalsh(repaired)
    diagnostics = {"min_eigenvalue_before": float(eig.min()), "max_eigenvalue_before": float(eig.max()),
                   "min_eigenvalue_after": float(after.min()), "max_eigenvalue_after": float(after.max()),
                   "condition_number": float(after.max() / after.min()),
                   "psd_relative_adjustment": relative, "eigenvalue_floor": floor,
                   "regularized": bool(np.any(eig < floor))}
    result = pd.DataFrame(repaired, index=cov.index, columns=cov.columns) if isinstance(cov, pd.DataFrame) else repaired
    return result, diagnostics


def estimate_historical_risk(returns, config=None):
    config = config or RiskConfig()
    data = returns.copy().sort_index()
    data.columns = data.columns.astype(str).str.strip()
    if data.index.has_duplicates or data.columns.has_duplicates or any(not t for t in data.columns):
        raise OptimizationInputError("Duplicate/inconsistent dates or tickers in returns")
    data = data.astype(float)
    if np.isinf(data.to_numpy()).any():
        raise OptimizationInputError("Infinite observed return in dataset")
    counts = data.notna().sum()
    overlaps = data.notna().astype(int).T @ data.notna().astype(int)
    excluded, tickers = {}, []
    for ticker in data.columns:
        if counts[ticker] < config.min_history:
            excluded[ticker] = f"history {int(counts[ticker])} < {config.min_history}"
        elif not np.isfinite(data[ticker].var()) or data[ticker].var() <= 0:
            excluded[ticker] = "non_positive_observed_variance"
        else:
            tickers.append(ticker)
    while len(tickers) > 1:
        bad = [(a, b) for i, a in enumerate(tickers) for b in tickers[i+1:]
               if overlaps.loc[a, b] < config.min_overlap]
        if not bad:
            break
        a, b = bad[0]
        removed = min((a, b), key=lambda t: (counts[t], t))
        excluded[removed] = f"overlap {a}/{b}: {int(overlaps.loc[a,b])} < {config.min_overlap}"
        tickers.remove(removed)
    if not tickers:
        raise OptimizationInputError(f"No eligible assets after history/variance checks: {excluded}")
    eligible = data[tickers].dropna(how="all")
    cov, diagnostics = regularize_covariance(eligible.cov(min_periods=2), config)
    diagnostics.update({"calendar_policy": "pairwise_observed_no_imputation", "estimator": "historical",
                        "min_history": config.min_history, "min_overlap": config.min_overlap,
                        "excluded_assets": excluded, "eligible_assets": tickers,
                        "asset_observations": {t: int(counts[t]) for t in data},
                        "pair_observations": {a: {b: int(overlaps.loc[a,b]) for b in tickers} for a in tickers},
                        "available_dates": len(data), "global_complete_dates": int(eligible.notna().all(axis=1).sum())})
    return eligible, cov, diagnostics
