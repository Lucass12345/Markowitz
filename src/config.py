# src/config.py
from dataclasses import dataclass
from typing import Dict, List

TRADING_DAYS = 252

@dataclass(frozen=True)
class PortfolioConfig:
    blocks_order: List[str]
    targets: Dict[str, float]
    risk_free_annual_simple: float
    max_weight_blocks: float

DEFAULT_CONFIG = PortfolioConfig(
    blocks_order=["AR", "US", "FUNDS_EQUITY", "FUNDS_BONDS", "GOLD"],
    targets={"AR": 0.20, "US": 0.50, "FUNDS_EQUITY": 0.10, "FUNDS_BONDS": 0.10, "GOLD": 0.10},
    risk_free_annual_simple=0.0,
    max_weight_blocks=0.15,
)


def block_configuration(metadata):
    """Available blocks from the dataset; retain legacy targets for the legacy universe."""
    from src.optimization.errors import OptimizationInputError
    if not {"ticker", "block"}.issubset(metadata.columns):
        raise OptimizationInputError("Metadata requires ticker and block")
    if metadata.ticker.duplicated().any() or metadata.block.isna().any():
        raise OptimizationInputError("Metadata contains duplicate tickers or missing blocks")
    names = metadata.block.map(lambda value: str(value).strip())
    if names.eq("").any():
        raise OptimizationInputError("Metadata block cannot be empty")
    counts = {name: int(value) for name, value in names.value_counts().sort_index().items()}
    if not counts:
        raise OptimizationInputError("No available blocks")
    targets = dict(DEFAULT_CONFIG.targets) if set(counts) == set(DEFAULT_CONFIG.targets) else {name: 1 / len(counts) for name in counts}
    return {"blocks": counts, "default_targets": targets, "legacy_targets": dict(DEFAULT_CONFIG.targets)}
