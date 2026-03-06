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