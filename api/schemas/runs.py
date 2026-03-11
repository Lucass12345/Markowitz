from __future__ import annotations

from pydantic import BaseModel


class RunHistoryItem(BaseModel):
    id: int
    created_at: str
    run_label: str | None
    model_type: str
    max_weight: float
    rf_annual_simple: float
    n_frontier_points: int
    return_annual: float | None = None
    vol_annual: float | None = None
    sharpe_annual: float | None = None


class RunDetailResponse(BaseModel):
    run: dict