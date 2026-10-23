from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from src.optimization.errors import validate_targets

ModelType = Literal["classic", "ai_returns", "ai_returns_lstm", "ai_volatility"]


class OptimizeRequest(BaseModel):
    model_type: ModelType = Field(..., description="Optimization mode")
    targets: dict[str, float] = Field(..., description="Block targets")
    max_weight: float = Field(0.15, gt=0.0, le=1.0)
    rf_annual_simple: float = Field(0.0, ge=0.0, le=1.0)
    n_frontier_points: int = Field(80, ge=10, le=500)
    min_history: int = Field(252, ge=2)
    min_overlap: int = Field(126, ge=2)
    eigenvalue_floor: float = Field(1e-8, gt=0, lt=1)
    max_psd_adjustment: float = Field(1.0, ge=0, allow_inf_nan=False)
    prediction_fallback: Literal["historical", "required"] = "historical"
    max_prediction_age_days: int = Field(7, ge=0)
    run_label: str | None = Field(default=None)

    @model_validator(mode="after")
    def validate_targets(self) -> "OptimizeRequest":
        validate_targets(self.targets)

        return self


class OptimizeResponse(BaseModel):
    run_id: int | None
    status: str
    message: str
    result: dict
    mlflow_run_id: str | None = None
    mlflow_error: str | None = None